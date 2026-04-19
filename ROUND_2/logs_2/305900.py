"""Round 2 trader v2: all fixes from live log analysis (305440).

FIXES from trader_1 → trader_2:

1. ADAPTIVE FAIR VALUE (was: hardcoded 10000)
   Live mid averaged 10004, so fair=10000 made us sell constantly and never cover.
   Now: EMA of mid price, updated every tick via traderData. Starts at mid of first
   tick and converges within ~30 ticks. Entries and exits are relative to this
   adaptive fair, so the strategy self-corrects if the market shifts.

2. WIDER EXIT THRESHOLDS (was: exit only at fair ±0)
   In live, ask almost never reached 10000 to cover shorts (0.3% of ticks).
   Now: exit at fair ± EXIT_BUFFER (2 ticks). Accept a small loss per unit on
   exit to free position room → enables more round-trips. The edge on entry
   (3+ ticks) more than covers the 2-tick exit cost.

3. POSITION-GATED ENTRIES (was: kept selling even at -76)
   Accumulating one-sided inventory killed PnL. Now:
   - Skip edge-1 entries entirely (too thin to justify the risk)
   - At edge 2: only add if |position| < 20
   - At edge 3+: only add if |position| < 50 (not full 80)
   - Never add in the SAME direction as existing position beyond the cap

4. MAKER QUOTES RESTORED (was: removed in trader_1)
   In live, passive orders DO get filled. Place buy at fair-3 and sell at fair+3
   with remaining position room. These earn spread and help flatten inventory.

PEPPER: unchanged (works correctly, PnL scales with time).
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── PEPPER parameters ───────────────────────────────────────────────────
PEPPER_SLOPE = 0.001
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# ── OSMIUM parameters ──────────────────────────────────────────────────

# EMA span for adaptive fair value (number of ticks)
FAIR_EMA_N = 30

# Exit: accept up to EXIT_BUFFER ticks of adverse price to free position
EXIT_BUFFER = 1

# Position caps per edge tier (how deep below/above fair)
# Edge 1: skip entirely (too thin, caused the -76 blowup)
# Edge 2: moderate position
# Edge 3+: larger but still capped (not full 80)
EDGE_2_POS_CAP = 30
EDGE_3_POS_CAP = 40

# Maker quote offset from fair (passive orders inside spread)
MAKER_OFFSET = 2


def _ema_alpha(n: int) -> float:
    return 2.0 / (n + 1.0)


_FAIR_ALPHA = _ema_alpha(FAIR_EMA_N)


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, data)

        return result, 0, self._serialize(data)

    # ── state helpers ────────────────────────────────────────────────────

    def _deserialize(self, raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _serialize(self, data: dict) -> str:
        try:
            return json.dumps(data)
        except (TypeError, ValueError):
            return ""

    # ── INTARIAN_PEPPER_ROOT: aggressive trend-follow to +80 ─────────────

    def _trade_pepper(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[PEPPER]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        if "pepper_init_mid" not in data:
            data["pepper_init_mid"] = (best_bid + best_ask) / 2.0

        fair = data["pepper_init_mid"] + PEPPER_SLOPE * state.timestamp
        position = state.position.get(PEPPER, 0)

        bought = 0
        for ask in sorted(depth.sell_orders.keys()):
            if ask > fair + PEPPER_TAKE_BUFFER:
                break
            room = POSITION_LIMIT - position - bought
            if room <= 0:
                break
            ask_vol = -depth.sell_orders[ask]
            qty = min(ask_vol, room)
            if qty > 0:
                orders.append(Order(PEPPER, int(ask), qty))
                bought += qty

        room = POSITION_LIMIT - position - bought
        if room > 0:
            orders.append(Order(PEPPER, int(math.floor(fair)), room))

        net_long = position + bought
        if net_long > 0:
            orders.append(
                Order(PEPPER, int(math.ceil(fair)) + PEPPER_OUTLIER_EDGE, -net_long)
            )

        return orders

    # ── ASH_COATED_OSMIUM: adaptive fair + position-gated + makers ───────

    def _trade_osmium(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        # ── FIX 1: Adaptive fair value via EMA ──────────────────────────
        prev_fair = data.get("osmium_fair")
        if prev_fair is None:
            fair = mid
        else:
            fair = _FAIR_ALPHA * mid + (1.0 - _FAIR_ALPHA) * prev_fair
        data["osmium_fair"] = fair

        fair_int = int(round(fair))

        position = state.position.get(OSMIUM, 0)
        bought = 0
        sold = 0

        def proj():
            return position + bought - sold

        asks = sorted(depth.sell_orders.items())
        bids = sorted(depth.buy_orders.items(), reverse=True)

        # ── FIX 2: FLATTEN toward zero with wider exit thresholds ────────
        exit_buy_max = fair_int + EXIT_BUFFER   # cover shorts at ask <= fair+2
        exit_sell_min = fair_int - EXIT_BUFFER  # flatten longs at bid >= fair-2

        if position > 0:
            for bp, bv in bids:
                if bp < exit_sell_min:
                    break
                can_sell = min(bv, POSITION_LIMIT + position - sold, proj())
                if can_sell <= 0:
                    continue
                orders.append(Order(OSMIUM, int(bp), -can_sell))
                sold += can_sell
                if proj() <= 0:
                    break

        if position < 0:
            for ap, neg_av in asks:
                if ap > exit_buy_max:
                    break
                av = -neg_av
                can_buy = min(av, POSITION_LIMIT - position - bought, -proj())
                if can_buy <= 0:
                    continue
                orders.append(Order(OSMIUM, int(ap), can_buy))
                bought += can_buy
                if proj() >= 0:
                    break

        # ── FIX 3: Position-gated entries (edge-aware, direction-aware) ──
        # Only enter if edge justifies the position risk.
        # Edge 1: SKIP (caused the -76 blowup in live — too thin)
        # Edge 2: add up to EDGE_2_POS_CAP
        # Edge 3+: add up to EDGE_3_POS_CAP

        # Buy cheap asks (below fair)
        for ap, neg_av in asks:
            if ap >= fair_int:
                break
            av = -neg_av
            edge = fair_int - ap

            if edge >= 3:
                pos_cap = EDGE_3_POS_CAP
            elif edge == 2:
                pos_cap = EDGE_2_POS_CAP
            else:
                continue  # skip edge 1

            cur = proj()
            if cur >= pos_cap:
                continue
            room = min(pos_cap - cur, POSITION_LIMIT - cur)
            if room <= 0:
                continue
            qty = min(av, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(ap), qty))
                bought += qty

        # Sell rich bids (above fair)
        for bp, bv in bids:
            if bp <= fair_int:
                break
            edge = bp - fair_int

            if edge >= 3:
                pos_cap = EDGE_3_POS_CAP
            elif edge == 2:
                pos_cap = EDGE_2_POS_CAP
            else:
                continue  # skip edge 1

            cur = proj()
            if -cur >= pos_cap:
                continue
            room = min(pos_cap + cur, POSITION_LIMIT + cur)
            if room <= 0:
                continue
            qty = min(bv, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(bp), -qty))
                sold += qty

        # ── FIX 4: Maker quotes (passive orders inside spread) ──────────
        maker_buy_px = fair_int - MAKER_OFFSET
        maker_sell_px = fair_int + MAKER_OFFSET

        # Clamp to not cross the book
        maker_buy_px = min(maker_buy_px, best_bid)
        maker_sell_px = max(maker_sell_px, best_ask)

        buy_room = POSITION_LIMIT - proj()
        sell_room = POSITION_LIMIT + proj()

        if buy_room > 0:
            orders.append(Order(OSMIUM, int(maker_buy_px), buy_room))
        if sell_room > 0:
            orders.append(Order(OSMIUM, int(maker_sell_px), -sell_room))

        return orders