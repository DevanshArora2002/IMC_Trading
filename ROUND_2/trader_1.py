"""Round 2 trader: combined Pepper trend-follow + Osmium v2 edge-aware mean-reversion.

Self-contained — no imports from Round 1. Submittable as-is.

PEPPER (INTARIAN_PEPPER_ROOT):
  Linear drift ~+0.001/tick anchored on first-tick mid.
  Aggressively buy asks up to fair + buffer, passive bid at fair, outlier sell above.

OSMIUM (ASH_COATED_OSMIUM):
  Mean-reverts tightly around 10_000 with ~16-wide spread.
  v2 edge-aware strategy:
  1. FLATTEN AT FAIR: exit longs at bid>=10000, shorts at ask<=10000.
  2. EDGE-AWARE ENTRIES: deeper price => more aggressive position building.
     - Edge 1 (9999/10001): cap at 10 lots
     - Edge 2 (9998/10002): cap at 20 lots
     - Edge 3+ (<=9997/>=10003): fill up to full 80 (sustained moves)
  3. Always flatten before adding new risk.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math

# ── Product names ────────────────────────────────────────────────────────
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── PEPPER parameters ───────────────────────────────────────────────────
PEPPER_SLOPE = 0.001
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# ── OSMIUM parameters ──────────────────────────────────────────────────
OSMIUM_FAIR = 10_000

EXIT_SELL_MIN = 10_000
EXIT_BUY_MAX = 10_000

TIER_1_CAP = 10
TIER_2_CAP = 20


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state)

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

    # ── ASH_COATED_OSMIUM: edge-aware mean-reversion ─────────────────────

    def _trade_osmium(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        position = state.position.get(OSMIUM, 0)
        bought = 0
        sold = 0

        def proj():
            return position + bought - sold

        asks = sorted(depth.sell_orders.items())
        bids = sorted(depth.buy_orders.items(), reverse=True)

        # ── PHASE 1: FLATTEN toward zero ─────────────────────────────────
        if position > 0:
            for bp, bv in bids:
                if bp < EXIT_SELL_MIN:
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
                if ap > EXIT_BUY_MAX:
                    break
                av = -neg_av
                can_buy = min(av, POSITION_LIMIT - position - bought, -proj())
                if can_buy <= 0:
                    continue
                orders.append(Order(OSMIUM, int(ap), can_buy))
                bought += can_buy
                if proj() >= 0:
                    break

        # ── PHASE 2: NEW ENTRIES (edge-aware) ────────────────────────────
        for ap, neg_av in asks:
            if ap >= OSMIUM_FAIR:
                break
            av = -neg_av
            edge = OSMIUM_FAIR - ap

            if edge >= 3:
                pos_cap = POSITION_LIMIT
            elif edge == 2:
                pos_cap = TIER_2_CAP
            elif edge == 1:
                pos_cap = TIER_1_CAP
            else:
                continue

            room = min(pos_cap - proj(), POSITION_LIMIT - proj())
            if room <= 0:
                continue
            qty = min(av, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(ap), qty))
                bought += qty

        for bp, bv in bids:
            if bp <= OSMIUM_FAIR:
                break
            edge = bp - OSMIUM_FAIR

            if edge >= 3:
                pos_cap = POSITION_LIMIT
            elif edge == 2:
                pos_cap = TIER_2_CAP
            elif edge == 1:
                pos_cap = TIER_1_CAP
            else:
                continue

            room = min(pos_cap + proj(), POSITION_LIMIT + proj())
            if room <= 0:
                continue
            qty = min(bv, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(bp), -qty))
                sold += qty

        return orders
