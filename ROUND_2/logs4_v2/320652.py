"""Round 2 trader v4: pure maker (no Osmium take) + bid().

WHY v4 — diagnosis from log 306699 (trader_3)
─────────────────────────────────────────────
On the live 306699 day, trader_3's Osmium fills broke down per role as:

  MAKER_BUY   17 fills, 85 lots,  avg_edge_vs_mid = +7.09  →  intent +602
  MAKER_SELL  23 fills, 102 lots, avg_edge_vs_mid = +7.02  →  intent +720
  TAKER_BUY    1 fill,   8 lots,  avg_edge_vs_mid = -3.00  →  intent  -24

40 / 41 fills were profitable. The single TAKER_BUY at ts=85700 lifted
a fleeting ask of 10001 (L1) while L2 sat at 10011 — classic adverse
selection on phantom liquidity. The book one tick later snapped back
to ask=10011, immediately giving up the move.

Conclusion: the take loop is a small but consistently negative-EV
sub-strategy in this regime (median spread 16, never-tightening). The
maker pillar alone produced +1,322 on Osmium that day; deleting the
take loop would have lifted the day to +1,346.

CHANGES v3 → v4
───────────────
1. **Take loop REMOVED for Osmium.** Maker quotes only.
   Eliminates the adverse-selection lottery on flash dislocations.

2. Pepper logic restored to the proven trader_1/trader_3 implementation
   (`int(math.floor(fair))` resting bid, no half-finished history
   feature).

3. Penny-jump maker, anti-cross guard, inventory skew, and `bid()`
   method are all kept exactly as in trader_3.

If you ever want a take fallback for genuine deep dislocations, add a
gap guard (`ask_price_2 - ask_price_1 > 4` ⇒ skip) — but the data shows
this is rare enough that it's not worth the risk.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math

# ── Product names ────────────────────────────────────────────────────────
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── PEPPER parameters (unchanged) ───────────────────────────────────────
PEPPER_SLOPE = 0.001
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# ── OSMIUM parameters ──────────────────────────────────────────────────
FAIR_EMA_N = 30
FAIR_ALPHA = 2.0 / (FAIR_EMA_N + 1.0)

# Maker offsets — penny-jump = +/-1 from BBO
MAKER_PENNY = 1.5

# Inventory skew: shift quotes by position // SKEW_DIV.
# Smaller divisor ⇒ more aggressive flatten via maker.
SKEW_DIV = 12

# Quote sizing — never quote more than this on either side.
MAX_MAKE_SIZE = 80

# When spread is tighter than this, fall back to fair-relative quotes
NARROW_SPREAD = 4

# ── BID for Round-2 side mechanic ──────────────────────────────────────
BID_VALUE = 45


class Trader:
    # ── Round-2 sealed bid ────────────────────────────────────────────
    def bid(self) -> int:
        """Required by Round 2. Returns an integer bid value used by the
        side mechanic. Ignored in all other rounds."""
        return BID_VALUE

    # ── main entry ────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, data)

        return result, 0, self._serialize(data)

    # ── state helpers ─────────────────────────────────────────────────
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

    # ── INTARIAN_PEPPER_ROOT (proven trader_1/trader_3 logic) ─────────
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

    # ── ASH_COATED_OSMIUM: pure penny-jump maker (no take) ────────────
    def _trade_osmium(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        # adaptive fair (EMA of mid)
        prev = data.get("osmium_fair")
        fair = mid if prev is None else FAIR_ALPHA * mid + (1.0 - FAIR_ALPHA) * prev
        data["osmium_fair"] = fair
        fair_int = int(round(fair))

        position = state.position.get(OSMIUM, 0)

        # Penny-jump maker quotes (with inventory skew)
        skew = position // SKEW_DIV  # long ⇒ shift quotes down

        if spread <= NARROW_SPREAD:
            bid_px = fair_int - 1 - skew
            ask_px = fair_int + 1 - skew
        else:
            bid_px = best_bid + MAKER_PENNY - skew
            ask_px = best_ask - MAKER_PENNY - skew

        # Anti-cross guard: never quote across the fair value
        bid_px = min(bid_px, fair_int - 1)
        ask_px = max(ask_px, fair_int + 1)

        # Anti-self-cross
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        # Position-scaled sizing
        buy_room = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        buy_size = min(buy_room, MAX_MAKE_SIZE)
        sell_size = min(sell_room, MAX_MAKE_SIZE)

        if buy_size > 0:
            orders.append(Order(OSMIUM, int(bid_px), buy_size))
        if sell_size > 0:
            orders.append(Order(OSMIUM, int(ask_px), -sell_size))

        return orders