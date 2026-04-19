"""Osmium: spike mean-reversion around fair (same pepper logic as trader_final).

Fair value is fixed at 10_000 (see basics / Round 2 notes). On upward excursions
(mid above fair + entry band), lean short by lifting bids >= fair+1; when price
returns to fair or below, cover by lifting asks at or below fair. Symmetrically for
downward spikes. Inventory is capped below POSITION_LIMIT to stay near flat on average.
"""

from __future__ import annotations

from datamodel import Order, OrderDepth, TradingState

import trader_final as base

PEPPER = base.PEPPER
OSMIUM = base.OSMIUM
POSITION_LIMIT = base.POSITION_LIMIT

PEPPER_SLOPE = base.PEPPER_SLOPE
PEPPER_TAKE_BUFFER = base.PEPPER_TAKE_BUFFER
PEPPER_OUTLIER_EDGE = base.PEPPER_OUTLIER_EDGE

OSMIUM_FAIR = base.OSMIUM_FAIR
OSMIUM_TAKE_ASK_MAX = base.OSMIUM_TAKE_ASK_MAX
OSMIUM_TAKE_BID_MIN = base.OSMIUM_TAKE_BID_MIN

# Spike detection: mid must clear fair by this amount (half-spread-scale) to add risk.
OSMIUM_SPIKE_ENTRY = 1.0
# Max absolute Osmium position — keeps exposure closer to zero than full limit.
OSMIUM_MAX_ABS_POS = 45


class Trader(base.Trader):
    """Identical to trader_final except _trade_osmium (spike mean-reversion)."""

    def _trade_osmium(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        fair = float(OSMIUM_FAIR)
        position = state.position.get(OSMIUM, 0)
        cap = OSMIUM_MAX_ABS_POS

        bought = 0
        sold = 0

        def projected() -> int:
            return position + bought - sold

        # ── 1) Cover shorts when price has come back to fair or below ─────────
        if position < 0 and mid <= fair:
            for ask in sorted(depth.sell_orders.keys()):
                if ask > int(fair):
                    break
                room = POSITION_LIMIT - position - bought
                if room <= 0:
                    break
                ask_vol = -depth.sell_orders[ask]
                qty = min(ask_vol, room)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(ask), qty))
                    bought += qty

        # ── 2) Flatten longs when price is back at fair or above ─────────────
        if position > 0 and mid >= fair:
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if bid < int(fair):
                    break
                room = POSITION_LIMIT + position - sold
                if room <= 0:
                    break
                bid_vol = depth.buy_orders[bid]
                qty = min(bid_vol, room)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(bid), -qty))
                    sold += qty

        # ── 3) Add to short on upward spike (price rich vs fair) ─────────────
        if mid > fair + OSMIUM_SPIKE_ENTRY and projected() > -cap:
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if bid < OSMIUM_TAKE_BID_MIN:
                    break
                room = POSITION_LIMIT + position - sold
                if room <= 0:
                    break
                # Stay within inventory cap (more negative than -cap forbidden)
                max_short_left = projected() + cap
                if max_short_left <= 0:
                    break
                bid_vol = depth.buy_orders[bid]
                qty = min(bid_vol, room, max_short_left)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(bid), -qty))
                    sold += qty

        # ── 4) Add to long on downward spike (price cheap vs fair) ───────────
        if mid < fair - OSMIUM_SPIKE_ENTRY and projected() < cap:
            for ask in sorted(depth.sell_orders.keys()):
                if ask > OSMIUM_TAKE_ASK_MAX:
                    break
                room = POSITION_LIMIT - position - bought
                if room <= 0:
                    break
                max_long_left = cap - projected()
                if max_long_left <= 0:
                    break
                ask_vol = -depth.sell_orders[ask]
                qty = min(ask_vol, room, max_long_left)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(ask), qty))
                    bought += qty

        return orders
