"""Osmium v2: fast-cycling mean-reversion that capitalizes on sustained moves.

Core insight: Osmium reverts around 10000. The spread is ~16 wide.
Profitable fills (ask<=9999 or bid>=10001) appear ~5% of ticks.
Position room (limit=80) is the binding constraint.

Strategy:
1. FLATTEN AT FAIR: exit longs at bid>=10000, shorts at ask<=10000.
2. EDGE-AWARE ENTRIES: the deeper the price, the more aggressively we fill.
   - Small edge (1-2 ticks from fair): only add if position is small.
   - Large edge (3+ ticks from fair): load up toward full limit — the reversion
     profit per unit is high enough to justify holding through the dip.
3. SUSTAINED MOVES: during a 141-tick cheap streak (ask~9995), don't throttle.
   Keep buying every tick at 9995/9996/9997 — each unit profits 3-5 on reversion.
   The old tier system killed this by requiring deeper prices as position grew.

Pepper: unchanged from trader_final.
"""

from __future__ import annotations

from datamodel import Order, OrderDepth, TradingState

import trader_final as base

PEPPER = base.PEPPER
OSMIUM = base.OSMIUM
POSITION_LIMIT = base.POSITION_LIMIT

OSMIUM_FAIR = 10_000

# Exit at fair to cycle fast
EXIT_SELL_MIN = 10_000
EXIT_BUY_MAX = 10_000

# Edge tiers: how far from fair determines how much position room we're willing to use
#   edge 1 (ask=9999/bid=10001): cap position at TIER_1_CAP
#   edge 2 (ask=9998/bid=10002): cap at TIER_2_CAP
#   edge 3+ (ask<=9997/bid>=10003): go up to full POSITION_LIMIT
TIER_1_CAP = 10    # tiny edge => small position
TIER_2_CAP = 20    # moderate edge => moderate position
# edge 3+ => full 80


class Trader(base.Trader):
    """Same pepper, redesigned Osmium."""

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

        # ── PHASE 1: FLATTEN toward zero ─────────────────────────────────────
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

        # ── PHASE 2: NEW ENTRIES (edge-aware, NOT position-throttled) ────────
        # Buy cheap asks: the deeper below fair, the more we're willing to hold
        for ap, neg_av in asks:
            if ap >= OSMIUM_FAIR:
                break
            av = -neg_av
            edge = OSMIUM_FAIR - ap  # how far below fair (positive = good)

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

        # Sell rich bids: symmetric
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
