"""Osmium: bid/ask EMA crossovers after spikes; flatten toward fair / zero.

Separate fast/slow EMAs on best bid and on best ask (not mid). On a sharp move away
from fair we do not trade immediately; we arm a pending direction and only enter when
both sides show the matching crossover (fast crosses above slow for longs after a
down-spike; fast crosses below slow for shorts after an up-spike). When price is back
at fair we offload toward a flat position. Pepper logic matches trader_final.
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

# EMA spans (ticks): fast vs slow on bid and on ask independently
EMA_FAST_N = 8
EMA_SLOW_N = 21

# Mid must clear fair by this much to count as a "spike" and arm pending (no instant fill)
SPIKE_MID_DELTA = 2.0
# If we armed but price settles near fair without a trade, drop the pending signal
SPIKE_CANCEL_DELTA = 0.75

# Cap absolute Osmium inventory (stay closer to zero than full limit)
OSMIUM_MAX_ABS_POS = 45
# After a crossover, keep trying to work the book for this many ticks (liquidity gaps).
CHASE_TICKS = 4


def _alpha(n: int) -> float:
    return 2.0 / (n + 1.0)


_ALPHA_F = _alpha(EMA_FAST_N)
_ALPHA_S = _alpha(EMA_SLOW_N)


class Trader(base.Trader):
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, data)

        return result, 0, self._serialize(data)

    def _trade_osmium(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid = float(best_bid)
        ask = float(best_ask)
        mid = (bid + ask) / 2.0
        fair = float(OSMIUM_FAIR)
        position = state.position.get(OSMIUM, 0)
        cap = OSMIUM_MAX_ABS_POS

        st = data.setdefault(
            "osmium_ema",
            {},
        )

        # Previous-tick EMAs (for crossover detection)
        obf = st.get("bid_fast")
        obs = st.get("bid_slow")
        oaf = st.get("ask_fast")
        oas = st.get("ask_slow")

        if obf is None:
            obf = obs = bid
            oaf = oas = ask

        new_obf = _ALPHA_F * bid + (1.0 - _ALPHA_F) * obf
        new_obs = _ALPHA_S * bid + (1.0 - _ALPHA_S) * obs
        new_oaf = _ALPHA_F * ask + (1.0 - _ALPHA_F) * oaf
        new_oas = _ALPHA_S * ask + (1.0 - _ALPHA_S) * oas

        bull_bid = obf < obs and new_obf >= new_obs
        bear_bid = obf > obs and new_obf <= new_obs
        bull_ask = oaf < oas and new_oaf >= new_oas
        bear_ask = oaf > oas and new_oaf <= new_oas
        bullish_cross = bull_bid and bull_ask
        bearish_cross = bear_bid and bear_ask

        st["bid_fast"] = new_obf
        st["bid_slow"] = new_obs
        st["ask_fast"] = new_oaf
        st["ask_slow"] = new_oas

        pending = st.get("pending")  # None | "long" | "short"
        chase = st.get("chase")

        bought = 0
        sold = 0

        def projected() -> int:
            return position + bought - sold

        # ── Cancel stale pending if market re-centered before entry ───────────
        if pending and position == 0 and abs(mid - fair) < SPIKE_CANCEL_DELTA and not chase:
            st["pending"] = None
            pending = None

        # ── 1) At fair: offload toward zero (priority) ─────────────────────────
        if position < 0 and mid <= fair:
            for ap in sorted(depth.sell_orders.keys()):
                if ap > int(fair):
                    break
                room = POSITION_LIMIT - position - bought
                if room <= 0:
                    break
                av = -depth.sell_orders[ap]
                qty = min(av, room)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(ap), qty))
                    bought += qty

        if position > 0 and mid >= fair:
            for bp in sorted(depth.buy_orders.keys(), reverse=True):
                if bp < int(fair):
                    break
                room = POSITION_LIMIT + position - sold
                if room <= 0:
                    break
                bv = depth.buy_orders[bp]
                qty = min(bv, room)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(bp), -qty))
                    sold += qty

        # Spike alone does not trade — only arm pending. Crossover opens a short chase window.
        if pending == "long" and bullish_cross:
            st["chase"] = "long"
            st["chase_left"] = CHASE_TICKS
            st["pending"] = None
        elif pending == "short" and bearish_cross:
            st["chase"] = "short"
            st["chase_left"] = CHASE_TICKS
            st["pending"] = None

        chase = st.get("chase")
        chase_left = int(st.get("chase_left") or 0)

        # ── 2) Chase window: work entry after crossover (not an instant spike fill) ─
        if chase == "long" and chase_left > 0 and projected() < cap:
            for ap in sorted(depth.sell_orders.keys()):
                if ap > OSMIUM_TAKE_ASK_MAX:
                    break
                room = POSITION_LIMIT - position - bought
                if room <= 0:
                    break
                max_long_left = cap - projected()
                if max_long_left <= 0:
                    break
                av = -depth.sell_orders[ap]
                qty = min(av, room, max_long_left)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(ap), qty))
                    bought += qty

        if chase == "short" and chase_left > 0 and projected() > -cap:
            for bp in sorted(depth.buy_orders.keys(), reverse=True):
                if bp < OSMIUM_TAKE_BID_MIN:
                    break
                room = POSITION_LIMIT + position - sold
                if room <= 0:
                    break
                max_short_left = projected() + cap
                if max_short_left <= 0:
                    break
                bv = depth.buy_orders[bp]
                qty = min(bv, room, max_short_left)
                if qty > 0:
                    orders.append(Order(OSMIUM, int(bp), -qty))
                    sold += qty

        if chase and chase_left > 0:
            st["chase_left"] = chase_left - 1
            if st["chase_left"] <= 0 or abs(projected()) >= cap:
                st["chase"] = None
                st["chase_left"] = 0

        if projected() != 0:
            st["pending"] = None
            st["chase"] = None
            st["chase_left"] = 0

        # ── 3) Arm pending on spike (flat, not in chase) ───────────────────────
        if projected() == 0 and st.get("pending") is None and st.get("chase") is None:
            if mid > fair + SPIKE_MID_DELTA:
                st["pending"] = "short"
            elif mid < fair - SPIKE_MID_DELTA:
                st["pending"] = "long"

        return orders
