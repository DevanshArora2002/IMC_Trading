from datamodel import OrderDepth, TradingState, Order
from typing import Optional
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── OSMIUM tunables (momentum + reversal detection) ──────────────────
OSMIUM_ANCHOR = 10_000
OSMIUM_EMA_FAST_ALPHA = 0.5
OSMIUM_EMA_SLOW_ALPHA = 0.2

# Slope tracking: we store the last N ema_fast values to compute slope
OSMIUM_SLOPE_WINDOW = 5          # ticks of ema_fast history for slope calc
OSMIUM_REVERSAL_CONFIRM = 3      # consecutive ticks of opposing slope to confirm reversal

# Inventory skew for making quotes (same as trader_6)
OSMIUM_INV_SKEW_K = 2.0


class Trader:

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        trader_state = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, trader_state)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, trader_state)

        trader_data_str = self._serialize(trader_state)
        return result, conversions, trader_data_str

    # ── state persistence ──────────────────────────────────────────────

    def _deserialize(self, data: str) -> dict:
        if data and data.strip():
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _serialize(self, trader_state: dict) -> str:
        try:
            return json.dumps(trader_state)
        except (TypeError, ValueError):
            return ""

    # ── INTARIAN_PEPPER_ROOT: buy-and-hold ─────────────────────────────

    def _trade_pepper(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(PEPPER, 0)
        order_depth: OrderDepth = state.order_depths[PEPPER]

        max_buy = POSITION_LIMIT - position
        if max_buy <= 0 or not order_depth.sell_orders:
            return orders

        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            vol = abs(ask_vol)
            if vol > 0 and max_buy > 0:
                buy_vol = min(vol, max_buy)
                orders.append(Order(PEPPER, ask_price, buy_vol))
                max_buy -= buy_vol
                if max_buy <= 0:
                    break

        return orders

    # ── ASH_COATED_OSMIUM: directional momentum + reversal flipping ───

    def _trade_osmium(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(OSMIUM, 0)
        order_depth: OrderDepth = state.order_depths[OSMIUM]

        mid_price = self._compute_mid(order_depth)
        if mid_price is None:
            return orders

        # ── read persisted state ──
        ema_fast = trader_state.get("osmium_ema_fast", mid_price)
        ema_slow = trader_state.get("osmium_ema_slow", mid_price)
        tick = int(trader_state.get("osmium_tick", 0))
        ema_fast_history: list[float] = trader_state.get("osmium_ema_fast_hist", [])
        # regime: "short" = we are actively shorting the initial drop
        #         "long"  = we flipped to buying after a reversal from short
        #         "neutral" = waiting / no strong directional bias
        regime: str = trader_state.get("osmium_regime", "init")
        reversal_counter: int = int(trader_state.get("osmium_reversal_ctr", 0))

        # ── update indicators ──
        prev_ema_fast = ema_fast
        ema_fast = OSMIUM_EMA_FAST_ALPHA * mid_price + (1 - OSMIUM_EMA_FAST_ALPHA) * ema_fast
        ema_slow = OSMIUM_EMA_SLOW_ALPHA * mid_price + (1 - OSMIUM_EMA_SLOW_ALPHA) * ema_slow
        tick += 1

        # track ema_fast history for slope detection
        ema_fast_history.append(ema_fast)
        if len(ema_fast_history) > OSMIUM_SLOPE_WINDOW + 2:
            ema_fast_history = ema_fast_history[-(OSMIUM_SLOPE_WINDOW + 2):]

        # ── compute ema_fast slope (delta over last few ticks) ──
        ema_delta = ema_fast - prev_ema_fast  # instantaneous slope

        # slope sign over recent ticks: positive = rising, negative = falling
        recent_deltas = []
        if len(ema_fast_history) >= 2:
            for i in range(1, len(ema_fast_history)):
                recent_deltas.append(ema_fast_history[i] - ema_fast_history[i - 1])

        # count consecutive positive / negative deltas at the tail
        consec_positive = 0
        consec_negative = 0
        for d in reversed(recent_deltas):
            if d > 0:
                consec_positive += 1
            else:
                break
        for d in reversed(recent_deltas):
            if d < 0:
                consec_negative += 1
            else:
                break

        # ── regime transitions ──
        if regime == "init":
            if mid_price > OSMIUM_ANCHOR:
                regime = "short"
            else:
                regime = "long"

        if regime == "short":
            # looking for reversal: ema_fast was falling, now has 3+ ticks positive slope
            if consec_positive >= OSMIUM_REVERSAL_CONFIRM:
                regime = "long"
                reversal_counter += 1
        elif regime == "long":
            # looking for reversal: ema_fast was rising, now has 3+ ticks negative slope
            if consec_negative >= OSMIUM_REVERSAL_CONFIRM:
                regime = "short"
                reversal_counter += 1

        # ── persist state ──
        trader_state["osmium_ema_fast"] = ema_fast
        trader_state["osmium_ema_slow"] = ema_slow
        trader_state["osmium_tick"] = tick
        trader_state["osmium_ema_fast_hist"] = ema_fast_history
        trader_state["osmium_regime"] = regime
        trader_state["osmium_reversal_ctr"] = reversal_counter

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position
        bought = 0
        sold = 0

        # ── PHASE: first tick (no history at all) → if above 10k, start shorting ──
        if tick == 1 and mid_price > OSMIUM_ANCHOR:
            return self._aggressive_sell(order_depth, position, max_sell)

        # ── REGIME: SHORT — aggressively sell, passively make on the sell side ──
        if regime == "short":
            # aggressive sell: hit every bid available
            buy_orders_sorted = sorted(order_depth.buy_orders.items(), reverse=True)
            for bid_price, bid_vol in buy_orders_sorted:
                can_sell = min(abs(bid_vol), max_sell - sold)
                if can_sell > 0:
                    orders.append(Order(OSMIUM, bid_price, -can_sell))
                    sold += can_sell
                if sold >= max_sell:
                    break

            # passive sell: post remaining at best_ask - 1 or mid
            remaining_sell = max_sell - sold
            if remaining_sell > 0:
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    ask_px = best_ask - 1
                else:
                    ask_px = int(math.ceil(mid_price))
                ask_px = max(ask_px, int(math.ceil(mid_price)) - 1)
                orders.append(Order(OSMIUM, ask_px, -remaining_sell))

            # if we have room to buy, post a defensive bid far below to catch panics
            remaining_buy = max_buy - bought
            if remaining_buy > 0:
                defensive_bid = int(math.floor(mid_price)) - 4
                orders.append(Order(OSMIUM, defensive_bid, remaining_buy))

            return orders

        # ── REGIME: LONG — aggressively buy, passively make on the buy side ──
        if regime == "long":
            # aggressive buy: lift every ask available
            sell_orders_sorted = sorted(order_depth.sell_orders.items())
            for ask_price, ask_vol in sell_orders_sorted:
                can_buy = min(abs(ask_vol), max_buy - bought)
                if can_buy > 0:
                    orders.append(Order(OSMIUM, ask_price, can_buy))
                    bought += can_buy
                if bought >= max_buy:
                    break

            # passive buy: post remaining at best_bid + 1 or mid
            remaining_buy = max_buy - bought
            if remaining_buy > 0:
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    bid_px = best_bid + 1
                else:
                    bid_px = int(math.floor(mid_price))
                bid_px = min(bid_px, int(math.floor(mid_price)) + 1)
                orders.append(Order(OSMIUM, bid_px, remaining_buy))

            # defensive ask far above to catch spikes
            remaining_sell = max_sell - sold
            if remaining_sell > 0:
                defensive_ask = int(math.ceil(mid_price)) + 4
                orders.append(Order(OSMIUM, defensive_ask, -remaining_sell))

            return orders

        # fallback: neutral market-make around mid (should rarely hit)
        return self._neutral_make(order_depth, mid_price, position, max_buy, max_sell)

    def _aggressive_sell(self, order_depth: OrderDepth, position: int,
                         max_sell: int) -> list[Order]:
        """Hit all bids + post aggressive ask to reach short limit ASAP."""
        orders: list[Order] = []
        sold = 0

        buy_orders_sorted = sorted(order_depth.buy_orders.items(), reverse=True)
        for bid_price, bid_vol in buy_orders_sorted:
            can_sell = min(abs(bid_vol), max_sell - sold)
            if can_sell > 0:
                orders.append(Order(OSMIUM, bid_price, -can_sell))
                sold += can_sell
            if sold >= max_sell:
                break

        remaining = max_sell - sold
        if remaining > 0:
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_px = best_ask - 1
            elif order_depth.buy_orders:
                ask_px = max(order_depth.buy_orders.keys())
            else:
                ask_px = OSMIUM_ANCHOR
            orders.append(Order(OSMIUM, ask_px, -remaining))

        return orders

    def _neutral_make(self, order_depth: OrderDepth, mid: float,
                      position: int, max_buy: int, max_sell: int) -> list[Order]:
        """Fallback: tight quotes around mid with inventory skew."""
        orders: list[Order] = []
        ref = mid - OSMIUM_INV_SKEW_K * (position / POSITION_LIMIT)

        bid_px = int(math.floor(ref)) - 1
        ask_px = int(math.ceil(ref)) + 1
        if bid_px >= ask_px:
            ask_px = bid_px + 2

        if max_buy > 0:
            orders.append(Order(OSMIUM, bid_px, max_buy))
        if max_sell > 0:
            orders.append(Order(OSMIUM, ask_px, -max_sell))

        return orders

    def _compute_mid(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None