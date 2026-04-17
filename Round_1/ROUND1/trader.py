from datamodel import OrderDepth, TradingState, Order
from typing import Optional
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

OSMIUM_ANCHOR = 10_000
OSMIUM_ANCHOR_WEIGHT = 0.5
OSMIUM_EMA_WEIGHT = 0.5

OSMIUM_EMA_FAST_ALPHA = 0.15
OSMIUM_EMA_SLOW_ALPHA = 0.05
OSMIUM_Z_ENTRY = 1.5
OSMIUM_Z_WINDOW = 40


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

        best_ask = min(order_depth.sell_orders.keys())
        vol = min(abs(order_depth.sell_orders[best_ask]), max_buy)
        if vol > 0:
            orders.append(Order(PEPPER, best_ask, vol))

        return orders

    # ── ASH_COATED_OSMIUM: mean-reversion with momentum ───────────────

    def _trade_osmium(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(OSMIUM, 0)
        order_depth: OrderDepth = state.order_depths[OSMIUM]

        mid_price = self._compute_mid(order_depth)
        if mid_price is None:
            return orders

        # ── state: read persisted indicators ──
        ema_fast = trader_state.get("osmium_ema_fast", mid_price)
        ema_slow = trader_state.get("osmium_ema_slow", mid_price)
        history: list[float] = trader_state.get("osmium_history", [])

        # ── update indicators ──
        ema_fast = OSMIUM_EMA_FAST_ALPHA * mid_price + (1 - OSMIUM_EMA_FAST_ALPHA) * ema_fast
        ema_slow = OSMIUM_EMA_SLOW_ALPHA * mid_price + (1 - OSMIUM_EMA_SLOW_ALPHA) * ema_slow

        history.append(mid_price)
        if len(history) > OSMIUM_Z_WINDOW:
            history = history[-OSMIUM_Z_WINDOW:]

        # ── persist state ──
        trader_state["osmium_ema_fast"] = ema_fast
        trader_state["osmium_ema_slow"] = ema_slow
        trader_state["osmium_history"] = history

        # ── z-score ──
        if len(history) >= 2:
            h_mean = sum(history) / len(history)
            h_var = sum((x - h_mean) ** 2 for x in history) / len(history)
            h_std = math.sqrt(h_var) if h_var > 0 else 1e-9
            z_score = (mid_price - h_mean) / h_std
        else:
            z_score = 0.0

        # ── reversal signals ──
        # buy_signal = (z_score < -OSMIUM_Z_ENTRY) and (ema_fast > ema_slow) and (mid_price > ema_fast)
        buy_signal = (ema_fast > ema_slow) and (mid_price > ema_fast)
        # sell_signal = (z_score > OSMIUM_Z_ENTRY) and (ema_fast < ema_slow) and (mid_price < ema_fast)
        sell_signal = (ema_fast < ema_slow) and (mid_price < ema_fast)

        fair_value = OSMIUM_ANCHOR_WEIGHT * OSMIUM_ANCHOR + OSMIUM_EMA_WEIGHT * ema_slow

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position
        bought = 0
        sold = 0

        # ── taking: only when a reversal signal fires ──
        if buy_signal:
            sell_orders = sorted(order_depth.sell_orders.items())
            for ask_price, ask_vol in sell_orders:
                if ask_price > fair_value:
                    break
                can_buy = min(abs(ask_vol), max_buy - bought)
                if can_buy > 0:
                    orders.append(Order(OSMIUM, ask_price, can_buy))
                    bought += can_buy

        if sell_signal:
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            for bid_price, bid_vol in buy_orders:
                if bid_price < fair_value:
                    break
                can_sell = min(abs(bid_vol), max_sell - sold)
                if can_sell > 0:
                    orders.append(Order(OSMIUM, bid_price, -can_sell))
                    sold += can_sell

        # ── making: tight passive quotes every tick ──
        bid_price = int(math.floor(fair_value)) - 1
        ask_price = int(math.ceil(fair_value)) + 1
        if bid_price == ask_price:
            ask_price += 1

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            overbid = best_bid + 1
            if overbid < fair_value:
                bid_price = max(bid_price, overbid)

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            underask = best_ask - 1
            if underask > fair_value:
                ask_price = min(ask_price, underask)

        remaining_buy = max_buy - bought
        if remaining_buy > 0:
            orders.append(Order(OSMIUM, bid_price, remaining_buy))

        remaining_sell = max_sell - sold
        if remaining_sell > 0:
            orders.append(Order(OSMIUM, ask_price, -remaining_sell))

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

