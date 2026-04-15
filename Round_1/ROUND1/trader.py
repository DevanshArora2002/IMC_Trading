from datamodel import OrderDepth, TradingState, Order
from typing import Optional
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

OSMIUM_ANCHOR = 10_000
OSMIUM_EMA_ALPHA = 0.05
OSMIUM_ANCHOR_WEIGHT = 0.5
OSMIUM_EMA_WEIGHT = 0.5
OSMIUM_HISTORY_WINDOW = 100


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
        if max_buy <= 0:
            return orders

        bought = 0

        sell_orders = sorted(order_depth.sell_orders.items())
        for ask_price, ask_vol in sell_orders:
            if bought >= max_buy:
                break
            can_buy = min(abs(ask_vol), max_buy - bought)
            if can_buy > 0:
                orders.append(Order(PEPPER, ask_price, can_buy))
                bought += can_buy

        remaining = max_buy - bought
        if remaining > 0:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is not None and best_ask is not None:
                passive_price = best_ask - 1
            elif best_bid is not None:
                passive_price = best_bid + 1
            else:
                passive_price = 10_000

            orders.append(Order(PEPPER, passive_price, remaining))

        return orders

    # ── ASH_COATED_OSMIUM: mean-reversion with momentum ───────────────

    def _trade_osmium(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(OSMIUM, 0)
        order_depth: OrderDepth = state.order_depths[OSMIUM]

        mid_price = self._compute_mid(order_depth)
        if mid_price is None:
            return orders

        ema = trader_state.get("osmium_ema", mid_price)
        history = trader_state.get("osmium_history", [])

        ema = OSMIUM_EMA_ALPHA * mid_price + (1 - OSMIUM_EMA_ALPHA) * ema
        history.append(mid_price)
        if len(history) > OSMIUM_HISTORY_WINDOW:
            history = history[-OSMIUM_HISTORY_WINDOW:]

        trader_state["osmium_ema"] = ema
        trader_state["osmium_history"] = history

        fair_value = OSMIUM_ANCHOR_WEIGHT * OSMIUM_ANCHOR + OSMIUM_EMA_WEIGHT * ema

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position
        bought = 0
        sold = 0

        # ── taking: sweep all levels below/above fair value ──
        sell_orders = sorted(order_depth.sell_orders.items())
        for ask_price, ask_vol in sell_orders:
            if ask_price > fair_value - 1:
                if ask_price <= fair_value and position < 0:
                    vol = min(abs(ask_vol), max_buy - bought, abs(position) - bought)
                    if vol > 0:
                        orders.append(Order(OSMIUM, ask_price, vol))
                        bought += vol
                break
            can_buy = min(abs(ask_vol), max_buy - bought)
            if can_buy > 0:
                orders.append(Order(OSMIUM, ask_price, can_buy))
                bought += can_buy

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        for bid_price, bid_vol in buy_orders:
            if bid_price < fair_value + 1:
                if bid_price >= fair_value and position > 0:
                    vol = min(abs(bid_vol), max_sell - sold, position - sold)
                    if vol > 0:
                        orders.append(Order(OSMIUM, bid_price, -vol))
                        sold += vol
                break
            can_sell = min(abs(bid_vol), max_sell - sold)
            if can_sell > 0:
                orders.append(Order(OSMIUM, bid_price, -can_sell))
                sold += can_sell

        # ── making: tight market around fair value ──
        bid_price = int(math.floor(fair_value)) - 1
        ask_price = int(math.ceil(fair_value)) + 1
        if bid_price == ask_price:
            ask_price += 1

        # Overbid/underbid if profitable
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

