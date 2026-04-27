"""
Round 3 VELVETFRUIT_EXTRACT compound strategy.

Regime 1, range-bound market:
    If the current BBO mid is inside [5220, 5260], use 5240 as fair value.
    Take clearly mispriced liquidity, then make around fair with inventory skew.

Regime 2, outside range:
    Stop quoting both sides and only take directional exposure from an SMA signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_R2 = Path(__file__).resolve().parent.parent / "ROUND_2"
if str(_R2) not in sys.path:
    sys.path.insert(0, str(_R2))

from datamodel import Order, OrderDepth, TradingState  # noqa: E402


VELVET = "VELVETFRUIT_EXTRACT"
POSITION_LIMIT = 200
BID_VALUE = 0

RANGE_LOW = 5220
RANGE_HIGH = 5260
RANGE_FAIR = 5240.0

FAST_WINDOW = 50
SLOW_WINDOW = 200
HISTORY_LIMIT = SLOW_WINDOW + 5

TAKE_EDGE = 2
MAKE_EDGE = 1
INVENTORY_SKEW = 0.02


class Trader:
    def bid(self) -> int:  # noqa: A001  # Prosperity interface
        return BID_VALUE

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if VELVET not in state.order_depths:
            return result, 0, self._serialize(data)

        depth: OrderDepth = state.order_depths[VELVET]
        if not depth.buy_orders or not depth.sell_orders:
            return result, 0, self._serialize(data)

        position = state.position.get(VELVET, 0)
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid = (best_bid + best_ask) / 2.0

        history = data.get("velvet_mid_history", [])
        history.append(mid)
        data["velvet_mid_history"] = history[-HISTORY_LIMIT:]

        if RANGE_LOW <= mid <= RANGE_HIGH:
            orders = self._trade_range(depth, position)
        else:
            orders = self._trade_directional(depth, position, data["velvet_mid_history"])

        if orders:
            result[VELVET] = orders

        return result, 0, self._serialize(data)

    def _trade_range(self, depth: OrderDepth, position: int) -> list[Order]:
        orders: list[Order] = []
        bought = 0
        sold = 0

        fair = RANGE_FAIR - INVENTORY_SKEW * position
        buy_room = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        # Take liquidity only when it is far enough from fair to pay the spread.
        for ask, volume in sorted(depth.sell_orders.items()):
            if ask > fair - TAKE_EDGE or buy_room <= 0:
                break
            qty = min(-volume, buy_room)
            if qty > 0:
                orders.append(Order(VELVET, int(ask), qty))
                bought += qty
                buy_room -= qty

        for bid, volume in sorted(depth.buy_orders.items(), reverse=True):
            if bid < fair + TAKE_EDGE or sell_room <= 0:
                break
            qty = min(volume, sell_room)
            if qty > 0:
                orders.append(Order(VELVET, int(bid), -qty))
                sold += qty
                sell_room -= qty

        post_position = position + bought - sold
        buy_room = POSITION_LIMIT - post_position
        sell_room = POSITION_LIMIT + post_position

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        bid_price = min(int(fair) - MAKE_EDGE, best_ask - 1)
        if best_bid + 1 < fair:
            bid_price = max(bid_price, best_bid + 1)

        ask_price = max(int(fair) + MAKE_EDGE, best_bid + 1)
        if best_ask - 1 > fair:
            ask_price = min(ask_price, best_ask - 1)

        if bid_price < ask_price:
            if buy_room > 0:
                orders.append(Order(VELVET, int(bid_price), buy_room))
            if sell_room > 0:
                orders.append(Order(VELVET, int(ask_price), -sell_room))

        return orders

    def _trade_directional(
        self, depth: OrderDepth, position: int, history: list[float]
    ) -> list[Order]:
        if len(history) < SLOW_WINDOW:
            return []

        fast_sma = sum(history[-FAST_WINDOW:]) / FAST_WINDOW
        slow_sma = sum(history[-SLOW_WINDOW:]) / SLOW_WINDOW

        if fast_sma > slow_sma:
            target_position = POSITION_LIMIT
        elif fast_sma < slow_sma:
            target_position = -POSITION_LIMIT
        else:
            target_position = position

        delta = target_position - position
        if delta > 0:
            return self._buy_to_target(depth, delta)
        if delta < 0:
            return self._sell_to_target(depth, -delta)
        return []

    def _buy_to_target(self, depth: OrderDepth, quantity: int) -> list[Order]:
        orders: list[Order] = []
        remaining = quantity
        for ask, volume in sorted(depth.sell_orders.items()):
            if remaining <= 0:
                break
            qty = min(-volume, remaining)
            if qty > 0:
                orders.append(Order(VELVET, int(ask), qty))
                remaining -= qty
        return orders

    def _sell_to_target(self, depth: OrderDepth, quantity: int) -> list[Order]:
        orders: list[Order] = []
        remaining = quantity
        for bid, volume in sorted(depth.buy_orders.items(), reverse=True):
            if remaining <= 0:
                break
            qty = min(volume, remaining)
            if qty > 0:
                orders.append(Order(VELVET, int(bid), -qty))
                remaining -= qty
        return orders

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
