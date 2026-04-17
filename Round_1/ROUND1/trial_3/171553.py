from datamodel import OrderDepth, TradingState, Order
from typing import Optional
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

OSMIUM_ANCHOR = 10_000

PEPPER_ASK_INTERCEPT = 12_006
PEPPER_BID_INTERCEPT = 11_991
PEPPER_SLOPE = 0.05
PEPPER_SPREAD_THRESHOLD = 7


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

    # ── INTARIAN_PEPPER_ROOT ───────────────────────────────────────────
    #
    # Phase 1 (initial): buy at best ask to accumulate position.
    # Phase 2 (volatility): once position is full, watch for spikes
    #   where bid-ask spread collapses below PEPPER_SPREAD_THRESHOLD.
    #   - If bid > bid_eq  → sell (bid is spiking above equation value)
    #   - If ask < ask_eq  → buy  (ask is dipping below equation value)

    def _trade_pepper(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(PEPPER, 0)
        order_depth: OrderDepth = state.order_depths[PEPPER]
        t = state.timestamp

        ask_eq = PEPPER_ASK_INTERCEPT + PEPPER_SLOPE * t
        bid_eq = PEPPER_BID_INTERCEPT + PEPPER_SLOPE * t

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position

        # Phase 1: accumulate — buy at best ask while we have room
        if max_buy > 0 and best_ask is not None:
            vol = min(abs(order_depth.sell_orders[best_ask]), max_buy)
            if vol > 0:
                orders.append(Order(PEPPER, best_ask, vol))
                max_buy -= vol

        # Phase 2: volatility spike trading
        if best_ask is not None and best_bid is not None:
            spread = best_ask - best_bid
            if spread < PEPPER_SPREAD_THRESHOLD:
                # Bid spiking above its equation value → sell
                if best_bid > bid_eq and max_sell > 0:
                    sell_vol = min(abs(order_depth.buy_orders[best_bid]), max_sell)
                    if sell_vol > 0:
                        orders.append(Order(PEPPER, best_bid, -sell_vol))

                # Ask dipping below its equation value → buy
                if best_ask < ask_eq and max_buy > 0:
                    buy_vol = min(abs(order_depth.sell_orders[best_ask]), max_buy)
                    if buy_vol > 0:
                        orders.append(Order(PEPPER, best_ask, buy_vol))

        return orders

    # ── ASH_COATED_OSMIUM: hard anchor at 10,000 ──────────────────────
    #
    # Simple: buy everything below 10k, sell everything above 10k.

    def _trade_osmium(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(OSMIUM, 0)
        order_depth: OrderDepth = state.order_depths[OSMIUM]

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position
        bought = 0
        sold = 0

        # Buy: sweep all asks at or below the anchor
        if order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price > OSMIUM_ANCHOR:
                    break
                can_buy = min(abs(ask_vol), max_buy - bought)
                if can_buy > 0:
                    orders.append(Order(OSMIUM, ask_price, can_buy))
                    bought += can_buy

        # Sell: hit all bids at or above the anchor
        if order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price < OSMIUM_ANCHOR:
                    break
                can_sell = min(abs(bid_vol), max_sell - sold)
                if can_sell > 0:
                    orders.append(Order(OSMIUM, bid_price, -can_sell))
                    sold += can_sell

        # Passive maker quotes to capture spread around the anchor
        remaining_buy = max_buy - bought
        if remaining_buy > 0:
            orders.append(Order(OSMIUM, OSMIUM_ANCHOR - 1, remaining_buy))

        remaining_sell = max_sell - sold
        if remaining_sell > 0:
            orders.append(Order(OSMIUM, OSMIUM_ANCHOR + 1, -remaining_sell))

        return orders