import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class Trader:
    """
    Combined strategy for Tutorial Round 1:
      - EMERALDS: fixed fair value of 10,000 (stable peg)
      - TOMATOES: simple moving average of last 20 mid-prices (trending product)

    For each product, sweeps all profitable levels in the order book while
    respecting position limits.
    """

    POSITION_LIMITS = {"EMERALDS": 20, "TOMATOES": 20}
    EMERALDS_FAIR = 10_000
    SMA_WINDOW = 20

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if state.traderData:
            saved = json.loads(state.traderData)
        else:
            saved = {"tomatoes_history": []}

        tomatoes_history: list = saved.get("tomatoes_history", [])

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)

            if product == "EMERALDS":
                fair = self.EMERALDS_FAIR

            elif product == "TOMATOES":
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
                current_mid = (best_bid + best_ask) / 2 if best_bid and best_ask else (best_bid or best_ask)
                tomatoes_history.append(current_mid)
                if len(tomatoes_history) > self.SMA_WINDOW:
                    tomatoes_history = tomatoes_history[-self.SMA_WINDOW:]
                fair = sum(tomatoes_history) / len(tomatoes_history)

            else:
                result[product] = []
                continue

            # BUY: sweep asks priced below fair value
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price >= fair:
                    break
                ask_vol = order_depth.sell_orders[ask_price]
                can_buy = limit - position
                if can_buy <= 0:
                    break
                buy_qty = min(-ask_vol, can_buy)
                orders.append(Order(product, ask_price, buy_qty))
                position += buy_qty

            # SELL: sweep bids priced above fair value
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price <= fair:
                    break
                bid_vol = order_depth.buy_orders[bid_price]
                can_sell = limit + position
                if can_sell <= 0:
                    break
                sell_qty = min(bid_vol, can_sell)
                orders.append(Order(product, bid_price, -sell_qty))
                position -= sell_qty

            result[product] = orders

        saved["tomatoes_history"] = tomatoes_history
        traderData = json.dumps(saved)

        return result, 0, traderData
