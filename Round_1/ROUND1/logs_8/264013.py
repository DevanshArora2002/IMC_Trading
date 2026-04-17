from datamodel import OrderDepth, TradingState, Order
import json

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

OSMIUM_ANCHOR = 10_000


class Trader:

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        trader_state = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, trader_state)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state)

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

    # ── ASH_COATED_OSMIUM: full-book sweep + pegged makers ────────────

    def _trade_osmium(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(OSMIUM, 0)
        order_depth: OrderDepth = state.order_depths[OSMIUM]

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position

        bought = 0
        sold = 0

        # Buy every displayed ask strictly below 10_000 (walk the book)
        if max_buy > 0 and order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price >= OSMIUM_ANCHOR:
                    break
                can_buy = min(abs(ask_vol), max_buy - bought)
                if can_buy > 0:
                    orders.append(Order(OSMIUM, ask_price, can_buy))
                    bought += can_buy
                if bought >= max_buy:
                    break

        # Sell into every displayed bid strictly above 10_000
        if max_sell > 0 and order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price <= OSMIUM_ANCHOR:
                    break
                can_sell = min(abs(bid_vol), max_sell - sold)
                if can_sell > 0:
                    orders.append(Order(OSMIUM, bid_price, -can_sell))
                    sold += can_sell
                if sold >= max_sell:
                    break

        # Remaining capacity: aggressive buy one tick below best ask, sell one tick above best bid
        rem_buy = max_buy - bought
        if rem_buy > 0 and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            orders.append(Order(OSMIUM, best_ask - 1, rem_buy))

        rem_sell = max_sell - sold
        if rem_sell > 0 and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order(OSMIUM, best_bid + 1, -rem_sell))

        return orders