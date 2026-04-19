"""Round 2 trader v6: INTARIAN_PEPPER_ROOT unchanged from trader_4; Osmium uses
the tutorial StaticTrader wall-mid flow (take through book + make at bid_wall+1 /
ask_wall-1 with overbid/underbid adjustments).

OSMIUM logic matches TUTORIAL_ROUND_1/new_trader.py `StaticTrader` adapted to
`ASH_COATED_OSMIUM` and POSITION_LIMIT=80.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math

# ── Product names ────────────────────────────────────────────────────────
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

POS_LIMITS = {
    OSMIUM: POSITION_LIMIT,
}

# ── PEPPER parameters (unchanged from trader_4) ─────────────────────────
PEPPER_SLOPE = 0.001
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# ── BID for Round-2 side mechanic ────────────────────────────────────────
BID_VALUE = 30


class ProductTrader:
    """Minimal ProductTrader for Round 2 (walls + take/make helpers)."""

    def __init__(self, name, state, prints, new_trader_data, product_group=None):
        self.orders = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = (
            self.get_max_allowed_volume()
        )

    def get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except Exception:
            pass
        return best_bid, best_ask

    def get_walls(self):
        bid_wall = wall_mid = ask_wall = None
        try:
            bid_wall = min(self.mkt_buy_orders.keys())
        except Exception:
            pass
        try:
            ask_wall = max(self.mkt_sell_orders.keys())
        except Exception:
            pass
        try:
            wall_mid = (bid_wall + ask_wall) / 2
        except Exception:
            pass
        return bid_wall, wall_mid, ask_wall

    def get_max_allowed_volume(self):
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def get_order_depth(self):
        buy_orders, sell_orders = {}, {}
        try:
            order_depth = self.state.order_depths[self.name]
        except Exception:
            return buy_orders, sell_orders
        try:
            buy_orders = {
                bp: abs(bv)
                for bp, bv in sorted(
                    order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True
                )
            }
        except Exception:
            pass
        try:
            sell_orders = {
                sp: abs(sv)
                for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            }
        except Exception:
            pass
        return buy_orders, sell_orders

    def bid(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging:
            self.log(
                "BUYO",
                {"p": price, "s": self.name, "v": int(volume)},
                product_group="ORDERS",
            )
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        if logging:
            self.log(
                "SELLO",
                {"p": price, "s": self.name, "v": int(volume)},
                product_group="ORDERS",
            )
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(order)

    def log(self, kind, message, product_group=None):
        if product_group is None:
            product_group = self.product_group
        if product_group == "ORDERS":
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message
        self.prints[product_group] = group

    def get_orders(self):
        return {}


class StaticTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(OSMIUM, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is not None:
            ##########################################################
            ####### 1. TAKING
            ##########################################################
            for sp, sv in self.mkt_sell_orders.items():
                if sp <= self.wall_mid - 1:
                    self.bid(sp, sv, logging=False)
                elif sp <= self.wall_mid and self.initial_position < 0:
                    volume = min(sv, abs(self.initial_position))
                    self.bid(sp, volume, logging=False)

            for bp, bv in self.mkt_buy_orders.items():
                if bp >= self.wall_mid + 1:
                    self.ask(bp, bv, logging=False)
                elif bp >= self.wall_mid and self.initial_position > 0:
                    volume = min(bv, self.initial_position)
                    self.ask(bp, volume, logging=False)

            ###########################################################
            ####### 2. MAKING
            ###########################################################
            bid_price = int(self.bid_wall + 1)
            ask_price = int(self.ask_wall - 1)

            for bp, bv in self.mkt_buy_orders.items():
                overbidding_price = bp + 1
                if bv > 1 and overbidding_price < self.wall_mid:
                    bid_price = max(bid_price, overbidding_price)
                    break
                elif bp < self.wall_mid:
                    bid_price = max(bid_price, bp)
                    break

            for sp, sv in self.mkt_sell_orders.items():
                underbidding_price = sp - 1
                if sv > 1 and underbidding_price > self.wall_mid:
                    ask_price = min(ask_price, underbidding_price)
                    break
                elif sp > self.wall_mid:
                    ask_price = min(ask_price, sp)
                    break

            self.bid(bid_price, self.max_allowed_buy_volume)
            self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


class Trader:
    def bid(self) -> int:
        return BID_VALUE

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, data)

        return result, 0, self._serialize(data)

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

    def _trade_pepper(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[PEPPER]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        if "pepper_init_mid" not in data:
            data["pepper_init_mid"] = (best_bid + best_ask) / 2.0

        fair = data["pepper_init_mid"] + PEPPER_SLOPE * state.timestamp
        position = state.position.get(PEPPER, 0)

        bought = 0
        for ask in sorted(depth.sell_orders.keys()):
            if ask > fair + PEPPER_TAKE_BUFFER:
                break
            room = POSITION_LIMIT - position - bought
            if room <= 0:
                break
            ask_vol = -depth.sell_orders[ask]
            qty = min(ask_vol, room)
            if qty > 0:
                orders.append(Order(PEPPER, int(ask), qty))
                bought += qty

        room = POSITION_LIMIT - position - bought
        if room > 0:
            orders.append(Order(PEPPER, int(math.floor(fair)), room))

        net_long = position + bought
        if net_long > 0:
            orders.append(
                Order(PEPPER, int(math.ceil(fair)) + PEPPER_OUTLIER_EDGE, -net_long)
            )

        return orders

    def _trade_osmium(self, state: TradingState, data: dict) -> list[Order]:
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return []

        prints: dict = {}
        new_trader_data: dict = {}
        st = StaticTrader(state, prints, new_trader_data)
        return st.get_orders().get(OSMIUM, [])
