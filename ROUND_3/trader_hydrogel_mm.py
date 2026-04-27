"""
Round 3: HYDROGEL_PACK market making — StaticTrader wall-mid flow (Round 2 trader_6
take/make with overbid/underbid) at POSITION_LIMIT 200.

Optional hybrid: |wall_mid - 10_000| > EXTREME_DIST skews quotes one tick toward
inventory reduction (no EMAs; avoids high-turnover rules).

`datamodel` is imported from `ROUND_2` on sys.path; run with project root on PYTHONPATH
or from a runner that already loads datamodel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_R2 = Path(__file__).resolve().parent.parent / "ROUND_2"
if str(_R2) not in sys.path:
    sys.path.insert(0, str(_R2))

from datamodel import Order, OrderDepth, TradingState  # noqa: E402

HYDROGEL = "HYDROGEL_PACK"
POSITION_LIMIT = 200

# Prosperity round 2/3: optional separate bid slot (unused here; keep interface)
BID_VALUE = 0

# Hybrid regime: 0 = pure StaticTrader. Set e.g. 30–50 to enable anchor bias.
EXTREME_DIST = 0
ANCHOR = 10_000.0


class ProductTrader:
    def __init__(self, name, state, prints, new_trader_data, product_group=None):
        self.orders = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.position_limit = POSITION_LIMIT
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


class HydrogelStaticTrader(ProductTrader):
    """tutor StaticTrader: take at wall mid + make at bid_wall+1 / ask_wall-1 with over/under bid."""

    def __init__(self, state, prints, new_trader_data):
        super().__init__(HYDROGEL, state, prints, new_trader_data)

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

            self._apply_hybrid_quotes(bid_price, ask_price)
        return {self.name: self.orders}

    def _apply_hybrid_quotes(self, bid_price, ask_price):
        """Optional one-tick skew from |wall_mid - ANCHOR|; default no-op."""
        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)


class HydrogelHybridStaticTrader(HydrogelStaticTrader):
    """
    Same as HydrogelStaticTrader, but when EXTREME_DIST > 0 and
    |wall_mid - ANCHOR| is large, nudge quotes 1 tick toward reducing inventory
    (long: lower ask; short: higher bid) without EMAs.
    """

    def _apply_hybrid_quotes(self, bid_price, ask_price):
        bp, ap = int(bid_price), int(ask_price)
        if (
            self.wall_mid is not None
            and EXTREME_DIST > 0
            and abs(self.wall_mid - ANCHOR) > EXTREME_DIST
        ):
            if self.initial_position > 0:
                nap = ap - 1
                if nap > bp:
                    ap = nap
            elif self.initial_position < 0:
                nbp = bp + 1
                if nbp < ap:
                    bp = nbp
        self.bid(bp, self.max_allowed_buy_volume)
        self.ask(ap, self.max_allowed_sell_volume)


class Trader:
    def bid(self) -> int:  # noqa: A001  # Prosperity interface
        return BID_VALUE

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data: dict = self._deserialize(state.traderData)

        if HYDROGEL in state.order_depths:
            depth: OrderDepth = state.order_depths[HYDROGEL]
            if depth.buy_orders and depth.sell_orders:
                impl = (
                    HydrogelHybridStaticTrader
                    if EXTREME_DIST > 0
                    else HydrogelStaticTrader
                )
                prints: dict = {}
                new_data: dict = {}
                st = impl(state, prints, new_data)
                out = st.get_orders().get(HYDROGEL, [])
                if out:
                    result[HYDROGEL] = out

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
