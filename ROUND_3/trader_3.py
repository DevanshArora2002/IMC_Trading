"""
Round 3 trader_3 — Delta-1 assets only (no options).

Trades only the two main products:
  1. HYDROGEL_PACK      — Static MM around wall_mid (take/make with overbid)
  2. VELVETFRUIT_EXTRACT — EMA-based MM with inventory skew (take + make)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

# ──────────────────────────────────────────────────────────
# Product names & limits
# ──────────────────────────────────────────────────────────
HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"

POS_LIMITS: Dict[str, int] = {
    HYDROGEL: 200,
    VELVET: 200,
}

# ──────────────────────────────────────────────────────────
# Hydrogel params
# ──────────────────────────────────────────────────────────
HYDROGEL_ANCHOR = 10_000.0
HYDROGEL_EXTREME_DIST = 0

# ──────────────────────────────────────────────────────────
# Velvet params (EMA-based MM)
# ──────────────────────────────────────────────────────────
VELVET_EMA_WINDOW = 10
VELVET_TAKE_EDGE = 3
VELVET_MAKE_EDGE = 1
VELVET_INVENTORY_SKEW = 0.02

# ──────────────────────────────────────────────────────────
# Helper: order depth parsing
# ──────────────────────────────────────────────────────────

def _parse_depth(state: TradingState, symbol: str):
    buy_orders: Dict[int, int] = {}
    sell_orders: Dict[int, int] = {}
    try:
        od: OrderDepth = state.order_depths[symbol]
    except KeyError:
        return buy_orders, sell_orders
    try:
        buy_orders = {
            bp: abs(bv)
            for bp, bv in sorted(od.buy_orders.items(), key=lambda x: x[0], reverse=True)
        }
    except Exception:
        pass
    try:
        sell_orders = {
            sp: abs(sv)
            for sp, sv in sorted(od.sell_orders.items(), key=lambda x: x[0])
        }
    except Exception:
        pass
    return buy_orders, sell_orders


# ──────────────────────────────────────────────────────────
# HYDROGEL trader — Static MM (take/make around wall_mid)
# ──────────────────────────────────────────────────────────

class HydrogelTrader:
    def __init__(self, state: TradingState, data: dict):
        self.state = state
        self.data = data
        self.orders: List[Order] = []
        self.position = state.position.get(HYDROGEL, 0)
        self.limit = POS_LIMITS[HYDROGEL]

        self.buy_orders, self.sell_orders = _parse_depth(state, HYDROGEL)

        self.bid_wall: Optional[int] = None
        self.ask_wall: Optional[int] = None
        self.wall_mid: Optional[float] = None
        self.best_bid: Optional[int] = None
        self.best_ask: Optional[int] = None

        if self.buy_orders:
            self.bid_wall = min(self.buy_orders.keys())
            self.best_bid = max(self.buy_orders.keys())
        if self.sell_orders:
            self.ask_wall = max(self.sell_orders.keys())
            self.best_ask = min(self.sell_orders.keys())
        if self.bid_wall is not None and self.ask_wall is not None:
            self.wall_mid = (self.bid_wall + self.ask_wall) / 2.0

        self.max_buy = self.limit - self.position
        self.max_sell = self.limit + self.position

    def _bid(self, price: int, volume: int):
        vol = min(abs(volume), self.max_buy)
        if vol <= 0:
            return
        self.orders.append(Order(HYDROGEL, int(price), vol))
        self.max_buy -= vol

    def _ask(self, price: int, volume: int):
        vol = min(abs(volume), self.max_sell)
        if vol <= 0:
            return
        self.orders.append(Order(HYDROGEL, int(price), -vol))
        self.max_sell -= vol

    def get_orders(self) -> Dict[str, List[Order]]:
        if self.wall_mid is None:
            return {}

        # --- Taking ---
        for sp, sv in self.sell_orders.items():
            if sp <= self.wall_mid - 1:
                self._bid(sp, sv)
            elif sp <= self.wall_mid and self.position < 0:
                vol = min(sv, abs(self.position))
                self._bid(sp, vol)

        for bp, bv in self.buy_orders.items():
            if bp >= self.wall_mid + 1:
                self._ask(bp, bv)
            elif bp >= self.wall_mid and self.position > 0:
                vol = min(bv, self.position)
                self._ask(bp, vol)

        # --- Making ---
        bid_price = int(self.bid_wall + 1) if self.bid_wall is not None else None
        ask_price = int(self.ask_wall - 1) if self.ask_wall is not None else None

        if bid_price is None or ask_price is None:
            return {HYDROGEL: self.orders} if self.orders else {}

        for bp, bv in self.buy_orders.items():
            overbid = bp + 1
            if bv > 1 and overbid < self.wall_mid:
                bid_price = max(bid_price, overbid)
                break
            elif bp < self.wall_mid:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in self.sell_orders.items():
            underbid = sp - 1
            if sv > 1 and underbid > self.wall_mid:
                ask_price = min(ask_price, underbid)
                break
            elif sp > self.wall_mid:
                ask_price = min(ask_price, sp)
                break

        if (
            HYDROGEL_EXTREME_DIST > 0
            and self.wall_mid is not None
            and abs(self.wall_mid - HYDROGEL_ANCHOR) > HYDROGEL_EXTREME_DIST
        ):
            if self.position > 0:
                nap = ask_price - 1
                if nap > bid_price:
                    ask_price = nap
            elif self.position < 0:
                nbp = bid_price + 1
                if nbp < ask_price:
                    bid_price = nbp

        self._bid(bid_price, self.max_buy)
        self._ask(ask_price, self.max_sell)

        return {HYDROGEL: self.orders} if self.orders else {}


# ──────────────────────────────────────────────────────────
# VELVET trader — EMA-based MM with inventory skew
# ──────────────────────────────────────────────────────────

class VelvetTrader:
    def __init__(self, state: TradingState, data: dict):
        self.state = state
        self.data = data
        self.orders: List[Order] = []
        self.position = state.position.get(VELVET, 0)
        self.limit = POS_LIMITS[VELVET]

    def get_orders(self) -> Dict[str, List[Order]]:
        if VELVET not in self.state.order_depths:
            return {}

        depth = self.state.order_depths[VELVET]
        if not depth.buy_orders or not depth.sell_orders:
            return {}

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid = (best_bid + best_ask) / 2.0

        old_ema = self.data.get("velvet_ema", mid)
        alpha = 2.0 / (VELVET_EMA_WINDOW + 1)
        fair = alpha * mid + (1.0 - alpha) * old_ema
        self.data["velvet_ema"] = fair

        skewed_fair = fair - VELVET_INVENTORY_SKEW * self.position

        buy_room = self.limit - self.position
        sell_room = self.limit + self.position
        bought = 0
        sold = 0

        # --- Take mispriced liquidity ---
        for ask, volume in sorted(depth.sell_orders.items()):
            if ask > skewed_fair - VELVET_TAKE_EDGE or buy_room <= 0:
                break
            qty = min(-volume, buy_room)
            if qty > 0:
                self.orders.append(Order(VELVET, int(ask), qty))
                bought += qty
                buy_room -= qty

        for bid, volume in sorted(depth.buy_orders.items(), reverse=True):
            if bid < skewed_fair + VELVET_TAKE_EDGE or sell_room <= 0:
                break
            qty = min(volume, sell_room)
            if qty > 0:
                self.orders.append(Order(VELVET, int(bid), -qty))
                sold += qty
                sell_room -= qty

        post_position = self.position + bought - sold
        buy_room = self.limit - post_position
        sell_room = self.limit + post_position

        # --- Make at fair +/- edge ---
        bid_price = min(int(skewed_fair) - VELVET_MAKE_EDGE, best_ask - 1)
        if best_bid + 1 < skewed_fair:
            bid_price = max(bid_price, best_bid + 1)

        ask_price = max(int(skewed_fair) + VELVET_MAKE_EDGE, best_bid + 1)
        if best_ask - 1 > skewed_fair:
            ask_price = min(ask_price, best_ask - 1)

        if bid_price < ask_price:
            if buy_room > 0:
                self.orders.append(Order(VELVET, int(bid_price), buy_room))
            if sell_room > 0:
                self.orders.append(Order(VELVET, int(ask_price), -sell_room))

        return {VELVET: self.orders} if self.orders else {}


# ──────────────────────────────────────────────────────────
# Unified Trader
# ──────────────────────────────────────────────────────────

class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        data = self._deserialize(state.traderData)

        # --- HYDROGEL ---
        if HYDROGEL in state.order_depths:
            depth = state.order_depths[HYDROGEL]
            if depth.buy_orders and depth.sell_orders:
                ht = HydrogelTrader(state, data)
                for sym, orders in ht.get_orders().items():
                    if orders:
                        result[sym] = orders

        # --- VELVET ---
        if VELVET in state.order_depths:
            vt = VelvetTrader(state, data)
            for sym, orders in vt.get_orders().items():
                if orders:
                    result[sym] = orders

        return result, 0, self._serialize(data)

    @staticmethod
    def _deserialize(raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    @staticmethod
    def _serialize(data: dict) -> str:
        try:
            return json.dumps(data)
        except (TypeError, ValueError):
            return ""
