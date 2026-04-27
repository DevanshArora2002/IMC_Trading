"""
Round 3 Unified Trader — trades HYDROGEL_PACK, VELVETFRUIT_EXTRACT, and
10 VEV_* option vouchers simultaneously.

Strategy overview
-----------------
* HYDROGEL_PACK:  Static market-making around wall_mid (take mispriced orders,
                  make at bid_wall+1 / ask_wall−1) with optional anchor-reversion
                  bias when |wall_mid − 10 000| > EXTREME_DIST.
* VELVETFRUIT_EXTRACT: Level-based mean reversion (buy near 5230, sell near 5265,
                  exit near 5240, step_size 25).
* VEV_* options: Vol-smile fitting → IV scalping for ATM/near-ATM strikes,
                  mean-reversion for deep-ITM strikes, plus delta hedging via
                  the underlying.

All three product groups share a single JSON traderData blob that persists EMAs,
mid-price history, and theo-diff tracking across ticks.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

# ──────────────────────────────────────────────────────────
# Product names & limits
# ──────────────────────────────────────────────────────────
HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"

OPTION_SYMBOLS = [
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300",
    "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
]

POS_LIMITS: Dict[str, int] = {
    HYDROGEL: 200,
    VELVET: 200,
}
for _s in OPTION_SYMBOLS:
    POS_LIMITS[_s] = 300

# ──────────────────────────────────────────────────────────
# Hydrogel params
# ──────────────────────────────────────────────────────────
HYDROGEL_ANCHOR = 10_000.0
HYDROGEL_EXTREME_DIST = 0  # 0 = pure static MM; set 30-50 for hybrid

# ──────────────────────────────────────────────────────────
# Velvet params (level-based mean reversion)
# ──────────────────────────────────────────────────────────
VELVET_BUY_LEVEL = 5230
VELVET_SELL_LEVEL = 5265
VELVET_EXIT_LEVEL = 5240
VELVET_STEP_SIZE = 25

# ──────────────────────────────────────────────────────────
# Option params
# ──────────────────────────────────────────────────────────
OPTION_DAY = 3  # Round 3 → TTE starts at 5 days
DAYS_PER_YEAR = 365

# Vol-smile fitted coefficients: v(m) = a*m^2 + b*m + c
VOL_SMILE_COEFFS = [0.27362531, 0.01007566, 0.14876677]

THR_OPEN = 0.5
THR_CLOSE = 0.0
LOW_VEGA_THR_ADJ = 0.5

THEO_NORM_WINDOW = 20
IV_SCALPING_THR = 0.7
IV_SCALPING_WINDOW = 100

UNDERLYING_MR_THR = 15
UNDERLYING_MR_WINDOW = 10
OPTIONS_MR_THR = 5
OPTIONS_MR_WINDOW = 30

# IV-scalping strikes: ATM / near-ATM
IV_SCALPING_STRIKES = {5000, 5100, 5200, 5300}
# MR strikes: deep ITM
MR_STRIKES = {4000, 4500}

# ──────────────────────────────────────────────────────────
# Normal CDF / PDF (pure-python, no scipy dependency)
# ──────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

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

        # 1. Taking
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

        # 2. Making
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

        # Optional hybrid skew
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
# VELVET trader — Level-based mean reversion
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

        buy_room = self.limit - self.position
        sell_room = self.limit + self.position

        if mid <= VELVET_BUY_LEVEL:
            qty = min(VELVET_STEP_SIZE, buy_room)
            if qty > 0:
                for ask, vol in sorted(depth.sell_orders.items()):
                    if qty <= 0:
                        break
                    fill = min(-vol, qty)
                    if fill > 0:
                        self.orders.append(Order(VELVET, int(ask), fill))
                        qty -= fill
                        buy_room -= fill

        elif mid >= VELVET_SELL_LEVEL:
            qty = min(VELVET_STEP_SIZE, sell_room)
            if qty > 0:
                for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                    if qty <= 0:
                        break
                    fill = min(vol, qty)
                    if fill > 0:
                        self.orders.append(Order(VELVET, int(bid), -fill))
                        qty -= fill
                        sell_room -= fill

        elif VELVET_BUY_LEVEL < mid < VELVET_SELL_LEVEL:
            if self.position > 0 and mid >= VELVET_EXIT_LEVEL:
                qty = min(self.position, VELVET_STEP_SIZE, sell_room)
                if qty > 0:
                    for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                        if qty <= 0:
                            break
                        fill = min(vol, qty)
                        if fill > 0:
                            self.orders.append(Order(VELVET, int(bid), -fill))
                            qty -= fill
            elif self.position < 0 and mid <= VELVET_EXIT_LEVEL:
                qty = min(abs(self.position), VELVET_STEP_SIZE, buy_room)
                if qty > 0:
                    for ask, vol in sorted(depth.sell_orders.items()):
                        if qty <= 0:
                            break
                        fill = min(-vol, qty)
                        if fill > 0:
                            self.orders.append(Order(VELVET, int(ask), fill))
                            qty -= fill

        return {VELVET: self.orders} if self.orders else {}


# ──────────────────────────────────────────────────────────
# OPTION trader — Vol-smile IV scalping + MR + delta hedge
# ──────────────────────────────────────────────────────────

class _OptionProduct:
    """Thin wrapper around per-option state."""
    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.orders: List[Order] = []
        self.position = state.position.get(name, 0)
        self.limit = POS_LIMITS.get(name, 300)
        self.max_buy = self.limit - self.position
        self.max_sell = self.limit + self.position

        buy_orders, sell_orders = _parse_depth(state, name)
        self.buy_orders = buy_orders
        self.sell_orders = sell_orders

        self.bid_wall: Optional[int] = None
        self.ask_wall: Optional[int] = None
        self.wall_mid: Optional[float] = None
        self.best_bid: Optional[int] = None
        self.best_ask: Optional[int] = None

        if buy_orders:
            self.bid_wall = min(buy_orders.keys())
            self.best_bid = max(buy_orders.keys())
        if sell_orders:
            self.ask_wall = max(sell_orders.keys())
            self.best_ask = min(sell_orders.keys())

        if self.bid_wall is not None and self.ask_wall is not None:
            self.wall_mid = (self.bid_wall + self.ask_wall) / 2.0
        elif self.ask_wall is not None:
            self.wall_mid = self.ask_wall - 0.5
            self.bid_wall = self.ask_wall - 1
            self.best_bid = self.ask_wall - 1
        elif self.bid_wall is not None:
            self.wall_mid = self.bid_wall + 0.5
            self.ask_wall = self.bid_wall + 1
            self.best_ask = self.bid_wall + 1

    def bid(self, price: int, volume: int):
        vol = min(abs(int(volume)), self.max_buy)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), vol))
        self.max_buy -= vol

    def ask(self, price: int, volume: int):
        vol = min(abs(int(volume)), self.max_sell)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), -vol))
        self.max_sell -= vol


class OptionTrader:
    def __init__(self, state: TradingState, data: dict):
        self.state = state
        self.data = data

        self.options = [_OptionProduct(s, state) for s in OPTION_SYMBOLS]
        self.underlying = _OptionProduct(VELVET, state)

        self.indicators = self._calculate_indicators()

    @staticmethod
    def _bs_call(S: float, K: float, TTE: float, sigma: float):
        if TTE <= 0 or sigma <= 0:
            return max(S - K, 0.0), 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma**2 * TTE) / (sigma * TTE**0.5)
        d2 = d1 - sigma * TTE**0.5
        call_val = S * _norm_cdf(d1) - K * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        return call_val, delta

    @staticmethod
    def _bs_vega(S: float, K: float, TTE: float, sigma: float):
        if TTE <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + 0.5 * sigma**2 * TTE) / (sigma * TTE**0.5)
        return S * _norm_pdf(d1) * TTE**0.5

    @staticmethod
    def _get_iv(S: float, K: float, TTE: float) -> float:
        if TTE <= 0:
            return 0.15
        m = math.log(K / S) / TTE**0.5
        a, b, c = VOL_SMILE_COEFFS
        return a * m * m + b * m + c

    def _get_option_values(self, S: float, K: float, TTE: float):
        iv = self._get_iv(S, K, TTE)
        if iv <= 0:
            iv = 0.01
        call_val, delta = self._bs_call(S, K, TTE, iv)
        vega = self._bs_vega(S, K, TTE, iv)
        return call_val, delta, vega

    def _ema(self, key: str, window: int, value: float) -> float:
        old = self.data.get(key, 0.0)
        alpha = 2.0 / (window + 1)
        new_val = alpha * value + (1.0 - alpha) * old
        self.data[key] = new_val
        return new_val

    def _calculate_indicators(self) -> dict:
        ind: dict = {
            "ema_u_dev": None,
            "ema_o_dev": None,
            "mean_theo_diffs": {},
            "current_theo_diffs": {},
            "switch_means": {},
            "deltas": {},
            "vegas": {},
        }

        if self.underlying.wall_mid is None:
            return ind
        if self.underlying.best_bid is None or self.underlying.best_ask is None:
            return ind

        u_mid = self.underlying.best_bid * 0.5 + self.underlying.best_ask * 0.5

        new_u = self._ema("ema_u", UNDERLYING_MR_WINDOW, u_mid)
        ind["ema_u_dev"] = u_mid - new_u

        new_o = self._ema("ema_o", OPTIONS_MR_WINDOW, u_mid)
        ind["ema_o_dev"] = u_mid - new_o

        for opt in self.options:
            k = int(opt.name.split("_")[-1])

            if opt.wall_mid is None:
                continue

            tte_raw = 1.0 - (DAYS_PER_YEAR - 8 + OPTION_DAY + self.state.timestamp / 100.0 / 10_000.0) / DAYS_PER_YEAR
            if tte_raw <= 0:
                tte_raw = 1e-6

            theo, delta, vega = self._get_option_values(u_mid, k, tte_raw)
            theo_diff = opt.wall_mid - theo

            ind["current_theo_diffs"][opt.name] = theo_diff
            ind["deltas"][opt.name] = delta
            ind["vegas"][opt.name] = vega

            mean_diff = self._ema(f"{opt.name}_theo_diff", THEO_NORM_WINDOW, theo_diff)
            ind["mean_theo_diffs"][opt.name] = mean_diff

            avg_dev = self._ema(
                f"{opt.name}_avg_devs",
                IV_SCALPING_WINDOW,
                abs(theo_diff - mean_diff),
            )
            ind["switch_means"][opt.name] = avg_dev

        return ind

    def _iv_scalping_orders(self, options: List[_OptionProduct]) -> dict:
        out: dict = {}
        for opt in options:
            n = opt.name
            if (
                n in self.indicators["mean_theo_diffs"]
                and n in self.indicators["current_theo_diffs"]
                and n in self.indicators["switch_means"]
            ):
                sw = self.indicators["switch_means"][n]
                cur = self.indicators["current_theo_diffs"][n]
                mean_d = self.indicators["mean_theo_diffs"][n]
                vega = self.indicators["vegas"].get(n, 0)

                low_vega_adj = LOW_VEGA_THR_ADJ if vega <= 1 else 0.0

                if sw >= IV_SCALPING_THR:
                    if opt.best_bid is not None:
                        sig = cur - opt.wall_mid + opt.best_bid - mean_d
                        if sig >= (THR_OPEN + low_vega_adj) and opt.max_sell > 0:
                            opt.ask(opt.best_bid, opt.max_sell)
                        if sig >= THR_CLOSE and opt.position > 0:
                            opt.ask(opt.best_bid, opt.position)

                    if opt.best_ask is not None:
                        sig2 = cur - opt.wall_mid + opt.best_ask - mean_d
                        if sig2 <= -(THR_OPEN + low_vega_adj) and opt.max_buy > 0:
                            opt.bid(opt.best_ask, opt.max_buy)
                        if sig2 <= -THR_CLOSE and opt.position < 0:
                            opt.bid(opt.best_ask, -opt.position)
                else:
                    if opt.position > 0 and opt.best_bid is not None:
                        opt.ask(opt.best_bid, opt.position)
                    elif opt.position < 0 and opt.best_ask is not None:
                        opt.bid(opt.best_ask, -opt.position)

            out[n] = opt.orders
        return out

    def _mr_orders(self, options: List[_OptionProduct]) -> dict:
        out: dict = {}
        for opt in options:
            n = opt.name
            if (
                n in self.indicators["current_theo_diffs"]
                and n in self.indicators["mean_theo_diffs"]
                and self.indicators.get("ema_o_dev") is not None
            ):
                dev = self.indicators["ema_o_dev"]
                iv_dev = self.indicators["current_theo_diffs"][n] - self.indicators["mean_theo_diffs"][n]
                total_dev = dev + iv_dev

                if total_dev > OPTIONS_MR_THR and opt.max_sell > 0 and opt.best_bid is not None:
                    opt.ask(opt.best_bid, opt.max_sell)
                elif total_dev < -OPTIONS_MR_THR and opt.max_buy > 0 and opt.best_ask is not None:
                    opt.bid(opt.best_ask, opt.max_buy)

            out[n] = opt.orders
        return out

    def _underlying_orders(self) -> Dict[str, List[Order]]:
        u = self.underlying
        if self.indicators.get("ema_u_dev") is None:
            return {}

        dev = self.indicators["ema_o_dev"]
        if dev is None:
            return {}

        if dev > UNDERLYING_MR_THR and u.max_sell > 0 and u.bid_wall is not None:
            u.ask(u.bid_wall + 1, u.max_sell)
        elif dev < -UNDERLYING_MR_THR and u.max_buy > 0 and u.ask_wall is not None:
            u.bid(u.ask_wall - 1, u.max_buy)

        return {VELVET: u.orders} if u.orders else {}

    def get_orders(self) -> Dict[str, List[Order]]:
        min_warmup = min(THEO_NORM_WINDOW, UNDERLYING_MR_WINDOW, OPTIONS_MR_WINDOW)
        if self.state.timestamp / 100 < min_warmup:
            return {}

        iv_opts = [o for o in self.options if int(o.name.split("_")[-1]) in IV_SCALPING_STRIKES]
        mr_opts = [o for o in self.options if int(o.name.split("_")[-1]) in MR_STRIKES]

        result: dict = {}
        result.update(self._iv_scalping_orders(iv_opts))
        result.update(self._mr_orders(mr_opts))

        return result

    def get_underlying_orders(self) -> Dict[str, List[Order]]:
        if self.state.timestamp / 100 < UNDERLYING_MR_WINDOW:
            return {}
        return self._underlying_orders()


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
                h_orders = ht.get_orders()
                for sym, orders in h_orders.items():
                    if orders:
                        result[sym] = orders

        # --- VELVET (level-based MR) ---
        velvet_used_by_options = False
        if VELVET in state.order_depths:
            has_options = any(s in state.order_depths for s in OPTION_SYMBOLS)
            if has_options:
                ot = OptionTrader(state, data)
                opt_orders = ot.get_orders()
                for sym, orders in opt_orders.items():
                    if orders:
                        result[sym] = orders
                u_orders = ot.get_underlying_orders()
                if VELVET in u_orders and u_orders[VELVET]:
                    velvet_used_by_options = True
                    result[VELVET] = u_orders[VELVET]

            if not velvet_used_by_options:
                vt = VelvetTrader(state, data)
                v_orders = vt.get_orders()
                for sym, orders in v_orders.items():
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
