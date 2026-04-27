"""
Round 3 trader_2 — Fresh strategy rewrite.

Diagnosis: trader.py lost -3,966 on options due to stale vol-smile coefficients
borrowed from the Round 5 winner (VOLCANIC_ROCK).  The actual VEV smile is
essentially flat at ~24% IV — the old coefficients produced wildly wrong theos
for deep ITM strikes.

Strategy:
  1. HYDROGEL_PACK  — Static MM around wall_mid (proven +271 in prior logs)
  2. VELVETFRUIT_EXTRACT — EMA-based MM with inventory skew (replaces broken
     level-based MR)
  3. VEV options — Online vol-smile calibration, conservative IV scalping on
     ATM strikes only (5000-5400), delta hedging via VELVET, no deep ITM trading
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
HYDROGEL_EXTREME_DIST = 0

# ──────────────────────────────────────────────────────────
# Velvet params (EMA-based MM)
# ──────────────────────────────────────────────────────────
VELVET_EMA_WINDOW = 10
VELVET_TAKE_EDGE = 3
VELVET_MAKE_EDGE = 1
VELVET_INVENTORY_SKEW = 0.02

# ──────────────────────────────────────────────────────────
# Option params
# ──────────────────────────────────────────────────────────
OPTION_DAY = 3
DAYS_PER_YEAR = 365

# Fallback base IV when online calibration has insufficient data
FALLBACK_BASE_IV = 0.239

THR_OPEN = 1.0
THR_CLOSE = 0.0
LOW_VEGA_THR_ADJ = 0.5

THEO_NORM_WINDOW = 20
IV_SCALPING_THR = 0.7
IV_SCALPING_WINDOW = 100

# ATM strikes eligible for IV scalping
IV_SCALPING_STRIKES = {5000, 5100, 5200, 5300, 5400}

# Max position per option strike (conservative)
MAX_OPTION_POS = 50

# ──────────────────────────────────────────────────────────
# Normal CDF / PDF (pure-python)
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
# Black-Scholes helpers
# ──────────────────────────────────────────────────────────

def _bs_call(S: float, K: float, TTE: float, sigma: float):
    if TTE <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * TTE) / (sigma * TTE ** 0.5)
    d2 = d1 - sigma * TTE ** 0.5
    call_val = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    return call_val, delta


def _bs_vega(S: float, K: float, TTE: float, sigma: float) -> float:
    if TTE <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * TTE) / (sigma * TTE ** 0.5)
    return S * _norm_pdf(d1) * TTE ** 0.5


def _implied_vol(S: float, K: float, TTE: float, market_price: float,
                 tol: float = 1e-6, max_iter: int = 50) -> Optional[float]:
    """Newton-Raphson IV solver. Returns None if it fails to converge."""
    if TTE <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic + tol:
        return None

    sigma = 0.25
    for _ in range(max_iter):
        price, _ = _bs_call(S, K, TTE, sigma)
        vega = _bs_vega(S, K, TTE, sigma)
        if vega < 1e-10:
            return None
        diff = price - market_price
        sigma -= diff / vega
        if sigma <= 0.001:
            sigma = 0.001
        if abs(diff) < tol:
            return sigma
    return None


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

        # Update running EMA fair value
        old_ema = self.data.get("velvet_ema", mid)
        alpha = 2.0 / (VELVET_EMA_WINDOW + 1)
        fair = alpha * mid + (1.0 - alpha) * old_ema
        self.data["velvet_ema"] = fair

        # Inventory-skewed fair value
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

        # Update room after taking
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
# Option product wrapper
# ──────────────────────────────────────────────────────────

class _OptionProduct:
    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.orders: List[Order] = []
        self.position = state.position.get(name, 0)
        self.limit = min(POS_LIMITS.get(name, 300), MAX_OPTION_POS)
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


# ──────────────────────────────────────────────────────────
# OPTION trader — Online vol smile + IV scalping + delta hedge
# ──────────────────────────────────────────────────────────

class OptionTrader:
    def __init__(self, state: TradingState, data: dict):
        self.state = state
        self.data = data

        self.all_options = [_OptionProduct(s, state) for s in OPTION_SYMBOLS]
        self.underlying_buy, self.underlying_sell = _parse_depth(state, VELVET)

        self.underlying_best_bid: Optional[int] = None
        self.underlying_best_ask: Optional[int] = None
        self.underlying_mid: Optional[float] = None

        if self.underlying_buy:
            self.underlying_best_bid = max(self.underlying_buy.keys())
        if self.underlying_sell:
            self.underlying_best_ask = min(self.underlying_sell.keys())
        if self.underlying_best_bid is not None and self.underlying_best_ask is not None:
            self.underlying_mid = (self.underlying_best_bid + self.underlying_best_ask) / 2.0

        self.tte = self._compute_tte()
        self.smile_coeffs = self._fit_online_smile()
        self.indicators = self._calculate_indicators()

    def _compute_tte(self) -> float:
        tte = 1.0 - (DAYS_PER_YEAR - 8 + OPTION_DAY + self.state.timestamp / 100.0 / 10_000.0) / DAYS_PER_YEAR
        return max(tte, 1e-6)

    def _fit_online_smile(self) -> Optional[List[float]]:
        """Fit a parabolic vol smile from observed option mid-prices this tick.
        Returns [a, b, c] coefficients or None if insufficient data."""
        if self.underlying_mid is None or self.tte <= 0:
            return None

        S = self.underlying_mid
        points: List[tuple] = []  # (moneyness, iv)

        for opt in self.all_options:
            if opt.wall_mid is None:
                continue
            k = int(opt.name.split("_")[-1])
            iv = _implied_vol(S, k, self.tte, opt.wall_mid)
            if iv is None or iv <= 0.01 or iv > 2.0:
                continue
            m = math.log(k / S) / self.tte ** 0.5
            points.append((m, iv))

        if len(points) < 3:
            return None

        # Least-squares parabola fit: iv = a*m^2 + b*m + c
        # Normal equations for degree-2 polynomial
        n = len(points)
        s0 = float(n)
        s1 = sum(p[0] for p in points)
        s2 = sum(p[0] ** 2 for p in points)
        s3 = sum(p[0] ** 3 for p in points)
        s4 = sum(p[0] ** 4 for p in points)
        t0 = sum(p[1] for p in points)
        t1 = sum(p[0] * p[1] for p in points)
        t2 = sum(p[0] ** 2 * p[1] for p in points)

        # Solve 3x3 system via Cramer's rule
        # [s4 s3 s2] [a]   [t2]
        # [s3 s2 s1] [b] = [t1]
        # [s2 s1 s0] [c]   [t0]
        det = (s4 * (s2 * s0 - s1 * s1)
               - s3 * (s3 * s0 - s1 * s2)
               + s2 * (s3 * s1 - s2 * s2))
        if abs(det) < 1e-20:
            return None

        a = (t2 * (s2 * s0 - s1 * s1)
             - s3 * (t1 * s0 - s1 * t0)
             + s2 * (t1 * s1 - s2 * t0)) / det
        b = (s4 * (t1 * s0 - s1 * t0)
             - t2 * (s3 * s0 - s1 * s2)
             + s2 * (s3 * t0 - t1 * s2)) / det
        c = (s4 * (s2 * t0 - t1 * s1)
             - s3 * (s3 * t0 - t1 * s2)
             + t2 * (s3 * s1 - s2 * s2)) / det

        if c <= 0.01 or c > 1.5:
            return None

        return [a, b, c]

    def _get_iv(self, S: float, K: float) -> float:
        if self.tte <= 0:
            return FALLBACK_BASE_IV
        m = math.log(K / S) / self.tte ** 0.5
        if self.smile_coeffs is not None:
            a, b, c = self.smile_coeffs
            iv = a * m * m + b * m + c
            if 0.01 < iv < 2.0:
                return iv

        # Fallback: EMA of base IV, or static fallback
        ema_iv = self.data.get("ema_base_iv", FALLBACK_BASE_IV)
        return ema_iv

    def _get_option_values(self, S: float, K: float):
        iv = self._get_iv(S, K)
        if iv <= 0:
            iv = 0.01
        call_val, delta = _bs_call(S, K, self.tte, iv)
        vega = _bs_vega(S, K, self.tte, iv)
        return call_val, delta, vega

    def _ema(self, key: str, window: int, value: float) -> float:
        old = self.data.get(key, 0.0)
        alpha = 2.0 / (window + 1)
        new_val = alpha * value + (1.0 - alpha) * old
        self.data[key] = new_val
        return new_val

    def _calculate_indicators(self) -> dict:
        ind: dict = {
            "mean_theo_diffs": {},
            "current_theo_diffs": {},
            "switch_means": {},
            "deltas": {},
            "vegas": {},
        }

        if self.underlying_mid is None:
            return ind

        S = self.underlying_mid

        # Track EMA of base IV for fallback
        if self.smile_coeffs is not None:
            base_iv = self.smile_coeffs[2]
            self._ema("ema_base_iv", 50, base_iv)

        for opt in self.all_options:
            k = int(opt.name.split("_")[-1])
            if k not in IV_SCALPING_STRIKES:
                continue
            if opt.wall_mid is None:
                continue

            theo, delta, vega = self._get_option_values(S, k)
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

    def get_option_orders(self) -> Dict[str, List[Order]]:
        warmup = max(THEO_NORM_WINDOW, IV_SCALPING_WINDOW)
        if self.state.timestamp / 100 < warmup:
            return {}

        out: Dict[str, List[Order]] = {}

        for opt in self.all_options:
            k = int(opt.name.split("_")[-1])
            if k not in IV_SCALPING_STRIKES:
                continue

            n = opt.name
            if (
                n not in self.indicators["mean_theo_diffs"]
                or n not in self.indicators["current_theo_diffs"]
                or n not in self.indicators["switch_means"]
            ):
                continue

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
                    elif sig >= THR_CLOSE and opt.position > 0:
                        opt.ask(opt.best_bid, opt.position)

                if opt.best_ask is not None:
                    sig2 = cur - opt.wall_mid + opt.best_ask - mean_d
                    if sig2 <= -(THR_OPEN + low_vega_adj) and opt.max_buy > 0:
                        opt.bid(opt.best_ask, opt.max_buy)
                    elif sig2 <= -THR_CLOSE and opt.position < 0:
                        opt.bid(opt.best_ask, -opt.position)
            else:
                # Below threshold: flatten position
                if opt.position > 0 and opt.best_bid is not None:
                    opt.ask(opt.best_bid, opt.position)
                elif opt.position < 0 and opt.best_ask is not None:
                    opt.bid(opt.best_ask, -opt.position)

            out[n] = opt.orders

        return out

    def compute_aggregate_delta(self) -> float:
        """Sum delta * position across all option strikes."""
        total_delta = 0.0
        for opt in self.all_options:
            n = opt.name
            if n in self.indicators["deltas"]:
                total_delta += self.indicators["deltas"][n] * opt.position
        return total_delta

    def get_delta_hedge_orders(self) -> Dict[str, List[Order]]:
        """Hedge aggregate option delta by trading VELVET in the opposite direction."""
        if self.underlying_mid is None:
            return {}

        agg_delta = self.compute_aggregate_delta()
        velvet_pos = self.state.position.get(VELVET, 0)

        # Target VELVET position = -aggregate_delta (rounded)
        target_velvet = -int(round(agg_delta))
        target_velvet = max(-POS_LIMITS[VELVET], min(POS_LIMITS[VELVET], target_velvet))

        delta_to_trade = target_velvet - velvet_pos
        if delta_to_trade == 0:
            return {}

        orders: List[Order] = []

        if delta_to_trade > 0:
            remaining = delta_to_trade
            for ask, vol in sorted(self.underlying_sell.items()):
                if remaining <= 0:
                    break
                qty = min(vol, remaining)
                if qty > 0:
                    orders.append(Order(VELVET, int(ask), qty))
                    remaining -= qty
            if remaining > 0 and self.underlying_best_ask is not None:
                orders.append(Order(VELVET, int(self.underlying_best_ask), remaining))
        else:
            remaining = -delta_to_trade
            for bid, vol in sorted(self.underlying_buy.items(), reverse=True):
                if remaining <= 0:
                    break
                qty = min(vol, remaining)
                if qty > 0:
                    orders.append(Order(VELVET, int(bid), -qty))
                    remaining -= qty
            if remaining > 0 and self.underlying_best_bid is not None:
                orders.append(Order(VELVET, int(self.underlying_best_bid), -remaining))

        return {VELVET: orders} if orders else {}


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

        # --- OPTIONS + VELVET delta hedge ---
        has_options = any(s in state.order_depths for s in OPTION_SYMBOLS)
        velvet_used_by_options = False

        if has_options and VELVET in state.order_depths:
            ot = OptionTrader(state, data)
            opt_orders = ot.get_option_orders()
            for sym, orders in opt_orders.items():
                if orders:
                    result[sym] = orders

            hedge_orders = ot.get_delta_hedge_orders()
            if VELVET in hedge_orders and hedge_orders[VELVET]:
                velvet_used_by_options = True
                result[VELVET] = hedge_orders[VELVET]

        # --- VELVET standalone MM (only if not used for delta hedging) ---
        if not velvet_used_by_options and VELVET in state.order_depths:
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
