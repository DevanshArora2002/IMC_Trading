from datamodel import OrderDepth, TradingState, Order
from typing import Optional
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── OSMIUM tunables (regime-aware mean reversion + stop-loss) ──────────
OSMIUM_ANCHOR = 10_000
OSMIUM_ANCHOR_BAND = 1.5          # ticks; soft anchor only inside this band
OSMIUM_EMA_FAST_ALPHA = 0.20
OSMIUM_EMA_SLOW_ALPHA = 0.05
OSMIUM_INV_SKEW_K = 2.0           # ticks shifted per full position
OSMIUM_WARMUP_TICKS = 20          # maker-only until warm

# Stop-loss trigger family switch: "ema_cross_consec" | "rolling_drop" | "zscore_neg" | "none"
# Winner from trader_6_backtest.ipynb (days -2,-1,0, taker-only fill model):
#   mode="none" ranked #1 (pnl_total=251,853.5 vs trader.py ref=251,935); every flush
#   family underperformed because the taker-only sim doesn't reward maker fills, so
#   crossing the spread to flush just burns the edge. Regime-aware fair + inventory
#   skew alone match trader.py and retain downside protection via the skew pulling
#   quotes down as inventory grows.
OSMIUM_FLUSH_MODE = "none"
OSMIUM_FLUSH_CONSEC = 5           # best within ema_cross_consec family (fallback)
OSMIUM_FLUSH_W = 40               # window for rolling_drop / zscore_neg
OSMIUM_FLUSH_DROP = 3             # ticks drop for rolling_drop
OSMIUM_FLUSH_Z = 2.5              # best within zscore_neg family (fallback)


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

    # ── INTARIAN_PEPPER_ROOT: buy-and-hold (unchanged from trader.py) ──

    def _trade_pepper(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(PEPPER, 0)
        order_depth: OrderDepth = state.order_depths[PEPPER]

        max_buy = POSITION_LIMIT - position
        if max_buy <= 0 or not order_depth.sell_orders:
            return orders

        
        best_ask = min(order_depth.sell_orders.keys())
        vol = min(abs(order_depth.sell_orders[best_ask]), max_buy)
        if vol > 0:
            orders.append(Order(PEPPER, best_ask, vol))

        return orders

    # ── ASH_COATED_OSMIUM: regime-aware fair, inventory skew, stop-loss ─

    def _trade_osmium(self, state: TradingState, trader_state: dict) -> list[Order]:
        orders: list[Order] = []
        position = state.position.get(OSMIUM, 0)
        order_depth: OrderDepth = state.order_depths[OSMIUM]

        mid_price = self._compute_mid(order_depth)
        if mid_price is None:
            return orders

        # ── read persisted state ──
        ema_fast = trader_state.get("osmium_ema_fast", mid_price)
        ema_slow = trader_state.get("osmium_ema_slow", mid_price)
        mid_hist: list[float] = trader_state.get("osmium_mid_hist", [])
        down_streak = int(trader_state.get("osmium_down_streak", 0))
        up_streak = int(trader_state.get("osmium_up_streak", 0))
        warmup = int(trader_state.get("osmium_warmup", 0))

        # ── update indicators ──
        ema_fast = OSMIUM_EMA_FAST_ALPHA * mid_price + (1 - OSMIUM_EMA_FAST_ALPHA) * ema_fast
        ema_slow = OSMIUM_EMA_SLOW_ALPHA * mid_price + (1 - OSMIUM_EMA_SLOW_ALPHA) * ema_slow

        mid_hist.append(mid_price)
        max_window = max(OSMIUM_FLUSH_W, 60)
        if len(mid_hist) > max_window:
            mid_hist = mid_hist[-max_window:]

        warmup += 1

        # ── fair value (regime-aware soft anchor) ──
        ema_gap = ema_fast - ema_slow
        if abs(ema_gap) <= OSMIUM_ANCHOR_BAND:
            fair = 0.5 * OSMIUM_ANCHOR + 0.5 * ema_slow
        else:
            fair = ema_fast

        # ── reference price with linear inventory skew ──
        ref = fair - OSMIUM_INV_SKEW_K * (position / POSITION_LIMIT)

        # ── stop-loss trigger evaluation ──
        # track down-streak: ema_fast < ema_slow AND mid < ema_fast
        if ema_fast < ema_slow and mid_price < ema_fast:
            down_streak += 1
        else:
            down_streak = 0
        # symmetric up-streak for short-side flush
        if ema_fast > ema_slow and mid_price > ema_fast:
            up_streak += 1
        else:
            up_streak = 0

        flush_long = False    # sell to flatten positive inventory
        flush_short = False   # buy to flatten negative inventory

        if warmup >= OSMIUM_WARMUP_TICKS and OSMIUM_FLUSH_MODE != "none":
            if OSMIUM_FLUSH_MODE == "ema_cross_consec":
                if down_streak >= OSMIUM_FLUSH_CONSEC and position > 0:
                    flush_long = True
                if up_streak >= OSMIUM_FLUSH_CONSEC and position < 0:
                    flush_short = True
            elif OSMIUM_FLUSH_MODE == "rolling_drop":
                w = min(OSMIUM_FLUSH_W, len(mid_hist))
                if w >= 2:
                    window = mid_hist[-w:]
                    w_max = max(window)
                    w_min = min(window)
                    if position > 0 and mid_price <= w_max - OSMIUM_FLUSH_DROP:
                        flush_long = True
                    if position < 0 and mid_price >= w_min + OSMIUM_FLUSH_DROP:
                        flush_short = True
            elif OSMIUM_FLUSH_MODE == "zscore_neg":
                w = min(OSMIUM_FLUSH_W, len(mid_hist))
                if w >= 2:
                    window = mid_hist[-w:]
                    m = sum(window) / w
                    v = sum((x - m) ** 2 for x in window) / w
                    s = math.sqrt(v) if v > 0 else 1e-9
                    z = (mid_price - m) / s
                    if position > 0 and z <= -OSMIUM_FLUSH_Z:
                        flush_long = True
                    if position < 0 and z >= OSMIUM_FLUSH_Z:
                        flush_short = True

        # ── persist state ──
        trader_state["osmium_ema_fast"] = ema_fast
        trader_state["osmium_ema_slow"] = ema_slow
        trader_state["osmium_mid_hist"] = mid_hist
        trader_state["osmium_down_streak"] = down_streak
        trader_state["osmium_up_streak"] = up_streak
        trader_state["osmium_warmup"] = warmup

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position
        bought = 0
        sold = 0

        # ── stop-loss: hard flush crossing the spread ──
        if flush_long and order_depth.buy_orders and position > 0:
            best_bid = max(order_depth.buy_orders.keys())
            bid_vol = abs(order_depth.buy_orders[best_bid])
            can_sell = min(bid_vol, position, max_sell - sold)
            if can_sell > 0:
                orders.append(Order(OSMIUM, best_bid, -can_sell))
                sold += can_sell

        if flush_short and order_depth.sell_orders and position < 0:
            best_ask = min(order_depth.sell_orders.keys())
            ask_vol = abs(order_depth.sell_orders[best_ask])
            can_buy = min(ask_vol, -position, max_buy - bought)
            if can_buy > 0:
                orders.append(Order(OSMIUM, best_ask, can_buy))
                bought += can_buy

        # ── warmup: skip aggressive taking; maker-only ──
        if warmup < OSMIUM_WARMUP_TICKS:
            bid_px = int(math.floor(mid_price)) - 1
            ask_px = int(math.ceil(mid_price)) + 1
            if bid_px >= ask_px:
                ask_px = bid_px + 2
            remaining_buy = max_buy - bought
            if remaining_buy > 0:
                orders.append(Order(OSMIUM, bid_px, remaining_buy))
            remaining_sell = max_sell - sold
            if remaining_sell > 0:
                orders.append(Order(OSMIUM, ask_px, -remaining_sell))
            return orders

        # ── taking: gates use ref (inventory-skewed), not fair ──
        # buy sweep
        sell_orders_sorted = sorted(order_depth.sell_orders.items())
        for ask_price, ask_vol in sell_orders_sorted:
            if ask_price > ref - 0.5:
                break
            can_buy = min(abs(ask_vol), max_buy - bought)
            if can_buy > 0:
                orders.append(Order(OSMIUM, ask_price, can_buy))
                bought += can_buy
            if bought >= max_buy:
                break

        # sell sweep
        buy_orders_sorted = sorted(order_depth.buy_orders.items(), reverse=True)
        for bid_price, bid_vol in buy_orders_sorted:
            if bid_price < ref + 0.5:
                break
            can_sell = min(abs(bid_vol), max_sell - sold)
            if can_sell > 0:
                orders.append(Order(OSMIUM, bid_price, -can_sell))
                sold += can_sell
            if sold >= max_sell:
                break

        # ── making: quotes pegged to ref; skew handles inventory ──
        bid_price = int(math.floor(ref)) - 1
        ask_price = int(math.ceil(ref)) + 1
        if bid_price >= ask_price:
            ask_price = bid_price + 2

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            overbid = best_bid + 1
            if overbid < ref:
                bid_price = max(bid_price, overbid)

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            underask = best_ask - 1
            if underask > ref:
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
