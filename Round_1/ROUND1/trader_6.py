from datamodel import OrderDepth, TradingState, Order
from typing import Optional
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── OSMIUM tunables (regime-aware mean reversion + inventory skew) ─────
OSMIUM_ANCHOR = 10_000
OSMIUM_ANCHOR_BAND = 1.5          # ticks; soft anchor only inside this band
OSMIUM_EMA_FAST_ALPHA = 0.20
OSMIUM_EMA_SLOW_ALPHA = 0.05
OSMIUM_INV_SKEW_K = 2.0           # ticks shifted per full position
OSMIUM_WARMUP_TICKS = 20          # maker-only until warm

# Stop-loss trigger family switch: "ema_cross_consec" | "rolling_drop" | "zscore_neg" | "none"
# Winner from trader_6_backtest.ipynb (days -2,-1,0, taker-only fill model):
#   mode="none" ranked #1 with osmium_total=13,551.5 across 3 days
#   (day-2: 3710, day-1: 4861, day0: 4980; worst DD -1788).
#   Every flush family produced negative OSMIUM PnL (-12k to -30k) because the
#   taker-only model penalises spread-crossing. Regime-aware fair + inventory skew
#   alone provide sufficient downside defence without the flush cost.
#   Stop-loss code removed; kept as reference in trader_6_backtest.ipynb.


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

        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            vol = abs(ask_vol)
            if vol > 0 and max_buy > 0:
                buy_vol = min(vol, max_buy)
                orders.append(Order(PEPPER, ask_price, buy_vol))
                max_buy -= buy_vol
                if max_buy <= 0:
                    break

        return orders

    # ── ASH_COATED_OSMIUM: regime-aware fair + inventory skew ──────────

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
        warmup = int(trader_state.get("osmium_warmup", 0))

        # ── update indicators ──
        ema_fast = OSMIUM_EMA_FAST_ALPHA * mid_price + (1 - OSMIUM_EMA_FAST_ALPHA) * ema_fast
        ema_slow = OSMIUM_EMA_SLOW_ALPHA * mid_price + (1 - OSMIUM_EMA_SLOW_ALPHA) * ema_slow

        warmup += 1

        # ── fair value (regime-aware soft anchor) ──
        ema_gap = ema_fast - ema_slow
        if abs(ema_gap) <= OSMIUM_ANCHOR_BAND:
            fair = 0.5 * OSMIUM_ANCHOR + 0.5 * ema_slow
        else:
            fair = ema_fast

        # ── reference price with linear inventory skew ──
        ref = fair - OSMIUM_INV_SKEW_K * (position / POSITION_LIMIT)

        # ── persist state ──
        trader_state["osmium_ema_fast"] = ema_fast
        trader_state["osmium_ema_slow"] = ema_slow
        trader_state["osmium_warmup"] = warmup

        max_buy = POSITION_LIMIT - position
        max_sell = POSITION_LIMIT + position
        bought = 0
        sold = 0

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
