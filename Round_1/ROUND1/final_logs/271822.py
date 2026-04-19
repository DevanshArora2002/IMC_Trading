from datamodel import OrderDepth, TradingState, Order
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# PEPPER: linear drift of roughly +0.001 per timestamp unit anchored on first-tick mid.
PEPPER_SLOPE = 0.001
# Must be wide enough to cross best_ask on tick 0 (spread ≈ 12-14, so best_ask
# sits ~6-7 above fair). Buffer 8 maximises 3-day backtest PnL.
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# OSMIUM: mean-reverts tightly around 10000 with a ~16-wide book.
OSMIUM_FAIR = 10_000
OSMIUM_TAKE_ASK_MAX = OSMIUM_FAIR - 1
OSMIUM_TAKE_BID_MIN = OSMIUM_FAIR + 1
OSMIUM_MAKER_BUY_FLOOR = 9_996
OSMIUM_MAKER_BUY_CAP = 9_999
OSMIUM_MAKER_SELL_FLOOR = 10_001
OSMIUM_MAKER_SELL_CAP = 10_004


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state)

        return result, 0, self._serialize(data)

    # ── state helpers ────────────────────────────────────────────────────

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

    # ── INTARIAN_PEPPER_ROOT: aggressive trend-follow to +80 ─────────────

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
        # Take asks aggressively up to fair + buffer.
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

        # Passive top-up bid at fair.
        room = POSITION_LIMIT - position - bought
        if room > 0:
            orders.append(Order(PEPPER, int(math.floor(fair)), room))

        # Outlier-bid harvester: offer small size above fair to harvest spikes.
        net_long = position + bought
        if net_long > 0:
            orders.append(Order(PEPPER, int(math.ceil(fair)) + PEPPER_OUTLIER_EDGE, -net_long))

        return orders

    # ── ASH_COATED_OSMIUM: aggressive two-sided MM ───────────────────────

    def _trade_osmium(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        position = state.position.get(OSMIUM, 0)
        bought = 0
        sold = 0

        # Take any ask priced below fair.
        for ask in sorted(depth.sell_orders.keys()):
            if ask > OSMIUM_TAKE_ASK_MAX:
                break
            room = POSITION_LIMIT - position - bought
            if room <= 0:
                break
            ask_vol = -depth.sell_orders[ask]
            qty = min(ask_vol, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(ask), qty))
                bought += qty

        # Hit any bid priced above fair.
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid < OSMIUM_TAKE_BID_MIN:
                break
            room = POSITION_LIMIT + position - sold
            if room <= 0:
                break
            bid_vol = depth.buy_orders[bid]
            qty = min(bid_vol, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(bid), -qty))
                sold += qty

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        buy_px = min(max(best_bid + 1, OSMIUM_MAKER_BUY_FLOOR), OSMIUM_MAKER_BUY_CAP)
        sell_px = max(min(best_ask - 1, OSMIUM_MAKER_SELL_CAP), OSMIUM_MAKER_SELL_FLOOR)

        buy_room = POSITION_LIMIT - position - bought
        sell_room = POSITION_LIMIT + position - sold

        if buy_room > 0:
            orders.append(Order(OSMIUM, int(buy_px), buy_room))
        if sell_room > 0:
            orders.append(Order(OSMIUM, int(sell_px), -sell_room))

        return orders