"""Round 2 trader v3: penny-jump market making + bid() method.

WHY v3? — diagnosis from log 305900 (trader_2)
──────────────────────────────────────────────
trader_2 made ONE Osmium fill all day (5 lots @ 10006). Position was
well-controlled (+5 vs trader_1's -76), but trader_2 essentially stopped
participating. Total profit fell from 7,656 → 7,371 because Osmium PnL
went from +278 → -16 while Pepper stayed flat (~7,380).

Live order-book reality (log 305900):
  • Median spread = 16 ticks (vs ~4 in historical CSVs).
  • EMA fair ≈ 10004; best ask consistently ≥ 10009; best bid ≤ 9999.
  • Edge >= 2 vs fair occurs in <2% of ticks ⇒ trader_2's entry filter
    is unreachable in this regime.
  • 0/46 market trades happened INSIDE the spread — every trade hit
    the BBO. So the BBO is the queue everyone takes from.
  • A naive penny-jumped maker (bid+1 / ask-1) would have collected
    108 buys + 111 sells with end position -3 and MTM PnL ≈ +1,411
    on this same 1,000-tick day (notebook log_analysis_305900.ipynb).

FIXES v2 → v3
─────────────
1. PENNY-JUMP MAKER QUOTES (was: passive at fair±2)
   Quote at best_bid+1 and best_ask-1 to be the BBO. Capture nearly the
   full ~14-tick spread on each round trip in the wide-spread regime.
   Falls back gracefully when the spread tightens.

2. INVENTORY SKEW (new)
   Shift both quotes by `position // SKEW_DIV` to bias toward flatten.
   Long ⇒ quote lower (sell sooner, buy later). Short ⇒ quote higher.

3. ADAPTIVE FAIR + ANTI-CROSS GUARD
   Keep EMA-of-mid fair from v2. Never quote a bid above fair_int or an
   ask below fair_int — protects from chasing a runaway move.

4. EDGE-1 TAKE LOGIC RE-ENABLED (was: skipped)
   With 80-lot cap and inventory skew on the maker side, edge-1 takes
   are safe and add cream when the spread *does* tighten.

5. POSITION-SCALED QUOTE SIZE
   Quote size shrinks as |position| nears the limit so we never get
   stuck without room to flatten.

6. bid() METHOD ADDED (new for Round 2)
   Returns an integer used by the Round-2 side game. Default 100 — a
   moderate bid intended to win small auctions without overpaying.
   Tune by changing BID_VALUE.

PEPPER LOGIC: unchanged (works as designed; PnL ~ drift × 80 × ticks).
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math

# ── Product names ────────────────────────────────────────────────────────
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── PEPPER parameters (unchanged) ───────────────────────────────────────
PEPPER_SLOPE = 0.001
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# ── OSMIUM parameters ──────────────────────────────────────────────────
FAIR_EMA_N = 30
FAIR_ALPHA = 2.0 / (FAIR_EMA_N + 1.0)

# Take only on obvious dislocations (edge ≥ TAKE_EDGE vs fair).
# In wide-spread regimes, small-edge takes cause the same one-sided
# inventory build-up that killed trader_1.
TAKE_EDGE = 4

# Maker offsets — penny-jump = +/-1 from BBO
MAKER_PENNY = 1

# Inventory skew: shift quotes by position // SKEW_DIV.
# Smaller divisor ⇒ more aggressive flatten via maker.
SKEW_DIV = 12

# Quote sizing — never quote more than this on either side.
# Set to POSITION_LIMIT to maximise fill capacity (single-tick fills are
# small in practice, this is just an upper bound).
MAX_MAKE_SIZE = 80

# When spread is tighter than this, fall back to fair-relative quotes
NARROW_SPREAD = 4

# ── BID for Round-2 side mechanic ──────────────────────────────────────
# Reasonable default; tune freely.
BID_VALUE = 45


class Trader:
    # ── Round-2 sealed bid ────────────────────────────────────────────
    def bid(self) -> int:
        """Required by Round 2. Returns an integer bid value used by the
        side mechanic. Ignored in all other rounds. See BID_VALUE."""
        return BID_VALUE

    # ── main entry ────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, data)

        return result, 0, self._serialize(data)

    # ── state helpers ─────────────────────────────────────────────────
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

    # ── INTARIAN_PEPPER_ROOT (unchanged) ──────────────────────────────
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
            orders.append(Order(PEPPER, int(0.95*math.floor(fair)), room))

        net_long = position + bought
        if net_long > 0:
            orders.append(
                Order(PEPPER, int(math.ceil(fair)) + PEPPER_OUTLIER_EDGE, -net_long)
            )

        return orders

    
    # ── ASH_COATED_OSMIUM: penny-jump market maker ────────────────────
    def _trade_osmium(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        ask_prices_hist = data.get("ask_prices_hist", [])
        bid_prices_hist = data.get("bid_prices_hist", [])

        ask_prices_hist.append(best_ask)
        bid_prices_hist.append(best_bid)
        if len(ask_prices_hist) > MAX_HISTORY_LENGTH:
            ask_prices_hist = ask_prices_hist[-MAX_HISTORY_LENGTH:]
            bid_prices_hist = bid_prices_hist[-MAX_HISTORY_LENGTH:]
        
        



        # ── adaptive fair (EMA of mid) ────────────────────────────────
        prev = data.get("osmium_fair")
        fair = mid if prev is None else FAIR_ALPHA * mid + (1.0 - FAIR_ALPHA) * prev
        data["osmium_fair"] = fair
        fair_int = int(round(fair))

        position = state.position.get(OSMIUM, 0)
        bought = 0
        sold = 0

        def proj():
            return position + bought - sold

        asks = sorted(depth.sell_orders.items())
        bids = sorted(depth.buy_orders.items(), reverse=True)

        # ── 1. TAKE crossable orders (edge ≥ TAKE_EDGE vs fair) ───────
        for ap, neg_av in asks:
            if ap > fair_int - TAKE_EDGE:
                break
            av = -neg_av
            room = POSITION_LIMIT - proj()
            if room <= 0:
                break
            qty = min(av, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(ap), qty))
                bought += qty

        for bp, bv in bids:
            if bp < fair_int + TAKE_EDGE:
                break
            room = POSITION_LIMIT + proj()
            if room <= 0:
                break
            qty = min(bv, room)
            if qty > 0:
                orders.append(Order(OSMIUM, int(bp), -qty))
                sold += qty

        # ── 2. PENNY-JUMP MAKER QUOTES (with inventory skew) ──────────
        cur_pos = proj()
        skew = cur_pos // SKEW_DIV  # long ⇒ +ve skew ⇒ shift quotes down

        if spread <= NARROW_SPREAD:
            # Tight market — fall back to fair-relative quotes
            bid_px = fair_int - 1 - skew
            ask_px = fair_int + 1 - skew
        else:
            # Wide market — penny-jump the BBO
            bid_px = best_bid + MAKER_PENNY - skew
            ask_px = best_ask - MAKER_PENNY - skew

        # Anti-cross guard: never quote across the fair value
        bid_px = min(bid_px, fair_int - 1)
        ask_px = max(ask_px, fair_int + 1)

        # Anti-self-cross
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        # Position-scaled sizing
        buy_room = POSITION_LIMIT - cur_pos
        sell_room = POSITION_LIMIT + cur_pos

        buy_size = min(buy_room, MAX_MAKE_SIZE)
        sell_size = min(sell_room, MAX_MAKE_SIZE)

        if buy_size > 0:
            orders.append(Order(OSMIUM, int(bid_px), buy_size))
        if sell_size > 0:
            orders.append(Order(OSMIUM, int(ask_px), -sell_size))

        return orders
