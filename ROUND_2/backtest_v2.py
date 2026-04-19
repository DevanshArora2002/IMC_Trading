"""Trade-aware replay harness for Round 2.

Differences from backtest_maker.py:
  • Uses prices_round_2_day_X.csv for the depth + activity-based PnL.
  • Uses trades_round_2_day_X.csv for actual market trades that
    happened. For each market trade at price p that hit the existing
    BBO, our maker quote at penny-better can intercept it.

Fill model:
  • Take fills happen first (cross the existing book).
  • Maker fills:
      - For every market trade in the window [t, t+1):
          * If trade hit best_bid at t (someone sold to BBO):
              we'd be filled if we had a buy quote at price ≥ trade_px+1
              (we bid one tick higher than the BBO that got hit).
              Actually, more permissive: if our buy price ≥ trade_px,
              we're at least as good and intercept the trade.
          * If trade hit best_ask at t (someone bought BBO):
              filled if our sell price ≤ trade_px.
      - We only fill up to MIN(our remaining quote size, trade qty).
      - Position-limit enforced.

This better matches the live observation in 305900: penny-jumping the
BBO captures the wide spread.

Usage:
    python backtest_v2.py trader_3
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datamodel import Order, OrderDepth, TradingState  # noqa: E402

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80
DAYS = (-1, 0, 1)
PRICES_FMT = "prices_round_2_day_{}.csv"
TRADES_FMT = "trades_round_2_day_{}.csv"


def _row_to_depth(row) -> OrderDepth:
    depth = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv):
            depth.buy_orders[int(bp)] = int(bv)
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av):
            depth.sell_orders[int(ap)] = -int(abs(av))
    return depth


def load_data():
    days = {}
    for d in DAYS:
        prices = pd.read_csv(HERE / PRICES_FMT.format(d))
        prices = prices.sort_values(["timestamp", "product"]).reset_index(drop=True)
        ticks: dict[int, dict[str, OrderDepth]] = {}
        for _, row in prices.iterrows():
            ts = int(row["timestamp"])
            depth = _row_to_depth(row)
            if depth.buy_orders or depth.sell_orders:
                ticks.setdefault(ts, {})[row["product"]] = depth
        ordered = sorted(ticks.items())

        trades = pd.read_csv(HERE / TRADES_FMT.format(d), sep=";")
        trades_by_ts: dict[int, list[dict]] = defaultdict(list)
        for _, t in trades.iterrows():
            trades_by_ts[int(t["timestamp"])].append({
                "product": t["symbol"],
                "price": float(t["price"]),
                "qty": int(t["quantity"]),
            })
        days[d] = (ordered, trades_by_ts)
    return days


def _take_match(order, asks, bids, product, position, cash):
    qty = order.quantity
    if qty > 0:
        need = qty
        new_asks = []
        for ap, av in asks:
            if need <= 0 or ap > order.price:
                new_asks.append((ap, av))
                continue
            fill = min(need, av)
            position[product] += fill
            cash[product] -= fill * ap
            need -= fill
            av -= fill
            if av > 0:
                new_asks.append((ap, av))
        return new_asks, bids, qty - need
    elif qty < 0:
        need = -qty
        new_bids = []
        for bp, bv in bids:
            if need <= 0 or bp < order.price:
                new_bids.append((bp, bv))
                continue
            fill = min(need, bv)
            position[product] -= fill
            cash[product] += fill * bp
            need -= fill
            bv -= fill
            if bv > 0:
                new_bids.append((bp, bv))
        return asks, new_bids, -(qty - need)
    return asks, bids, 0


def simulate_day(trader, ordered_ticks, trades_by_ts):
    trader_data = ""
    position = {PEPPER: 0, OSMIUM: 0}
    cash = {PEPPER: 0.0, OSMIUM: 0.0}
    last_mid = {PEPPER: None, OSMIUM: None}

    n = len(ordered_ticks)
    for i in range(n):
        ts, depths = ordered_ticks[i]
        next_ts = ordered_ticks[i + 1][0] if i + 1 < n else ts + 100

        state = TradingState(
            traderData=trader_data,
            timestamp=ts,
            listings={},
            order_depths=depths,
            own_trades={},
            market_trades={},
            position=dict(position),
            observations=None,
        )
        result, _, trader_data = trader.run(state)

        # Group our orders by product
        my_orders_by_prod: dict[str, list[Order]] = defaultdict(list)
        for product, orders in result.items():
            for o in orders:
                my_orders_by_prod[product].append(o)

        for product, orders in my_orders_by_prod.items():
            depth = depths.get(product)
            if depth is None:
                continue
            asks = sorted((p, abs(v)) for p, v in depth.sell_orders.items())
            bids = sorted(
                ((p, abs(v)) for p, v in depth.buy_orders.items()), reverse=True
            )

            # Step 1: Take fills against existing book
            residual_buys: list[list] = []   # [price, qty]
            residual_sells: list[list] = []
            for o in orders:
                asks, bids, taken = _take_match(o, asks, bids, product, position, cash)
                left = o.quantity - taken
                if left > 0:
                    residual_buys.append([o.price, left])
                elif left < 0:
                    residual_sells.append([o.price, -left])

            # Step 2: Maker fills using market trades in [ts, next_ts)
            best_bid = max(depth.buy_orders) if depth.buy_orders else None
            best_ask = min(depth.sell_orders) if depth.sell_orders else None

            for tts in range(ts, next_ts):
                if tts not in trades_by_ts:
                    continue
                for tr in trades_by_ts[tts]:
                    if tr["product"] != product:
                        continue
                    px = tr["price"]
                    q = tr["qty"]
                    # If trade hit best_bid (or below) → seller hit BBO
                    # Our buys with price ≥ px can intercept.
                    if best_bid is not None and px <= best_bid:
                        # match against our residual buys priced ≥ px,
                        # in order from highest price first
                        residual_buys.sort(key=lambda x: -x[0])
                        for entry in residual_buys:
                            if q <= 0:
                                break
                            bp, bq = entry
                            if bp < px:
                                break
                            new_pos = position[product] + 1
                            room = POSITION_LIMIT - position[product]
                            if room <= 0:
                                break
                            fill = min(q, bq, room)
                            if fill <= 0:
                                continue
                            position[product] += fill
                            cash[product] -= fill * bp
                            entry[1] -= fill
                            q -= fill
                    # If trade hit best_ask (or above) → buyer hit BBO
                    if best_ask is not None and px >= best_ask:
                        residual_sells.sort(key=lambda x: x[0])
                        for entry in residual_sells:
                            if q <= 0:
                                break
                            sp, sq = entry
                            if sp > px:
                                break
                            room = POSITION_LIMIT + position[product]
                            if room <= 0:
                                break
                            fill = min(q, sq, room)
                            if fill <= 0:
                                continue
                            position[product] -= fill
                            cash[product] += fill * sp
                            entry[1] -= fill
                            q -= fill

        for prod, depth in depths.items():
            bb = max(depth.buy_orders) if depth.buy_orders else None
            ba = min(depth.sell_orders) if depth.sell_orders else None
            if bb is not None and ba is not None:
                last_mid[prod] = (bb + ba) / 2
            elif bb is not None:
                last_mid[prod] = bb
            elif ba is not None:
                last_mid[prod] = ba

    final_pnl = {
        p: cash[p] + position[p] * (last_mid[p] or 0) for p in (PEPPER, OSMIUM)
    }
    return {
        "pnl_pepper": final_pnl[PEPPER],
        "pnl_osmium": final_pnl[OSMIUM],
        "pnl_total": final_pnl[PEPPER] + final_pnl[OSMIUM],
        "end_pos_pepper": position[PEPPER],
        "end_pos_osmium": position[OSMIUM],
    }


def main() -> None:
    mod_name = sys.argv[1] if len(sys.argv) > 1 else "trader_3"
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)

    days = load_data()
    print(f"[trade-aware] tick counts: { {d: len(days[d][0]) for d in DAYS} }")

    rows = []
    for d in DAYS:
        trader = mod.Trader()
        ordered, trades_by_ts = days[d]
        r = simulate_day(trader, ordered, trades_by_ts)
        r["day"] = d
        rows.append(r)
        print(
            f"day {d:>3}: pepper={r['pnl_pepper']:>10.0f}  osmium={r['pnl_osmium']:>8.0f}  "
            f"total={r['pnl_total']:>10.0f}  end_pos=(P={r['end_pos_pepper']},O={r['end_pos_osmium']})"
        )

    totals = pd.DataFrame(rows)
    print("\n=== Totals (trade-aware) ===")
    print(f"  pepper : {totals['pnl_pepper'].sum():>10.0f}")
    print(f"  osmium : {totals['pnl_osmium'].sum():>10.0f}")
    print(f"  TOTAL  : {totals['pnl_total'].sum():>10.0f}")


if __name__ == "__main__":
    main()
