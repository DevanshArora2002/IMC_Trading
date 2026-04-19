"""Maker-aware replay harness for Round 2 traders.

Improves on backtest.py by also filling passive (maker) orders that beat
the existing BBO. Heuristic fill model (no future leak):

For each tick t with our orders O at prices p:
  • Take fills happen first (cross the existing book at tick t).
  • Maker fills are estimated by looking at the *next* tick's market
    activity proxy = number of asks/bids that crossed our quoted price
    in the depth snapshot at tick t+1. This approximates the bots
    "trading-on" our outstanding quote.
  • Specifically:
      - Our buy at price bp:  filled qty = min(buy_size,
                              sum of asks at next tick with ap <= bp)
      - Our sell at price sp: filled qty = min(sell_size,
                              sum of bids at next tick with bp >= sp)
  • This is a conservative approximation: in practice the market
    matching engine usually trades against the BBO.

Usage:
    python backtest_maker.py trader_3
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datamodel import Order, OrderDepth, TradingState  # noqa: E402

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80
DAYS = (-1, 0, 1)
PRICES_FMT = "prices_round_2_day_{}.csv"


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


def load_ticks_by_day() -> dict[int, list]:
    ticks_by_day: dict[int, list] = {}
    for d in DAYS:
        path = HERE / PRICES_FMT.format(d)
        df = pd.read_csv(path)
        df = df.sort_values(["timestamp", "product"]).reset_index(drop=True)
        ticks: dict[int, dict[str, OrderDepth]] = {}
        for _, row in df.iterrows():
            ts = int(row["timestamp"])
            product = row["product"]
            depth = _row_to_depth(row)
            if depth.buy_orders or depth.sell_orders:
                ticks.setdefault(ts, {})[product] = depth
        ordered = [(d, ts, depths) for ts, depths in sorted(ticks.items())]
        ticks_by_day[d] = ordered
    return ticks_by_day


def _take_match(order: Order, asks, bids, product, position, cash):
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
        return asks, new_bids, -(qty - (-need))
    return asks, bids, 0


def _maker_estimate_fill(o: Order, next_depth: OrderDepth):
    """Estimate fill of a passive maker order using next-tick depth.

    A passive buy at price bp is assumed filled by any asks in next_depth
    with ap <= bp (proxy for sellers willing to hit our quote).
    Symmetric for sells.
    """
    if next_depth is None:
        return 0
    qty = o.quantity
    if qty > 0:
        cross_vol = sum(
            -v for p, v in next_depth.sell_orders.items() if p <= o.price
        )
        return min(qty, cross_vol)
    elif qty < 0:
        cross_vol = sum(
            v for p, v in next_depth.buy_orders.items() if p >= o.price
        )
        return -min(-qty, cross_vol)
    return 0


def simulate_day(trader, ticks) -> dict:
    trader_data = ""
    position = {PEPPER: 0, OSMIUM: 0}
    cash = {PEPPER: 0.0, OSMIUM: 0.0}
    last_mid = {PEPPER: None, OSMIUM: None}

    n = len(ticks)
    for i in range(n):
        day, ts, depths = ticks[i]
        next_depths = ticks[i + 1][2] if i + 1 < n else {}

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

        for product, orders in result.items():
            depth = depths.get(product)
            if depth is None:
                continue
            asks = sorted((p, abs(v)) for p, v in depth.sell_orders.items())
            bids = sorted(
                ((p, abs(v)) for p, v in depth.buy_orders.items()), reverse=True
            )
            for o in orders:
                # Try take first
                asks, bids, taken = _take_match(o, asks, bids, product, position, cash)
                # Residual (passive) goes through maker model
                remaining_qty = o.quantity - taken
                if remaining_qty == 0:
                    continue
                resid = Order(o.symbol, o.price, remaining_qty)
                fill = _maker_estimate_fill(resid, next_depths.get(product))
                if fill > 0:
                    # We bought at our price (passive)
                    # Position-limit check
                    new_pos = position[product] + fill
                    if abs(new_pos) > POSITION_LIMIT:
                        fill = (POSITION_LIMIT * (1 if fill > 0 else -1)
                                - position[product])
                    position[product] += fill
                    cash[product] -= fill * o.price
                elif fill < 0:
                    new_pos = position[product] + fill
                    if abs(new_pos) > POSITION_LIMIT:
                        fill = (-POSITION_LIMIT - position[product])
                    position[product] += fill
                    cash[product] -= fill * o.price

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

    ticks_by_day = load_ticks_by_day()
    print(f"[maker-aware] tick counts: { {d: len(ticks_by_day[d]) for d in DAYS} }")

    rows = []
    for d in DAYS:
        trader = mod.Trader()
        r = simulate_day(trader, ticks_by_day[d])
        r["day"] = d
        rows.append(r)
        print(
            f"day {d:>3}: pepper={r['pnl_pepper']:>10.0f}  osmium={r['pnl_osmium']:>8.0f}  "
            f"total={r['pnl_total']:>10.0f}  end_pos=(P={r['end_pos_pepper']},O={r['end_pos_osmium']})"
        )

    totals = pd.DataFrame(rows)
    print("\n=== Totals (maker-aware) ===")
    print(f"  pepper : {totals['pnl_pepper'].sum():>10.0f}")
    print(f"  osmium : {totals['pnl_osmium'].sum():>10.0f}")
    print(f"  TOTAL  : {totals['pnl_total'].sum():>10.0f}")


if __name__ == "__main__":
    main()
