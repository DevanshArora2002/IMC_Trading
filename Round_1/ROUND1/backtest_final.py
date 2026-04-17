"""Taker-only replay harness for trader_final.Trader over days -2, -1, 0.

Fill model (matches trader_6_backtest):
- BUY @ P fills every ask level with price <= P at the ask's price, up to its volume.
- SELL @ P fills every bid level with price >= P at the bid's price, up to its volume.
- Passive (non-crossing) maker quotes are assumed unfilled. This under-estimates real PnL.

Run from this directory:
    python backtest_final.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datamodel  # noqa: E402
importlib.reload(datamodel)
from datamodel import Order, OrderDepth, TradingState  # noqa: E402

import trader_final  # noqa: E402
importlib.reload(trader_final)

PEPPER = trader_final.PEPPER
OSMIUM = trader_final.OSMIUM
POSITION_LIMIT = trader_final.POSITION_LIMIT

DAYS = (-2, -1, 0)
PRICES_FMT = "prices_round_1_day_{}.csv"


def load_ticks_by_day() -> dict[int, list[tuple[int, int, dict[str, OrderDepth]]]]:
    ticks_by_day: dict[int, list] = {}
    for d in DAYS:
        path = HERE / PRICES_FMT.format(d)
        df = pd.read_csv(path, sep=";") if _is_semicolon(path) else pd.read_csv(path)
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


def _is_semicolon(path: Path) -> bool:
    with path.open("r") as f:
        return ";" in f.readline()


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


def simulate_day(trader, ticks) -> dict:
    trader_data = ""
    position = {PEPPER: 0, OSMIUM: 0}
    cash = {PEPPER: 0.0, OSMIUM: 0.0}
    last_mid = {PEPPER: None, OSMIUM: None}
    max_dd = {PEPPER: 0.0, OSMIUM: 0.0}
    peak_pnl = {PEPPER: 0.0, OSMIUM: 0.0}

    for day, ts, depths in ticks:
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
            bids = sorted(((p, abs(v)) for p, v in depth.buy_orders.items()), reverse=True)
            for o in orders:
                asks, bids = _match(o, asks, bids, product, position, cash)

        for prod, depth in depths.items():
            bb = max(depth.buy_orders) if depth.buy_orders else None
            ba = min(depth.sell_orders) if depth.sell_orders else None
            if bb is not None and ba is not None:
                last_mid[prod] = (bb + ba) / 2
            elif bb is not None:
                last_mid[prod] = bb
            elif ba is not None:
                last_mid[prod] = ba

            pnl = cash[prod] + position[prod] * (last_mid[prod] or 0)
            peak_pnl[prod] = max(peak_pnl[prod], pnl)
            dd = peak_pnl[prod] - pnl
            max_dd[prod] = max(max_dd[prod], dd)

    final_pnl = {
        p: cash[p] + position[p] * (last_mid[p] or 0)
        for p in (PEPPER, OSMIUM)
    }
    return {
        "pnl_pepper": final_pnl[PEPPER],
        "pnl_osmium": final_pnl[OSMIUM],
        "pnl_total": final_pnl[PEPPER] + final_pnl[OSMIUM],
        "end_pos_pepper": position[PEPPER],
        "end_pos_osmium": position[OSMIUM],
        "max_dd_pepper": max_dd[PEPPER],
        "max_dd_osmium": max_dd[OSMIUM],
    }


def _match(order: Order, asks, bids, product, position, cash):
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
        return new_asks, bids
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
        return asks, new_bids
    return asks, bids


def main() -> None:
    importlib.reload(trader_final)
    ticks_by_day = load_ticks_by_day()
    print("tick counts:", {d: len(ticks_by_day[d]) for d in DAYS})

    rows = []
    for d in DAYS:
        trader = trader_final.Trader()
        r = simulate_day(trader, ticks_by_day[d])
        r["day"] = d
        rows.append(r)
        print(
            f"day {d:>3}: pepper={r['pnl_pepper']:>10.0f}  osmium={r['pnl_osmium']:>8.0f}  "
            f"total={r['pnl_total']:>10.0f}  end_pos=(P={r['end_pos_pepper']},O={r['end_pos_osmium']})  "
            f"max_dd=(P={r['max_dd_pepper']:.0f},O={r['max_dd_osmium']:.0f})"
        )

    totals = pd.DataFrame(rows)
    print("\n=== Totals ===")
    print(f"  pepper : {totals['pnl_pepper'].sum():>10.0f}")
    print(f"  osmium : {totals['pnl_osmium'].sum():>10.0f}")
    print(f"  TOTAL  : {totals['pnl_total'].sum():>10.0f}")
    print(f"  benchmark (trader.py) : 246001")


if __name__ == "__main__":
    main()
