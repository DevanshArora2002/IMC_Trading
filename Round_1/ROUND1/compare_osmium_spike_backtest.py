"""Compare trader_final vs all Osmium variants (same pepper) on Round 1 prices."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backtest_final  # noqa: E402
import trader_final  # noqa: E402
import trader_osmium_v2  # noqa: E402

importlib.reload(backtest_final)
importlib.reload(trader_final)
importlib.reload(trader_osmium_v2)

STRATEGIES = [
    ("trader_final", trader_final),
    ("osmium_v2", trader_osmium_v2),
]


def main() -> None:
    ticks_by_day = backtest_final.load_ticks_by_day()
    days = backtest_final.DAYS

    all_rows = []
    for d in days:
        row = {"day": d}
        for name, mod in STRATEGIES:
            importlib.reload(mod)
            trader = mod.Trader()
            r = backtest_final.simulate_day(trader, ticks_by_day[d])
            row[f"{name}_total"] = r["pnl_total"]
            row[f"{name}_osmium"] = r["pnl_osmium"]
            row[f"{name}_pepper"] = r["pnl_pepper"]
            row[f"{name}_end_os"] = r["end_pos_osmium"]
            print(
                f"day {d:>3}  {name:<16} total={r['pnl_total']:>10.0f}  "
                f"osmium={r['pnl_osmium']:>8.0f}  pepper={r['pnl_pepper']:>10.0f}  "
                f"end_pos_os={r['end_pos_osmium']:>4}"
            )
        all_rows.append(row)
        print()

    df = pd.DataFrame(all_rows)
    print("=== Sums over all days ===")
    for name, _ in STRATEGIES:
        t = df[f"{name}_total"].sum()
        o = df[f"{name}_osmium"].sum()
        p = df[f"{name}_pepper"].sum()
        print(f"  {name:<16} total={t:>12.0f}  osmium={o:>10.0f}  pepper={p:>10.0f}")

    totals = [(n, df[f"{n}_total"].sum()) for n, _ in STRATEGIES]
    best = max(totals, key=lambda x: x[1])
    print(f"\nBest: {best[0]} ({best[1]:.0f})")


if __name__ == "__main__":
    main()
