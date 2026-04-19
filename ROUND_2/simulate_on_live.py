"""Replay a trader against the actual live log book + trade history.

Loads ROUND_2/logs_2/305900.log, reconstructs the depth at each tick,
and uses the recorded market trades to simulate maker fills (same fill
model as backtest_v2.py).

Usage:
    python simulate_on_live.py [trader_module]
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from collections import defaultdict
from io import StringIO

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datamodel import OrderDepth, TradingState, Order  # noqa: E402

POSITION_LIMIT = 80
LOG_PATH = HERE / "logs_2/305900.log"


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


def main():
    mod_name = sys.argv[1] if len(sys.argv) > 1 else "trader_3"
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
    trader = mod.Trader()

    with open(LOG_PATH) as f:
        log = json.load(f)

    activities = pd.read_csv(StringIO(log["activitiesLog"]), sep=";")
    activities = activities.sort_values(["timestamp", "product"]).reset_index(drop=True)

    ticks: dict[int, dict[str, OrderDepth]] = {}
    for _, row in activities.iterrows():
        ts = int(row["timestamp"])
        depth = _row_to_depth(row)
        if depth.buy_orders or depth.sell_orders:
            ticks.setdefault(ts, {})[row["product"]] = depth
    ordered = sorted(ticks.items())

    trades_by_ts: dict[int, list[dict]] = defaultdict(list)
    for tr in log["tradeHistory"]:
        # Use actual market trades only (exclude SUBMISSION's own original)
        if tr["buyer"] == "SUBMISSION" or tr["seller"] == "SUBMISSION":
            continue
        trades_by_ts[int(tr["timestamp"])].append({
            "product": tr["symbol"],
            "price": float(tr["price"]),
            "qty": int(tr["quantity"]),
        })

    trader_data = ""
    products = sorted({p for _, depths in ordered for p in depths})
    position = {p: 0 for p in products}
    cash = {p: 0.0 for p in products}
    last_mid = {p: None for p in products}

    n = len(ordered)
    pos_log = []
    for i in range(n):
        ts, depths = ordered[i]
        next_ts = ordered[i + 1][0] if i + 1 < n else ts + 100

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

            residual_buys: list[list] = []
            residual_sells: list[list] = []
            for o in orders:
                asks, bids, taken = _take_match(o, asks, bids, product, position, cash)
                left = o.quantity - taken
                if left > 0:
                    residual_buys.append([o.price, left])
                elif left < 0:
                    residual_sells.append([o.price, -left])

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
                    if best_bid is not None and px <= best_bid:
                        residual_buys.sort(key=lambda x: -x[0])
                        for entry in residual_buys:
                            if q <= 0:
                                break
                            bp, bq = entry
                            if bp < px:
                                break
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
        pos_log.append(dict(position))

    final_pnl = {p: cash[p] + position[p] * (last_mid[p] or 0) for p in products}
    print(f"\n=== {mod_name} on live 305900 ({n} ticks) ===")
    for p in products:
        print(f"  {p}: PnL={final_pnl[p]:>10.2f}  end_pos={position[p]:>4}")
    print(f"  TOTAL PnL: {sum(final_pnl.values()):>10.2f}")

    osm = "ASH_COATED_OSMIUM"
    if osm in products:
        op = pd.Series([p[osm] for p in pos_log])
        print(f"  Osmium pos stats: mean={op.mean():.1f} std={op.std():.1f} min={op.min()} max={op.max()}")


if __name__ == "__main__":
    main()
