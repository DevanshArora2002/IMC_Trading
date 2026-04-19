"""Generate ROUND_2/trader_5.py with osmium_ols_weights.json embedded.

The OLS weights are tiny (β ≈ 2W*2 floats, α 2 floats) so they fit
inline as a Python literal with no compression. Trader stays a single
self-contained file as required by IMC submissions.

Usage:
    python embed_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

WEIGHTS_PATH = HERE / "osmium_ols_weights.json"
OUT_PATH = HERE / "trader_5.py"


TEMPLATE = r'''"""Round 2 trader v5: trader_4 + dynamic MAKER_PENNY from a small OLS model.

WHY v5
──────
trader_4 used a constant `MAKER_PENNY = 1`. v5 keeps every other line
of trader_4 untouched and only changes the penny size based on a
spread forecast from a tiny OLS regression (replaces the CNN that the
original plan called for; same input/target structure, but ~1 KB of
weights instead of ~170 KB).

Model (offline-trained in train_osmium_ols.py, weights embedded below):
    input   x = [ mid_norm[t-W:t] ‖ spr_norm[t-W:t] ]    ∈ R^{2W}
    output  ŷ = x @ β + α                                ∈ R^2
            ŷ = (Δmid_norm_pred, spread_norm_pred) at t + HORIZON

    Per-window normalization (must match the trainer):
        mid_norm = (mid - mid[t-1]) / max(std(mid_window), 1e-3)
        spr_norm = (spread - mean(spread_window)) / max(std(spread_window), 1e-3)

    The spread head is de-normalized back to ticks before use.

Decisions
─────────
1. Penny size from predicted spread (widened from the plan after the
   first round of measurements: spreads on this book sit at ~16
   constantly, so the original ≤5/≤12 thresholds picked penny=3 every
   tick and gave up edge vs trader_4):

       pred_spread ≤ 15   → MAKER_PENNY = 1
       pred_spread ≤ 20   → MAKER_PENNY = 2
       pred_spread > 20   → MAKER_PENNY = 3

2. Directional quote shift from predicted Δmid (the OLS mid-delta head
   has real lift over naive on val, ~44%). When the model expects mid
   to rise we lift both quotes by 1 tick (and vice-versa). Capped at
   ±MID_SKEW_CAP ticks so it never overrides the inventory skew or the
   anti-cross guards.

Pepper logic, bid(), anti-cross guard, and inventory skew are kept
exactly as in trader_4.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ── Product names ────────────────────────────────────────────────────────
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── PEPPER parameters (inherited from trader_4) ─────────────────────────
PEPPER_SLOPE = 0.001
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# ── OSMIUM parameters ──────────────────────────────────────────────────
FAIR_EMA_N = 30
FAIR_ALPHA = 2.0 / (FAIR_EMA_N + 1.0)

# trader_4 fallback for cold start / numpy unavailable / forward error.
MAKER_PENNY_FALLBACK = 1

SKEW_DIV = 12
MAX_MAKE_SIZE = 80
NARROW_SPREAD = 4

# Directional skew: tick adjustment derived from the predicted mid
# move. `pred_ticks * MID_SKEW_GAIN` is rounded then clipped to
# ±MID_SKEW_CAP. Conservative defaults; raise GAIN to lean harder on
# the prediction.
MID_SKEW_GAIN = 0.5
MID_SKEW_CAP = 1
# Only react to predictions of meaningful size (in ticks).
MID_SKEW_DEADBAND = 0.5

# ── BID for Round-2 side mechanic ──────────────────────────────────────
BID_VALUE = 30

# ── Embedded OLS model (produced by train_osmium_ols.py) ───────────────
_WEIGHTS_JSON = __WEIGHTS_JSON__

_MODEL = json.loads(_WEIGHTS_JSON)
_ARCH = _MODEL["arch"]
_W = _MODEL["weights"]

_WINDOW = _ARCH["window"]
_HORIZON = _ARCH["horizon"]

if _HAS_NUMPY:
    _BETA = np.asarray(_W["beta"], dtype=np.float64)    # (2W, 2)
    _ALPHA = np.asarray(_W["alpha"], dtype=np.float64)  # (2,)


# ── numpy-only forward pass (matmul + bias) ────────────────────────────
def _forward(mid_hist, spr_hist):
    """Run the OLS model on one window.

    Returns (pred_mid_delta_ticks, pred_spread_ticks) — both in *tick*
    units (already de-normalized), ready for the decision logic.
    """
    mid = np.asarray(mid_hist, dtype=np.float64)
    spr = np.asarray(spr_hist, dtype=np.float64)

    ref_mid = float(mid[-1])
    mid_std = max(float(mid.std()), 1e-3)
    mid_norm = (mid - ref_mid) / mid_std

    spr_mean = float(spr.mean())
    spr_std = max(float(spr.std()), 1e-3)
    spr_norm = (spr - spr_mean) / spr_std

    x = np.concatenate([mid_norm, spr_norm])    # (2W,)
    y = x @ _BETA + _ALPHA                      # (2,)

    pred_mid_delta = float(y[0]) * mid_std
    pred_spread = float(y[1]) * spr_std + spr_mean
    return pred_mid_delta, pred_spread


def _pick_penny(pred_spread: float) -> int:
    """Dynamic maker-penny from predicted spread.

    Thresholds widened after live measurement: this book's spread sits
    near 16 nearly always, so under the original ≤5/≤12 cuts we'd
    pay penny=3 forever and bleed edge vs trader_4.
    """
    if pred_spread <= 15.0:
        return 1
    if pred_spread <= 20.0:
        return 2
    return 3


def _dir_shift(pred_mid_delta: float) -> int:
    """Tick adjustment from the predicted mid move.

    Positive = mid expected to rise → lift both quotes; negative =
    drop both quotes. Deadbanded so we don't react to tiny noise, then
    clipped to ±MID_SKEW_CAP.
    """
    if abs(pred_mid_delta) < MID_SKEW_DEADBAND:
        return 0
    raw = round(pred_mid_delta * MID_SKEW_GAIN)
    if raw > MID_SKEW_CAP:
        return MID_SKEW_CAP
    if raw < -MID_SKEW_CAP:
        return -MID_SKEW_CAP
    return int(raw)


class Trader:
    # ── Round-2 sealed bid ────────────────────────────────────────────
    def bid(self) -> int:
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

    # ── INTARIAN_PEPPER_ROOT (inherited from trader_4) ────────────────
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
            orders.append(Order(PEPPER, int(math.floor(fair)), room))

        net_long = position + bought
        if net_long > 0:
            orders.append(
                Order(PEPPER, int(math.ceil(fair)) + PEPPER_OUTLIER_EDGE, -net_long)
            )

        return orders

    # ── ASH_COATED_OSMIUM: penny-jump maker w/ OLS-driven penny size ──
    def _trade_osmium(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        prev = data.get("osmium_fair")
        fair = mid if prev is None else FAIR_ALPHA * mid + (1.0 - FAIR_ALPHA) * prev
        data["osmium_fair"] = fair
        fair_int = int(round(fair))

        # Rolling buffers for the OLS forecast.
        mid_hist = data.get("osmium_mid_hist") or []
        spr_hist = data.get("osmium_spr_hist") or []
        mid_hist.append(mid)
        spr_hist.append(spread)
        if len(mid_hist) > _WINDOW:
            mid_hist = mid_hist[-_WINDOW:]
            spr_hist = spr_hist[-_WINDOW:]
        data["osmium_mid_hist"] = mid_hist
        data["osmium_spr_hist"] = spr_hist

        # OLS forecast → dynamic penny size + directional tick shift.
        # Cold start (or numpy missing / forward error) falls back to
        # trader_4's static behavior.
        maker_penny = MAKER_PENNY_FALLBACK
        dir_shift = 0
        if _HAS_NUMPY and len(mid_hist) >= _WINDOW:
            try:
                pred_mid_delta, pred_spread = _forward(mid_hist, spr_hist)
                maker_penny = _pick_penny(pred_spread)
                dir_shift = _dir_shift(pred_mid_delta)
                data["osmium_last_pred_spread"] = pred_spread
                data["osmium_last_pred_mid_delta"] = pred_mid_delta
                data["osmium_last_penny"] = maker_penny
                data["osmium_last_dir_shift"] = dir_shift
            except Exception:
                maker_penny = MAKER_PENNY_FALLBACK
                dir_shift = 0

        position = state.position.get(OSMIUM, 0)
        skew = position // SKEW_DIV

        if spread <= NARROW_SPREAD:
            bid_px = fair_int - 1 - skew + dir_shift
            ask_px = fair_int + 1 - skew + dir_shift
        else:
            bid_px = best_bid + maker_penny - skew + dir_shift
            ask_px = best_ask - maker_penny - skew + dir_shift

        # Anti-cross guard: never quote across the fair value.
        bid_px = min(bid_px, fair_int - 1)
        ask_px = max(ask_px, fair_int + 1)

        # Anti-self-cross.
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_room = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        buy_size = min(buy_room, MAX_MAKE_SIZE)
        sell_size = min(sell_room, MAX_MAKE_SIZE)

        if buy_size > 0:
            orders.append(Order(OSMIUM, int(bid_px), buy_size))
        if sell_size > 0:
            orders.append(Order(OSMIUM, int(ask_px), -sell_size))

        return orders
'''


def main() -> None:
    with open(WEIGHTS_PATH) as f:
        weights = json.load(f)
    # Drop the sweep table from the embedded blob — only arch + weights
    # + a small training summary need to ride along at runtime.
    embed_blob = {
        "arch": weights["arch"],
        "weights": weights["weights"],
        "train": weights.get("train", {}),
    }
    compact = json.dumps(embed_blob, separators=(",", ":"))
    assert "'''" not in compact
    embedded = "r'''" + compact + "'''"
    src = TEMPLATE.replace("__WEIGHTS_JSON__", embedded)
    with open(OUT_PATH, "w") as f:
        f.write(src)
    print(
        f"wrote {OUT_PATH}  ({len(src)} bytes)  "
        f"(W={weights['arch']['window']}, H={weights['arch']['horizon']})"
    )


if __name__ == "__main__":
    main()
