"""Offline OLS trainer for the Osmium spread predictor.

Replaces the earlier CNN with a plain linear model so the embedded
weights stay well under ~1 KB (critical for the single-file submission
size budget).

Model
─────
Per tick t, inputs are a flattened window of length 2*W:

    x = [ mid_norm[t-W : t]  ‖  spr_norm[t-W : t] ]     ∈ R^{2W}

Normalization matches trader_5 exactly:
    mid_norm = (mid - mid[t-1]) / max(std(mid_window), 1e-3)
    spr_norm = (spr - mean(spr_window)) / max(std(spr_window), 1e-3)

Targets (2-dim):
    y = [ Δmid_norm @ t+H ,  spread_norm @ t+H ]        ∈ R^2

OLS fit:                       ŷ = x β + α
    β  shape (2W, 2)
    α  shape (2,)

Sweep over W ∈ W_CHOICES × H ∈ H_CHOICES, pick the (W,H) with the best
*validation spread MSE* (that's the head trader_5 actually uses).

Output: ROUND_2/osmium_ols_weights.json
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

OSMIUM = "ASH_COATED_OSMIUM"
DAYS_TRAIN = (-1, 0)
DAYS_VAL = (1,)

W_CHOICES = (16, 24, 32, 48, 64, 96, 128)
H_CHOICES = (1, 3, 5, 8)

# Small ridge term to stabilize near-singular Gram matrices when the
# window is wide and mostly constant (spread is basically stationary on
# this book). Set to 0 for pure OLS.
RIDGE = 1e-6


# ── data ───────────────────────────────────────────────────────────────
def load_osmium_series(day: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(HERE / f"prices_round_2_day_{day}.csv")
    df = df[df["product"] == OSMIUM].sort_values("timestamp").reset_index(drop=True)
    mid = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    spr = df["ask_price_1"] - df["bid_price_1"]
    m = mid.notna() & spr.notna()
    return mid[m].to_numpy(dtype=np.float64), spr[m].to_numpy(dtype=np.float64)


def build_windows(
    mid: np.ndarray, spr: np.ndarray, W: int, H: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y). X:(N,2W), Y:(N,2). Also Y is in normalized units."""
    n = len(mid)
    last = n - H
    if last <= W:
        return np.zeros((0, 2 * W)), np.zeros((0, 2))
    xs = np.empty((last - W, 2 * W), dtype=np.float64)
    ys = np.empty((last - W, 2), dtype=np.float64)
    for i, t in enumerate(range(W, last)):
        w_mid = mid[t - W : t]
        w_spr = spr[t - W : t]
        ref_mid = w_mid[-1]
        mid_std = max(float(w_mid.std()), 1e-3)
        mid_norm = (w_mid - ref_mid) / mid_std
        spr_mean = float(w_spr.mean())
        spr_std = max(float(w_spr.std()), 1e-3)
        spr_norm = (w_spr - spr_mean) / spr_std
        xs[i, :W] = mid_norm
        xs[i, W:] = spr_norm
        ys[i, 0] = (mid[t + H] - ref_mid) / mid_std
        ys[i, 1] = (spr[t + H] - spr_mean) / spr_std
    return xs, ys


def build_split(days: tuple[int, ...], W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    Xp, Yp = [], []
    for d in days:
        mid, spr = load_osmium_series(d)
        X, Y = build_windows(mid, spr, W, H)
        Xp.append(X)
        Yp.append(Y)
    return np.concatenate(Xp, axis=0), np.concatenate(Yp, axis=0)


# ── fit ────────────────────────────────────────────────────────────────
def ols_fit(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve (X_aug'X_aug + λI) θ = X_aug'Y with an intercept column.

    Returns β shape (2W, 2) and α shape (2,).
    """
    N, D = X.shape
    X_aug = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])
    A = X_aug.T @ X_aug
    if RIDGE > 0:
        A += RIDGE * np.eye(D + 1, dtype=X.dtype)
        # don't regularize intercept
        A[-1, -1] -= RIDGE
    theta = np.linalg.solve(A, X_aug.T @ Y)  # (D+1, 2)
    beta = theta[:-1, :]
    alpha = theta[-1, :]
    return beta, alpha


def mse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    m = float(((pred[:, 0] - target[:, 0]) ** 2).mean())
    s = float(((pred[:, 1] - target[:, 1]) ** 2).mean())
    return m, s


# ── sweep ──────────────────────────────────────────────────────────────
def main() -> None:
    results: list[dict] = []
    best: dict | None = None

    # Pre-load the per-day series once per W (re-used across H values).
    train_series = [load_osmium_series(d) for d in DAYS_TRAIN]
    val_series = [load_osmium_series(d) for d in DAYS_VAL]

    print(f"{'W':>4} {'H':>3}  {'N_tr':>6}  {'val_mid_mse':>12}  {'val_spr_mse':>12}"
          f"  {'naive_spr':>10}  lift%")
    print("-" * 72)

    for W, H in product(W_CHOICES, H_CHOICES):
        Xtr = np.concatenate(
            [build_windows(m, s, W, H)[0] for (m, s) in train_series], axis=0
        )
        Ytr = np.concatenate(
            [build_windows(m, s, W, H)[1] for (m, s) in train_series], axis=0
        )
        Xva = np.concatenate(
            [build_windows(m, s, W, H)[0] for (m, s) in val_series], axis=0
        )
        Yva = np.concatenate(
            [build_windows(m, s, W, H)[1] for (m, s) in val_series], axis=0
        )
        if len(Xtr) == 0 or len(Xva) == 0:
            continue

        beta, alpha = ols_fit(Xtr, Ytr)
        pred = Xva @ beta + alpha
        m_mse, s_mse = mse(pred, Yva)

        naive_spr = float((Yva[:, 1] ** 2).mean())
        naive_mid = float((Yva[:, 0] ** 2).mean())
        lift = 100.0 * (naive_spr - s_mse) / max(naive_spr, 1e-12)

        row = {
            "W": W, "H": H, "N_train": len(Xtr), "N_val": len(Xva),
            "val_mid_mse": m_mse, "val_spr_mse": s_mse,
            "naive_val_mid_mse": naive_mid, "naive_val_spr_mse": naive_spr,
            "lift_spr_pct": lift,
            "beta": beta, "alpha": alpha,
        }
        results.append(row)

        print(f"{W:>4} {H:>3}  {len(Xtr):>6}  {m_mse:>12.5f}  {s_mse:>12.5f}"
              f"  {naive_spr:>10.5f}  {lift:>+5.2f}%")

        if best is None or s_mse < best["val_spr_mse"]:
            best = row

    assert best is not None
    print("-" * 72)
    print(f"best: W={best['W']} H={best['H']}  "
          f"val_spr_mse={best['val_spr_mse']:.5f}  "
          f"lift={best['lift_spr_pct']:+.2f}% vs naive")

    # ── save ──────────────────────────────────────────────────────────
    out = {
        "arch": {
            "window": best["W"],
            "horizon": best["H"],
            "channels": 2,   # flatten-order: [mid_norm * W ‖ spr_norm * W]
            "outputs": 2,    # (Δmid_norm, spread_norm)
        },
        "weights": {
            "beta": best["beta"].astype(np.float64).tolist(),   # (2W, 2)
            "alpha": best["alpha"].astype(np.float64).tolist(), # (2,)
        },
        "train": {
            "ridge": RIDGE,
            "days_train": list(DAYS_TRAIN),
            "days_val": list(DAYS_VAL),
            "val_mid_mse": best["val_mid_mse"],
            "val_spr_mse": best["val_spr_mse"],
            "naive_val_mid_mse": best["naive_val_mid_mse"],
            "naive_val_spr_mse": best["naive_val_spr_mse"],
            "lift_spr_pct": best["lift_spr_pct"],
        },
        "sweep": [
            {k: v for k, v in r.items() if k not in ("beta", "alpha")}
            for r in results
        ],
    }
    path = HERE / "osmium_ols_weights.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
