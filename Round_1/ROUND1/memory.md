# Round 1 — strategy memory (`trader.py`)

Concise reference for what this round's bot does. **All behaviour below is taken from** `Round_1/ROUND1/trader.py` **only.**

## Gist

- Two products: **`INTARIAN_PEPPER_ROOT`** and **`ASH_COATED_OSMIUM`**, each with **`POSITION_LIMIT = 80`**.
- **`run`** builds orders per product if it appears in `state.order_depths`, serialises state to **`traderData`** as JSON, returns `(result, 0, traderData)`.

## State (`traderData`)

- **`osmium_ema_fast`**: fast EMA of mid (alpha = 0.15, ~13-period equivalent).
- **`osmium_ema_slow`**: slow EMA of mid (alpha = 0.05, ~39-period equivalent).
- **`osmium_history`**: last **`OSMIUM_Z_WINDOW`** (40) mid prices (used for Z-score calculation).

## INTARIAN_PEPPER_ROOT

- **Goal:** buy-and-hold — only buy, never sell.
- **`max_buy = 80 - position`**. If `max_buy <= 0` or no asks in book, return no orders.
- **Single order at best ask:** place one LIMIT buy at `min(sell_orders.keys())` for `min(volume_at_best_ask, max_buy)`.
- No ask-level sweep, no passive maker orders.

## ASH_COATED_OSMIUM

### Indicators (computed each tick)

- **Mid:** `(best_bid + best_ask) / 2` if both sides exist; else lone best bid or best ask; if no book, skip.
- **Dual EMAs:** `ema_fast` (alpha 0.15) and `ema_slow` (alpha 0.05), both updated as `alpha * mid + (1 - alpha) * ema`. Seeded from first mid price.
- **Z-score:** `z = (mid - mean(history)) / std(history)` over the last 40 mid prices (population std; floor std at 1e-9). Z-score is 0 until at least 2 data points.

### Reversal signals

All three conditions must hold simultaneously:

- **Buy signal:** `z < -1.5` AND `ema_fast > ema_slow` (short-term recovering) AND `mid > ema_fast` (price lifting above fast average).
- **Sell signal:** `z > +1.5` AND `ema_fast < ema_slow` (short-term falling) AND `mid < ema_fast` (price dropping below fast average).

### Fair value

`fair_value = 0.5 * 10_000 + 0.5 * ema_slow` — blended anchor + slow EMA, used for both taking thresholds and maker quote placement.

### Taking (gated by signals)

- **On buy_signal:** sweep asks at or below `fair_value`, buying up to remaining capacity.
- **On sell_signal:** hit bids at or above `fair_value`, selling up to remaining capacity.
- **No signal:** no aggressive taking at all.

### Making (every tick)

- `bid_price = floor(fair_value) - 1`, `ask_price = ceil(fair_value) + 1`; if equal, bump `ask_price` by 1.
- If `best_bid + 1 < fair_value`, raise `bid_price` to that overbid.
- If `best_ask - 1 > fair_value`, lower `ask_price` to that under-ask.
- Post remaining buy and sell sizes (full leftover capacity after any taking).

## Constants

| Name | Value | Purpose |
|---|---|---|
| `POSITION_LIMIT` | 80 | Per-product position cap |
| `OSMIUM_ANCHOR` | 10,000 | Long-run price anchor |
| `OSMIUM_ANCHOR_WEIGHT` | 0.5 | Anchor weight in fair value |
| `OSMIUM_EMA_WEIGHT` | 0.5 | Slow EMA weight in fair value |
| `OSMIUM_EMA_FAST_ALPHA` | 0.15 | Fast EMA smoothing (~13-period) |
| `OSMIUM_EMA_SLOW_ALPHA` | 0.05 | Slow EMA smoothing (~39-period) |
| `OSMIUM_Z_ENTRY` | 1.5 | Z-score threshold for reversal signals |
| `OSMIUM_Z_WINDOW` | 40 | Rolling window length for Z-score |

## Position / risk (per tick)

- Pepper and Osmium both respect **`POSITION_LIMIT`** via **`max_buy` / `max_sell`** and the taking loops + final maker sizes; Osmium also tracks **`bought`** / **`sold`** within the tick.

## Recorded PnL (submission export, not defined in code)

- One finished Round 1 submission log in this folder reports **`profit`: ~8414.59** (`submission_1/166849.json`). Interpretation of that number is platform-defined; the bot logic is entirely as above.

---

# trader_3.py — revised strategies

## Gist

Same two products, same `POSITION_LIMIT = 80`. Simpler, more aggressive approach for both.

## State (`traderData`)

No persisted state — both strategies are stateless (all decisions based on current tick's order book and timestamp).

## INTARIAN_PEPPER_ROOT

### Price equations (observed on day 0 data)

- **ask_eq** = `12006 + 0.05 * t` (where `t = state.timestamp`)
- **bid_eq** = `11991 + 0.05 * t`
- Normal spread ≈ 15–16; spikes compress it below 7.

### Two-phase logic

1. **Phase 1 — accumulate:** while `position < 80`, buy at the best ask (single LIMIT order at `min(sell_orders)`, volume capped by remaining capacity). Same as trader.py Pepper logic.
2. **Phase 2 — volatility spikes:** when `best_ask - best_bid < 7` (spread collapse), check:
   - **Bid above equation** (`best_bid > bid_eq`): sell at `best_bid` — the bid is spiking, lock in profit.
   - **Ask below equation** (`best_ask < ask_eq`): buy at `best_ask` — the ask is dipping, grab cheap shares.
   - Both can fire in the same tick if conditions hold (position limits still enforced).

### Constants

| Name | Value |
|---|---|
| `PEPPER_ASK_INTERCEPT` | 12,006 |
| `PEPPER_BID_INTERCEPT` | 11,991 |
| `PEPPER_SLOPE` | 0.05 |
| `PEPPER_SPREAD_THRESHOLD` | 7 |

## ASH_COATED_OSMIUM

### Strategy: hard mean-reversion to 10,000

No EMAs, no Z-scores. Pure anchor trading:

- **Taking — buy:** sweep all asks at or below 10,000 (up to remaining buy capacity).
- **Taking — sell:** hit all bids at or above 10,000 (up to remaining sell capacity).
- **Making:** post passive bid at 9,999 and passive ask at 10,001 with all remaining capacity after taking. This captures spread when price hovers near the anchor.

### Constants

| Name | Value |
|---|---|
| `OSMIUM_ANCHOR` | 10,000 |

---

# trader_6.py — regime-aware OSMIUM with inventory skew + pluggable stop-loss

## Root cause fixed (from log `logs2/169826.json`, diagnosed against `trader.py`)

OSMIUM drew ~150 units while mid drifted 10002 → 9996 over `t=80000..84000` carrying +80 inventory. In `trader.py`:

- `fair_value = 0.5*10_000 + 0.5*ema_slow` stays pinned near 10,000 so the sell-take gate `bid_price < fair_value` blocks every hit while bids sit at 9987–9994.
- Passive maker bid = `floor(fair_value)-1 = 9999` keeps absorbing the dip, inventory climbs to +80.
- No inventory skew and no stop-loss → a pure mean-reverter bleeds on any 4–8 tick drift.

`trader_6.py` keeps PEPPER identical to `trader.py` and rewrites OSMIUM.

## State (`traderData`, OSMIUM only)

- `osmium_ema_fast` (alpha = 0.20, faster than trader.py's 0.15 for quicker regime detection).
- `osmium_ema_slow` (alpha = 0.05, same as trader.py).
- `osmium_mid_hist` — rolling mid prices, length = `max(OSMIUM_FLUSH_W, 60)`.
- `osmium_down_streak`, `osmium_up_streak` — consecutive-tick counters for `ema_cross_consec` family.
- `osmium_warmup` — tick counter; strategy is maker-only until `warmup >= OSMIUM_WARMUP_TICKS` (default 20).

## OSMIUM logic (per tick, after warm-up)

### Regime-aware fair value

- `ema_gap = ema_fast - ema_slow`.
- If `|ema_gap| <= OSMIUM_ANCHOR_BAND` (1.5 ticks, calm regime): `fair = 0.5*10_000 + 0.5*ema_slow` (trader.py's blend).
- Else (trending regime): `fair = ema_fast` — **the 10,000 anchor is dropped** so the fair tracks real drift and sell-taking unblocks.

### Linear inventory skew

- `ref = fair - OSMIUM_INV_SKEW_K * (position / POSITION_LIMIT)`, with `OSMIUM_INV_SKEW_K = 2.0`.
- Long inventory pulls both taking gates and maker quotes down, automatically encouraging sells + discouraging more buys.

### Taking gates use `ref` (not `fair`)

- Buy: sweep asks with `ask_price <= ref - 0.5` up to remaining buy capacity.
- Sell: hit bids with `bid_price >= ref + 0.5` up to remaining sell capacity.
- No separate "signal" gate — the skew + regime handle direction; the 0.5 buffer is the spread capture requirement.

### Making

- Post `bid = floor(ref) - 1`, `ask = ceil(ref) + 1` with remaining capacity.
- Peg to `best_bid + 1` or `best_ask - 1` when those improve on `ref`.

### Stop-loss (pluggable, switched via `OSMIUM_FLUSH_MODE`)

Three trigger families plus `"none"` baseline:

1. `ema_cross_consec` — trigger when `ema_fast < ema_slow AND mid < ema_fast` for `FLUSH_CONSEC` consecutive ticks AND `position > 0`. Symmetric short-side.
2. `rolling_drop` — trigger when `mid <= max(hist[-W:]) - FLUSH_DROP` AND `position > 0`.
3. `zscore_neg` — trigger when rolling z-score over `W` window `<= -FLUSH_Z` AND `position > 0`.

On trigger: sell `min(best_bid_volume, position)` at `best_bid` to flatten (cross the spread).

## Backtest results (`trader_6_backtest.ipynb`, days -2, -1, 0, taker-only fills)

Reference: `trader.py` total 3-day PnL = 251,935.00.

Top of ranked grid (23 combos):

| rank | combo | mode | params | osmium_total | pepper_total | pnl_total |
|---|---|---|---|---|---|---|
| 1 | none | none | — | 13,551.5 | 238,302.0 | **251,853.5** |
| 2 | ema_cross_consec_c5 | ema_cross_consec | CONSEC=5 | -12,732.0 | 238,302.0 | 225,570.0 |
| 3 | ema_cross_consec_c4 | ema_cross_consec | CONSEC=4 | -17,459.5 | 238,302.0 | 220,842.5 |
| 4 | zscore_neg_w60_z2.5 | zscore_neg | W=60 Z=2.5 | -20,359.0 | 238,302.0 | 217,943.0 |
| … | rolling_drop_w20_d2 | rolling_drop | W=20 DROP=2 | -30,181.0 | 238,302.0 | 208,121.0 |

Every flush family underperforms the baseline in this sim because the taker-only fill model never gives us credit for maker fills, so any cross-the-spread flush only spends edge. Pepper contribution is constant at 238,302 across all combos (logic unchanged). Full results saved to `trader_6_grid_results.csv`.

## Applied defaults

- `OSMIUM_FLUSH_MODE = "none"` (winning combo).
- Fallback params for the non-winning families are kept pointing at their per-family best (CONSEC=5, W=40/DROP=3, W=40/Z=2.5) so flipping `OSMIUM_FLUSH_MODE` to any other string gives the best-known parameters out of the box.

## Open caveats

- Taker-only fill model is strictly pessimistic vs the live engine. Expect real PnL higher than the table above.
- The 150-unit live dip that motivated trader_6 is not strongly present in days -2/-1/0, so the flush trigger's real-world value is untested by the grid. The regime-aware `fair` + inventory skew still fire in those conditions and are the primary defence; `ema_cross_consec_c5` (best trigger that did something) is one toggle away if live logs show renewed drawdown.
- PEPPER logic is byte-identical to trader.py; any PEPPER PnL difference vs trader.py in the table reflects noise in the sim (there is none — both show 238,302).
