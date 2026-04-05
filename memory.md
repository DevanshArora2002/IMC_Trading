# IMC Prosperity 4 — Project Memory

> **Purpose:** This file is the single source of truth for all experiments, results, decisions, and next steps.  
> Any model or collaborator reading this file should be able to instantly understand: what has been built, what numbers were observed, what failed, and where to focus next.  
> **Update this file after every new run or code change.**

---

## 1. Competition Overview

| Field | Detail |
|---|---|
| Competition | IMC Prosperity 4 |
| Currency | XIRECS |
| Submission format | Python `Trader` class with a `run(state: TradingState)` method |
| Simulation size | 1 000 iterations (test), 10 000 iterations (final) |
| `traderData` limit | 50 000 characters (persistent state across iterations) |
| Supported libraries | pandas, NumPy, statistics, math, typing, jsonpickle, all Python 3.12 stdlib |
| Position enforcement | Exchange rejects ALL orders in a product if aggregated qty would breach the limit |
| Bid method | `bid()` must return an integer — used only in Round 2 (ignore for Tutorial and Round 1) |

---

## 2. Active Round: Tutorial Round (Round 0)

### 2.1 Products in This Round

| Product | Fair Value Character | Position Limit (current bot) | Notes |
|---|---|---|---|
| `EMERALDS` | Stable peg ≈ 10 000 | 20 | Very tight spread; market-make around 10 000 |
| `TOMATOES` | Trending / noisy | 20 | SMA-20 used as fair value estimate |

### 2.2 Data Available

| File | Contents |
|---|---|
| `prices_round_0_day_-2.csv` | Order-book snapshots day -2 (~20 002 rows) |
| `prices_round_0_day_-1.csv` | Order-book snapshots day -1 (~20 002 rows) |
| `trades_round_0_day_-2.csv` | Market trades day -2 (~590 rows) |
| `trades_round_0_day_-1.csv` | Market trades day -1 (~633 rows) |
| `28169.json` | Submission run result (run ID 28169) |
| `28169_report.html` | Browser-viewable visualization of 28169.json |

---

## 3. Experiment Log

### Experiment 0 — Baseline Template (never submitted)

**File:** `challenge_info.txt` example code  
**Strategy:** Hard-coded `acceptable_price = 10` for all products; buys best ask below it, sells best bid above it.  
**Result:** Not submitted. Used only to understand the framework.  
**Takeaway:** Needed per-product fair-value logic.

---

### Experiment 1 — SMA + Fixed Peg (`trader.py`) ✅ SUBMITTED

**Submission run ID:** `28169`  
**File:** `TUTORIAL_ROUND_1/trader.py`

#### Strategy Logic

```
EMERALDS:
  - Fair value = 10 000 (hard-coded constant — market is a stable peg)
  - Sweep all asks < 10 000 (buy), sweep all bids > 10 000 (sell)
  - Position limit: ±20

TOMATOES:
  - Fair value = SMA of last 20 mid-prices (stored in traderData JSON)
  - Mid-price = (best_bid + best_ask) / 2 at each tick
  - Sweep all asks < SMA (buy), bids > SMA (sell)
  - Position limit: ±20
```

#### Key Parameters

| Parameter | Value |
|---|---|
| `EMERALDS_FAIR` | 10 000 (constant) |
| `SMA_WINDOW` | 20 ticks |
| Position limit (both products) | ±20 |

#### Results

| Metric | Value |
|---|---|
| `profit` (final score) | **76.15625 XIRECS** |
| `round` | 0 (Tutorial) |
| `status` | FINISHED |
| Final position — TOMATOES | +16 (net long) |
| Final XIRECS balance | −79 869 (positions are marked at fair; unrealised counts) |
| P&L at t=0 | 0.0 |
| P&L at t=199 600 (end) | 49.41 (graphLog) |
| Reported total profit | 76.16 (includes inventory mark-to-market) |

#### Observations

- EMERALDS contributed most of the profit; the spread around 10 000 is reliable and bot activity is high.
- TOMATOES SMA-20 lags during fast moves; the final net-long position (+16) suggests the bot was buying into a trending-up market near the end of the run but did not fully unwind.
- P&L curve (graphLog) starts flat for the first ~4 000 timestamps, then gradually rises — consistent with EMERALDS accumulating small, steady gains from market-making.
- No conversion requests were used (conversions = 0).

#### Weaknesses Identified

1. **TOMATOES fair value is lagged** — SMA-20 chases price; during fast moves we are often buying at slightly overpriced levels.
2. **No inventory management** — bot can get stuck near the position limit with no mechanism to flatten.
3. **No quoting / passive orders** — all orders are aggressive (sweep); we leave spread revenue on the table.
4. **EMERALDS** — could post passive quotes at 9 993 / 10 007 to earn the spread, rather than only sweeping.

---

### Experiment 2 — Advanced ProductTrader Framework (`new_trader.py`) 🚧 IN PROGRESS / NOT SUBMITTED

**File:** `TUTORIAL_ROUND_1/new_trader.py`  
**Status:** Framework written; individual product strategies NOT yet completed. File currently only contains the base `ProductTrader` class and global constants.

#### What is Already Built

- `ProductTrader` base class with:
  - `bid()` / `ask()` methods that auto-cap to position limits
  - `get_order_depth()`, `get_best_bid_ask()`, `get_walls()`, `get_total_market_buy_sell_volume()`
  - Persistent `traderData` load/save
  - `check_for_informed()` — detects known informed trader `"Olivia"` from `market_trades`, tracks direction (LONG / NEUTRAL / SHORT)
  - Structured logging via `self.log()`

#### Products Referenced in Constants (future rounds)

| Symbol | Type | Position Limit |
|---|---|---|
| `RAINFOREST_RESIN` | Static (peg) | 50 |
| `KELP` | Dynamic | 50 |
| `SQUID_INK` | Dynamic | 50 |
| `PICNIC_BASKET1` | ETF | 60 |
| `PICNIC_BASKET2` | ETF | 100 |
| `CROISSANTS` | ETF constituent | 250 |
| `JAMS` | ETF constituent | 350 |
| `DJEMBES` | ETF constituent | 60 |
| `VOLCANIC_ROCK` | Options underlying | 400 |
| `VOLCANIC_ROCK_VOUCHER_9500–10500` | Options (5 strikes) | 200 each |
| `MAGNIFICENT_MACARONS` | Commodity | 75 |

#### Strategy Stubs (not yet coded)

- **ETF arbitrage** — compare basket price to NAV of constituents; trade the spread when premium > threshold.
- **Options (Black-Scholes IV)** — compute implied vol per strike, trade mean-reversion in IV; scalp when IV deviates > threshold.
- **Commodity (`MAGNIFICENT_MACARONS`)** — conversion-based (import/export tariff arbitrage using `ConversionObservation`).
- **Informed trader signal** — follow Olivia's direction for directional trades.

---

## 4. Key Decisions & Rationale

| Decision | Rationale |
|---|---|
| EMERALDS fixed at 10 000 | Order-book data confirms mid-price is extremely stable; no drift observed over 20 001 rows |
| TOMATOES uses SMA not EWM | Simpler; EWM is better but requires tuning alpha — deferred |
| No passive quoting in Exp 1 | Risk of queue position and adverse selection — conservative choice for first submission |
| `traderData` stored as JSON | Native Python, no import needed; safe for the 50k char limit |
| `bid()` returns 15 | Placeholder for Round 2; harmless in all other rounds |

---

## 5. Tooling Built

| Tool | File | Purpose |
|---|---|---|
| JSON visualizer | `visualize_submission_json.py` | Generates `*_report.html` from any run JSON; uses only stdlib |
| Run report | `28169_report.html` | Visual summary of run 28169 (open in browser) |
| Analysis notebook | `tutorial_round_1.ipynb` | EDA on price/trade CSVs |
| Sample notebook | `sample.ipynb` | Platform-provided reference notebook |

---

## 6. What to Do Next — Prioritised

### Priority 1 — Improve EMERALDS (highest leverage, quick win)

- [ ] Add **passive quoting**: post a bid at 9 993 and an ask at 10 007 each tick so bots fill us and we earn the spread.
- [ ] Measure: compare P&L with Exp 1; target > 100 XIRECS.

### Priority 2 — Fix TOMATOES fair-value signal

- [ ] Try **EWM (alpha ≈ 0.1)** instead of SMA-20; reduces lag.
- [ ] Add **inventory skew**: if position > +15, raise fair value slightly to encourage selling (flatten).
- [ ] Backtest on `prices_round_0_day_-1.csv` and `day_-2.csv` before submitting.

### Priority 3 — Inventory / risk management

- [ ] Hard stop: if position hits ±19 (1 away from limit), aggressively unwind at market.
- [ ] End-of-day flatten: as timestamp approaches 200 000, widen the threshold to reduce open inventory.

### Priority 4 — Complete `new_trader.py` for future rounds

- [ ] Write `EmeraldTrader` and `TomatoTrader` classes inside `new_trader.py` using the base class.
- [ ] Wire a top-level `Trader.run()` that delegates to each product trader.
- [ ] Test that the file compiles and that `traderData` round-trips correctly.

### Priority 5 — ETF Arbitrage (later rounds)

- [ ] Understand basket constituents: PICNIC_BASKET1 = 6 CROISSANTS + 3 JAMS + 1 DJEMBES.
- [ ] Compute NAV from constituent mid-prices; trade when premium > threshold (starts at ±80 for basket1, ±50 for basket2).
- [ ] Hedge: trade constituents opposite to the basket position.

### Priority 6 — Options (later rounds)

- [ ] Implement Black-Scholes pricing for VOLCANIC_ROCK vouchers (5 strikes: 9500–10500).
- [ ] Use remaining days (`DAY = 5` constant in new_trader.py) to compute theta correctly.
- [ ] IV mean-reversion: open when IV deviates > 0.5 vol, close at 0.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| `traderData` | String persisted between ticks; max 50 000 chars |
| Mid-price | (best_bid + best_ask) / 2 at a given timestamp |
| SMA-N | Simple moving average over last N observations |
| EWM | Exponentially weighted moving average |
| Position limit | Absolute max long or short; violations cancel all orders |
| XIRECS | Competition currency |
| `graphLog` | `timestamp;value` CSV inside run JSON — cumulative P&L curve |
| `activitiesLog` | Full order-book snapshot CSV inside run JSON |
| Olivia | Known informed trader in `market_trades`; following her is a free signal |
| NAV | Net Asset Value of ETF basket = sum of (constituent qty × constituent mid) |

---

*Last updated: 2026-04-05 | Experiment count: 2 | Best result: 76.15625 XIRECS (run 28169)*
