# Round 3 — Trading Strategy (v2)

## Diagnosis: What Went Wrong with trader.py

The **entire loss (-3,966 across VEV options)** came from the options strategy. HYDROGEL was profitable (+271), VELVET was roughly breakeven (-24).

**Root cause:** The vol-smile coefficients `[0.273, 0.010, 0.149]` in `trader.py` were borrowed directly from the Round 5 winner's code, which was calibrated for `VOLCANIC_ROCK` options — a completely different underlying with different dynamics.

The actual VEV vol smile from Round 3 data:

| Metric | Winner's value (VOLCANIC_ROCK) | Actual VEV value |
|--------|-------------------------------|------------------|
| Curvature `a` | 0.274 | 0.029 (10x smaller) |
| Base IV `c` | 0.149 | 0.239 (60% higher) |
| R^2 of parabola fit | high | 0.047 (essentially flat) |

This caused systematically wrong theoretical prices, especially for deep ITM options (VEV_4000 got IV=1.61, wildly incorrect), leading to large directional bets on incorrect signals.

**Additional problems:**
- Level-based mean reversion on VELVET was fragile and regime-dependent
- Deep ITM strikes (4000, 4500) had 20+ tick spreads — untradeable for scalping
- MR strategy on deep ITM used the same broken IV coefficients

## New Strategy: trader_2.py

### Architecture

```
Trader.run()
├── HydrogelTrader     (Static MM around wall_mid — unchanged, proven)
├── VelvetTrader       (EMA-based MM with inventory skew — new)
└── OptionTrader       (Online vol smile + IV scalping — rewritten)
    ├── Online Vol Smile Fit    (parabola fitted fresh each tick)
    ├── IV Scalping             (ATM strikes only: 5000-5400)
    ├── Delta Hedging           (via VELVETFRUIT_EXTRACT)
    └── No deep ITM trading     (4000, 4500, 5500, 6000, 6500 excluded)
```

### 1. HYDROGEL_PACK — Static Market Making (unchanged)

**Rationale:** HYDROGEL oscillates tightly around 10,000. This strategy made +271 in prior logs. No changes needed.

**Approach:**
- **Taking:** Buy when asks <= wall_mid - 1; sell when bids >= wall_mid + 1. Inventory-neutral fills at wall_mid.
- **Making:** Post bids at bid_wall + 1 (with overbidding) and asks at ask_wall - 1 (with underbidding).
- Optional hybrid skew when `EXTREME_DIST > 0` (disabled by default).

### 2. VELVETFRUIT_EXTRACT — EMA-Based Market Making (new)

**Rationale:** The old level-based MR (`buy at 5230, sell at 5265`) was fragile and regime-dependent. Replacing with a proper market-making strategy around a dynamically-computed fair value.

**Approach:**
- Compute a running EMA (window = 20) of mid-price as fair value
- Apply inventory skew: `skewed_fair = fair - 0.02 * position`
- **Take:** Sweep mispriced liquidity when price deviates > 2 from skewed fair
- **Make:** Post bids at `skewed_fair - 1` and asks at `skewed_fair + 1`, adjusted to not cross the book
- Respects position limits (200)

### 3. VEV Options — Complete Rewrite

**Key data insights:**
- Base IV is ~0.24, standard deviation ~0.004
- IV shows strong mean reversion (autocorrelation = -0.44)
- Vol smile is essentially flat for ATM strikes
- Deep ITM options (4000, 4500) have 20+ tick spreads — untradeable
- ATM options (5200, 5300) have 2-3 tick spreads — best for scalping

**a) Online vol smile calibration**

Each tick, the OptionTrader computes implied volatility for all options with observable prices using Newton-Raphson, then fits a parabola `iv(m) = a*m^2 + b*m + c` via least-squares. Falls back to EMA of base IV when fewer than 3 data points are available.

**b) ATM-only trading**

Only trades VEV_5000 through VEV_5400 (strikes within ~200 of underlying). Never trades deep ITM/OTM (4000, 4500, 5500, 6000, 6500) due to illiquidity and unreliable pricing.

**c) Conservative IV scalping**

Same signal framework as the winner's code but with:
- Properly calibrated vol smile (online fitted, not hardcoded)
- Higher entry threshold (THR_OPEN = 1.0 instead of 0.5)
- Position sizing limited to 50 per strike instead of max 300
- Flatten positions when switch_mean drops below threshold

**d) Delta hedging via VELVET**

After computing aggregate delta across all option positions, hedges by trading VELVET in the opposite direction. Target VELVET position = -round(aggregate_delta).

**e) Conflict resolution**

When options are actively generating delta-hedge orders for VELVET, the VelvetTrader is disabled. VelvetTrader only runs standalone when no delta-hedge orders are needed.

## Parameters

| Parameter | Old (trader.py) | New (trader_2.py) | Rationale |
|-----------|----------------|-------------------|-----------|
| VOL_SMILE_COEFFS | `[0.274, 0.010, 0.149]` | Online fitted | Stale coefficients were the #1 bug |
| Base IV | 0.149 | ~0.239 (fallback) | Measured from R3 data |
| IV_SCALPING strikes | 5000-5300 | 5000-5400 | Focus on liquid ATM only |
| MR strikes | 4000, 4500 | **removed** | Illiquid, huge spreads |
| THR_OPEN | 0.5 | 1.0 | More conservative entry |
| Max position per strike | 300 | 50 | Reduce risk per strike |
| VELVET strategy | Level MR | EMA-based MM | More robust, adapts to regime |
| VELVET_EMA_WINDOW | N/A | 10 | Fast-adapting fair value (EMA=10 outperforms EMA=20 in take-only backtest) |
| VELVET_TAKE_EDGE | N/A | 3 | Conservative: edge=3 profitable on take-only, edge=2 was too tight |
| VELVET_INVENTORY_SKEW | N/A | 0.02 | Skew quotes toward flat |

## State Management

A single `traderData` JSON blob persists across ticks, storing:
- `velvet_ema`: Running EMA of VELVET mid-price
- `ema_base_iv`: EMA of online-fitted base IV (fallback)
- Per-option `{name}_theo_diff`: EMA of theo vs market diff
- Per-option `{name}_avg_devs`: EMA of absolute deviation (switch-mean)

## Files

| File | Description |
|------|-------------|
| `trader_2.py` | **New unified trader** with all 3 strategies (this README documents it) |
| `trader.py` | Old unified trader (v1, kept for reference) |
| `datamodel.py` | Data model (shared) |
| `round3_analysis.ipynb` | Comprehensive analysis notebook |
| `trader_hydrogel_mm.py` | Original standalone hydrogel MM trader |
| `trader_velvet_compound.py` | Original standalone velvet compound trader |
| `Round_3_write_up.md` | Competition write-up with sample option trader code |

## Risk Management

- Position limits strictly enforced via `max_buy`/`max_sell` tracking
- Option positions capped at 50 per strike (vs 300 limit) to reduce tail risk
- Online vol smile avoids systematic mispricing from stale coefficients
- Deep ITM/OTM strikes excluded entirely — no trading in illiquid markets
- Delta hedging keeps aggregate option exposure near neutral
- Inventory skew on VELVET MM biases quotes toward reducing position
