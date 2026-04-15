# Round 1 — strategy memory (`trader.py`)

Concise reference for what this round’s bot does. **All behaviour below is taken from** `Round_1/ROUND1/trader.py` **only.**

## Gist

- Two products: **`INTARIAN_PEPPER_ROOT`** and **`ASH_COATED_OSMIUM`**, each with **`POSITION_LIMIT = 80`**.
- **`run`** builds orders per product if it appears in `state.order_depths`, serialises state to **`traderData`** as JSON, returns `(result, 0, traderData)`.

## State (`traderData`)

- **`osmium_ema`**: exponential moving average of mid (updated each Osmium tick).
- **`osmium_history`**: last **`OSMIUM_HISTORY_WINDOW`** (100) mid prices (used only to bound history length; EMA uses the scalar `osmium_ema`).

## INTARIAN_PEPPER_ROOT

- **Goal while under limit:** buy only (no sells).
- **`max_buy = 80 - position`**. If `max_buy <= 0`, return no orders.
- **Taking:** walk asks in price order; for each level, buy `min(ask_volume, remaining max_buy)` until capacity is filled.
- **Making:** if capacity remains, place one passive **buy** at:
  - `best_ask - 1` if both bid and ask exist;
  - else `best_bid + 1` if only bids;
  - else `10_000`.

## ASH_COATED_OSMIUM

- **Mid:** `(best_bid + best_ask) / 2` if both sides exist; else lone best bid or best ask; if no book, skip (no orders).
- **Fair value:**  
  `fair_value = OSMIUM_ANCHOR_WEIGHT * 10_000 + OSMIUM_EMA_WEIGHT * ema`  
  with **`OSMIUM_ANCHOR_WEIGHT = OSMIUM_EMA_WEIGHT = 0.5`**, **`OSMIUM_EMA_ALPHA = 0.05`**, and  
  `ema ← alpha * mid + (1 - alpha) * ema` (seed `ema` from state or first `mid`).
- **`max_buy = 80 - position`**, **`max_sell = 80 + position`**, with local **`bought` / `sold`** tallies for the tick.
- **Taking (asks):** for asks with `ask_price <= fair_value - 1`, buy up to remaining buy capacity. Near fair: if `ask_price > fair_value - 1`, only buy to cover a short (`position < 0`) at `ask_price <= fair_value`, capped by remaining capacity and `abs(position) - bought`, then stop the ask loop.
- **Taking (bids):** for bids with `bid_price >= fair_value + 1`, sell up to remaining sell capacity. Near fair: if `bid_price < fair_value + 1`, only sell to reduce a long (`position > 0`) at `bid_price >= fair_value`, capped similarly, then stop the bid loop.
- **Making:**  
  - `bid_price = floor(fair_value) - 1`, `ask_price = ceil(fair_value) + 1`; if equal, bump `ask_price` by 1.  
  - If `best_bid + 1 < fair_value`, raise `bid_price` to that overbid.  
  - If `best_ask - 1 > fair_value`, lower `ask_price` to that under-ask.  
  - Post **remaining** buy and sell sizes (full leftover capacity after taking), not a single side only.

## Position / risk (per tick)

- Pepper and Osmium both respect **`POSITION_LIMIT`** via **`max_buy` / `max_sell`** and the taking loops + final maker sizes; Osmium also tracks **`bought`** / **`sold`** within the tick when building aggressive orders.

## Recorded PnL (submission export, not defined in code)

- One finished Round 1 submission log in this folder reports **`profit`: ~8414.59** (`submission_1/166849.json`). Interpretation of that number is platform-defined; the bot logic is entirely as above.
