# Delta-hedged gains in the cross-section of individual equity options

An extension of **Bakshi & Kapadia (2003, RFS)** — *Delta-Hedged Gains and the Negative Market
Volatility Risk Premium* — from S&P 500 index options to the cross-section of **single-name US
equity options, 2017–2023**.

BK test one underlying (SPX, 1988–1995). Their Section 7 flags the cross-sectional test on
individual equities as a natural extension they did not perform. This project performs it, on a
point-in-time top-150 S&P 500 universe, one ~30-day ATM contract per stock-month, delta-hedged
daily at *physical* volatility.

> This is an **extension of the BK framework**, not a replication of BK. The extension was
> subsequently carried out in the published literature — Cao & Han (2013, JFE) and Goyal & Saretto
> (2009, JFE) are the closest relatives. See `docs/methodology.md` for how this setup differs.

**Core hypothesis** (BK Proposition 1, Eq. 14–16): under a one-dimensional diffusion, or under
stochastic volatility with *unpriced* volatility risk, the expected delta-hedged gain is zero. A
significantly negative mean gain implies a negative volatility risk premium.

---

## Status

| Stage | What it does | State |
|---|---|---|
| 0 — environment + schema recon | verify every WRDS table name and column | ✅ **complete** — 12/12 tables verified, Checkpoint 0 passes |
| 1 — point-in-time universe | top-150 S&P 500 by month-end market cap | ✅ **complete** — 84 months × 150 names, Checkpoint 1 passes |
| 2a — CRSP prices + OM↔CRSP link | daily prices, split adjustment, secid mapping | ✅ **complete** — 479,878 rows, 100% link coverage, Checkpoint 2a passes |
| 2b — option chains | screens, contract selection, holding paths | ✅ **complete** — 27,658 positions (monthlies only), Checkpoint 2b passes |
| 3 — physical volatility (VOL^h, VOL^g) | | next |
| 4 — the delta-hedging engine | | not started |
| 4.5 — SPX anchor | | not started |
| 5 — statistical analysis | | not started |

Stage 1's pure-pandas logic (the point-in-time join, the ranking) is fully unit-tested and passes
without a WRDS connection: `python -m pytest tests`.

Stage 1 output, as built: 241 distinct permnos across the 84 in-sample months, 2–5 names turning
over per month. January 2017's top names include XOM (#3) and GE (#7); December 2023's include
NVDA (#4) and TSLA (#7) — the ranking is genuinely point-in-time, not today's index projected
backwards. See `docs/methodology.md` for the full checkpoint evidence and the five Stage 0 schema
findings that change later stages.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your WRDS username in it
```

**Credentials.** `WRDS_USERNAME` goes in `.env`. The **password does not** — it belongs in
`~/.pgpass` (on Windows: `%APPDATA%\postgresql\pgpass.conf`), one line:

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>
```

The easiest way to create it is to run `python -c "import wrds; wrds.Connection()"` once and
answer `y` when it offers to store the credentials for you.

## Running

Every stage is a thin, idempotent CLI wrapper over `src/vrp/`. Each one ends by printing its
checkpoint from `docs/PHASE1_PLAN.md` and exits non-zero if the checkpoint fails, so stages
can be chained without silently building on broken data.

```bash
python scripts/00_verify_schema.py     # writes docs/wrds_schema_notes.md + _resolved.json
python scripts/01_build_universe.py    # writes data/interim/universe.parquet
python scripts/02_pull_crsp.py         # writes prices/link/linked_panel/entry_calendar
python scripts/03_pull_options.py      # Pass A candidates + zero curve + dividends
python scripts/04_select_contracts.py  # screens, selection, Pass B paths
```

Every WRDS query is cached to parquet under `data/raw/`, keyed by a hash of the SQL and its bound
parameters, with a `.json` sidecar recording the query and the pull time. Re-running a stage costs
nothing and never re-hits WRDS.

## Layout

```
config/config.yaml     every date, threshold and filter cut-off; nothing is hard-coded downstream
src/vrp/               the library
  config.py            config + path resolution
  wrds_conn.py         one connection, query→parquet cache, schema helpers
  schema.py            Stage 0: table resolution, Checkpoint 0
  data/universe.py     Stage 1: point-in-time membership + ranking, Checkpoint 1
  data/crsp_prices.py  Stage 2a: price cleaning, split normalisation, Checkpoint 2a
  data/linking.py      Stage 2a: OM↔CRSP link, date-ranged join, activity tie-break
  data/option_chains.py Stage 2b: two-pass chain pull, December year-boundary union
  data/rates_divs.py   Stage 2b: zero curve interpolation, escrowed dividends
  selection.py         Stage 2b: screen cascade with drop accounting, Checkpoint 2b
scripts/0N_*.py        one thin wrapper per stage
tests/                 unit tests that run without WRDS
docs/PHASE1_PLAN.md    the build plan (read this first)
docs/methodology.md    every "why did you do it that way" answer
docs/wrds_schema_notes.md   generated by Stage 0 — verified table/column names
```

## Data

CRSP and OptionMetrics via WRDS. Nothing under `data/` is committed — WRDS licensing forbids
redistribution, and every file is reproducible from the scripts plus a WRDS login.
