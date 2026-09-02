# Phase 1 Plan — Delta-Hedged Gains in the Cross-Section of Individual Equity Options

**Status:** planning only. No implementation code written yet.
**Primary source:** `docs/bakshi_kapadia_2003_rfs.pdf` — Bakshi & Kapadia (2003), *Delta-Hedged Gains
and the Negative Market Volatility Risk Premium*, RFS 16(2), 527–566. Read directly; all section
and equation citations below refer to that PDF.

---

## 0. Framing (get this right before you write a line of code)

### What the paper actually did
BK (2003) test **S&P 500 index options only**, 1988:01–1995:12, 36,237 calls and 35,030 puts on a
single underlying. Every empirical table in the paper is one underlying.

### What we are doing
Their Section 7 ("Final Remarks", p. 561) states verbatim:

> "There are two natural extensions to this article. First, given that volatilities of individual
> stocks and the market index comove highly, one could examine whether the volatility risk premium
> is negative in individual equity options. The cross-sectional restrictions on delta-hedged gains
> and the volatility risk premium can be tested in the cross section of individual equity options."

So the framing in the brief is correct and citable: **this is an extension the authors flagged and
did not perform.** Say "extension of the BK framework to the cross-section of individual equities,"
never "replication of BK."

### Honesty check you need before an interview
The extension BK flagged **was subsequently carried out in the published literature.** At minimum:

- **Cao & Han (2013), JFE**, "Cross section of option returns and idiosyncratic stock volatility" —
  uses BK-style delta-hedged gains on individual equity options.
- **Goyal & Saretto (2009), JFE**, "Cross-section of option returns and volatility" — related
  IV−RV signal on single-name options.

Neither invalidates the project — a careful, well-validated replication-class build is exactly the
right resume artifact, and being the person who *knows the follow-up literature* is strictly better
than being the person who claims novelty and gets corrected. **Action item: read both abstracts and
be able to say in one sentence how your setup differs (sample period 2017–2023, top-150 S&P 500 by
market cap, ATM-only single contract per stock-month).** Verify my characterizations of those two
papers yourself; I am citing them from general knowledge, not from a PDF in this repo.

### Target magnitudes — for scope calibration and bug detection only

What the paper actually reports (Table 1, Panel A; Sec. 3; Sec. 5). These are **not** results to
reproduce — different underlying, different decade — they are the range that tells you whether your
engine is broken.

| Measure | Paper value | Source |
|---|---|---|
| **Mean π/S, ATM** | **−0.10% to −0.11%** | Table 1 — **the stable anchor** |
| Mean π/S, all moneyness & maturities | −0.05% | Table 1 |
| Mean π/C, ATM buckets | −3.88% (y ∈ [0, 2.5%]), −7.59% (y ∈ [−2.5%, 0%]) | Table 1 |
| Mean π/C, full sample across all buckets | −12.18% | Sec. 3, p. 541 |
| Prose summary, ATM | "amounts to 8% of the option value" | Sec. 1, p. 529 |
| 30-day constant-maturity ATM series | mean π = −$0.47, t = −2.34 naive / −5.29 standardized | Sec. 3, pp. 543–544 |
| Fraction of ATM observations negative | 68% | Table 1, last column |

**Anchor on π/S, quote π/C alongside it.** π/C is unstable by construction — the option price sits
in the denominator and collapses for OTM options, which is why the deep-OTM buckets read −68% and
−97% and why the full-sample π/C (−12.18%) is three times the ATM figure (−3.88%). π/S has no such
problem, and Lemma 1 / Eq. (19) *prove* that π scales with S, making π/S the theoretically motivated
normalization rather than a convenience (M14). Since this project is ATM-only, π/C is well-behaved
enough to headline, but it should never travel alone.

**Not targets:** the figures −3.63%, −11.18% and −19.60% in Sec. 5 (pp. 553–554) are three
hand-picked dates (19 Aug 1992 at 8.05% vol, 19 Jul 1989 at 12.04%, 20 Nov 1991 at 15.86%)
illustrating how the *regression-implied* premium scales with the volatility level. They are neither
averages nor findings. Do not try to reproduce them.

### Core testable hypothesis (unchanged)
Proposition 1, Eq. (14)–(16): under a one-dimensional Markov diffusion, or under stochastic
volatility with **unpriced** volatility risk, E[π] = 0 up to O(1/N) discretization error. If
volatility risk *is* priced, E[π] carries the sign of λ (the volatility risk premium) times the
option vega, which is positive. **A significantly negative mean π ⟹ negative volatility risk
premium.**

---

## 1. Methodological specifics pulled from the paper (beyond the brief)

These are details the brief did not specify and that I took from the PDF. Each is cited so you can
defend it.

| # | Detail | Source |
|---|---|---|
| M1 | **Terminal value is intrinsic value at expiry, not a market mark.** π = max(S_T − K, 0) − C_t − Σ Δ_tn (S_tn+1 − S_tn) − Σ r_n (C_t − Δ_tn S_tn)(τ/N). The position is held to expiration. | Sec. 3, unnumbered eq. p. 540; Table 1 note |
| M2 | **Put version hedges symmetrically:** π = max(K − S_T, 0) − P_t − Σ Δ̂_tn (S_tn+1 − S_tn) − Σ r_n (P_t − Δ̂_tn S_tn)(τ/N), with Δ̂ the BS *put* delta (negative), so the "hedge" is a long stock position. | Table 2 note, p. 544 |
| M3 | **Rebalance frequency τ/N = 1 day**, at the close. | Table 1 note |
| M4 | **Interest rate r_n is updated daily** inside the financing term. | Sec. 3, p. 540 |
| M5 | **Hedge delta = Black–Scholes N[d1] evaluated at GARCH volatility**, Eq. (30). Confirms the brief: physical vol, not implied. | Sec. 3, Eq. (30) |
| M6 | **GARCH parameters re-estimated annually on one year of daily returns.** | Table 7 note, p. 553 |
| M7 | **VOL^g is a *backward* average of fitted conditional variances over the trailing τ days**, Eq. (28): VOL^g_t = sqrt((252/τ) Σ_{n=t−τ}^{t} σ̂²_n). It is *not* an h-step-ahead GARCH forecast. Same for the historical estimator VOL^h, Eq. (29). Easy to get wrong. | Sec. 2, Eqs. (28)–(29) |
| M8 | **Both vol estimators are used.** GARCH drives the hedge ratio; the rolling sample s.d. VOL^h has consistently *higher* R² in the time-series regressions. | Sec. 2; Sec. 5, p. 553 |
| M9 | **Dividends: PV of discrete dividends subtracted from the contemporaneous stock price** ("escrowed dividend"), not a continuous yield. | Sec. 2, p. 539 |
| M10 | **r inferred from put-call parity** on strike-and-maturity-matched pairs, midpoint of implied borrowing and lending rates. | Sec. 2, p. 539 |
| M11 | Screens: drop options violating arbitrage bounds (call outside (S e^{−zτ} − e^{−rτ}K, S e^{−zτ})); drop implied vol > 100% or < 1%; drop maturity < 14 days; drop maturity > 60 days; restrict to ±10% moneyness. | Sec. 2, p. 538 |
| M12 | **Moneyness definition** y ≡ S e^{(r−z)τ} / K. Call is OTM when y < 1. | Sec. 2, p. 540 |
| M13 | **Three normalizations reported:** dollar π, π/S (in %), π/C (in %). Plus the fraction of observations with π < 0. | Table 1 |
| M14 | **π scales with S** — this is a theoretical result (Lemma 1, Eq. (19)), which is *why* π/S is the theoretically-motivated normalization, not just a convenience. | Sec. 1.3, Lemma 1, Eq. (19) |
| M15 | **BK themselves flag that naive cross-sectional standard errors are too small** and construct a second t-stat by standardizing each π by its Bertsimas–Kogan–Lo theoretical s.d.; this moves t from −2.34 to −5.29 on 30-day ATM calls. | Sec. 3, pp. 543–544 |
| M16 | **Vol-regime test** bins ATM gains into seven volatility buckets by VOL^h and reports mean *and median* per bucket; losses deepen monotonically with volatility. | Sec. 3, Table 3 |
| M17 | **Time-series test:** GAINS_t = Ω0 + Ω1 VOL_t + Ω2 GAINS_{t−1} + ε_t, OLS with **Newey–West t-stats, lag 12**, lagged dependent variable included to kill residual autocorrelation. Null is Ω1 = 0. | Sec. 5, Eq. (33) |
| M18 | **Economic-significance benchmark: compare the mean loss to the mean bid-ask spread.** BK: $0.43 loss vs. $0.375 mean spread. | Sec. 3, p. 541 |
| M19 | Mishedging robustness: add contemporaneous underlying return R_{t,t+τ} as a regressor; Ω3 > 0 ⟹ BS underhedges and π is biased *upward* (i.e. against finding a loss). | Sec. 5.1, pp. 554–555, and fn. 3 |

**M18 deserves emphasis for our extension.** Single-name equity options have far wider relative
spreads than SPX. If the mean loss does not clearly exceed half the spread, the result is not
economically meaningful. This will be the first question a trading interviewer asks. Build the
spread comparison into the main results table, not an appendix.

---

## 2. Repo structure

```
hedging_project/
├── README.md                      # what this is, how to run, headline result
├── requirements.txt
├── .env                           # WRDS_USERNAME only; never the password (use ~/.pgpass)
├── .gitignore                     # data/, .env, *.parquet, .ipynb_checkpoints
├── config/
│   └── config.yaml                # dates, universe size, all filter thresholds, paths
├── docs/
│   ├── bakshi_kapadia_2003_rfs.pdf
│   ├── PHASE1_PLAN.md             # this file
│   ├── wrds_schema_notes.md       # VERIFIED table/column names + date verified  ← Stage 0 output
│   └── methodology.md             # the interview-defense doc; write as you go, not at the end
├── data/
│   ├── raw/                       # straight from WRDS, parquet, never mutated
│   ├── interim/                   # linked/cleaned panels
│   └── processed/                 # selected contracts, hedge results
├── src/vrp/
│   ├── __init__.py
│   ├── config.py                  # loads config.yaml, resolves paths
│   ├── wrds_conn.py               # single connection factory + query→parquet cache helper
│   ├── data/
│   │   ├── universe.py            # Stage 1: point-in-time S&P 500 → top-150 by mktcap
│   │   ├── linking.py             # permno ↔ secid via WRDS linking suite
│   │   ├── crsp_prices.py         # Stage 2a
│   │   ├── option_chains.py       # Stage 2b (two-pass: selection, then paths)
│   │   └── rates_divs.py          # zero curve + discrete dividend schedule
│   ├── vol.py                     # VOL^h (Eq. 29) and VOL^g (Eqs. 25–28)
│   ├── pricing.py                 # BS price/delta/vega; optional CRR American
│   ├── selection.py               # Stage 3: contract selection rules
│   ├── hedging.py                 # Stage 4: THE REUSABLE ENGINE
│   └── analysis/
│       ├── stats.py               # means, t-stats (3 flavors), frac negative
│       └── regimes.py             # vol-regime splits, Eq. (33) regressions
├── scripts/                       # thin CLI wrappers, one per stage, idempotent
│   ├── 00_verify_schema.py
│   ├── 01_build_universe.py
│   ├── 02_pull_crsp.py
│   ├── 03_pull_options.py
│   ├── 04_select_contracts.py
│   ├── 05_run_hedge.py
│   └── 06_analysis.py
├── notebooks/
│   ├── 00_wrds_schema_exploration.ipynb   # scratch; findings graduate to wrds_schema_notes.md
│   └── 90_results.ipynb
├── tests/
│   ├── test_pricing.py            # BS vs. known values; put-call parity; delta bounds
│   ├── test_vol.py                # VOL^h/VOL^g on synthetic series with known σ
│   ├── test_hedging.py            # ★ synthetic zero-VRP GBM ⟹ mean π ≈ 0
│   └── test_selection.py
└── output/
    ├── tables/                    # csv + latex
    └── figures/
```

### The one design decision that matters for Phase 2

`src/vrp/hedging.py` exposes a **single position-level function** and everything else is a loop over
it. Sketch of the contract (not implementation):

```python
@dataclass(frozen=True)
class OptionPosition:
    secid: int; permno: int; optionid: int
    cp_flag: str            # 'C' | 'P'
    strike: float
    entry_date: date; expiry: date
    entry_price: float      # bid-ask midpoint at entry
    quantity: float = 1.0   # signed: +1 long, -1 short  ← Phase 2 needs shorts

@dataclass
class HedgeResult:
    pnl: float                  # scalar π
    daily: pd.DataFrame         # date, S, sigma_hat, r, delta, d_hedge_pnl,
                                # d_financing, cum_pnl   ← Phase 2 needs the PATH
    diagnostics: dict           # n_rebalances, missing_days, terminal_moneyness, ...

def delta_hedged_gain(
    position: OptionPosition,
    underlying: pd.DataFrame,      # date, S_adj, pv_divs
    vol_path: pd.Series,           # date → physical sigma estimate
    rate_path: pd.Series,          # date → continuously-compounded r
    *,
    delta_fn: Callable = bs_delta,     # swappable: BS-European | CRR-American | OM's own delta
    terminal: str = "intrinsic",       # "intrinsic" (paper) | "mid" (mark-to-market)
    rebalance: str = "daily",
) -> HedgeResult: ...
```

Three properties earn their keep later at near-zero cost now:
1. **Signed `quantity`** — Phase 2 will short options; don't bake in "long."
2. **Return the daily path**, not just the scalar — Phase 2's portfolio backtest needs
   mark-to-market by date, and the path is free once you've computed it.
3. **`delta_fn` injectable** — lets you run the implied-vol-delta placebo (Stage 4 validation) and
   swap in an American delta without touching the engine.

Do not build a config-driven strategy framework. That is over-engineering. Three parameters is the
whole extensibility budget.

---

## 3. Staged build order with validation checkpoints

Rule for every stage: **1 stock (suggest AAPL, permno 14593), 1 year (suggest 2019 — no regime
drama, clean data) must pass before the stage scales to 150 × 7.** Concretely: each `scripts/0N_*.py`
takes `--permnos` and `--years` and defaults to the subsample.

---

### Stage 0 — Environment + schema reconnaissance
*Not in the brief's build order; add it. Everything downstream assumes table names you have not
verified.*

Install: `wrds`, `arch` (GARCH), `pyarrow` (parquet), `matplotlib`.
Currently present: numpy 2.3.4, pandas 2.3.3, scipy 1.16.3, statsmodels 0.14.5, python-dotenv, tqdm.
Missing: **wrds, arch, pyarrow, matplotlib.**

Work in `notebooks/00_wrds_schema_exploration.ipynb` and the WRDS web query builder side by side.
For each table below, record in `docs/wrds_schema_notes.md`: exact schema.table, column list,
row count, date range, and the date you verified it.

| Need | Best guess | Confidence |
|---|---|---|
| S&P 500 point-in-time membership | `crsp.dsp500list` (permno, start, ending) | high |
| Monthly market cap | `crsp.msf` (permno, date, prc, shrout) | high |
| Daily stock prices | `crsp.dsf` (permno, date, prc, ret, cfacpr, cfacshr, shrout, bidlo, askhi, vol) | high |
| Delisting returns | `crsp.dsedelist` or `crsp.dse` (permno, dlstdt, dlret, dlstcd) | medium |
| Daily option prices | `optionm.opprcd2017` … `opprcd2023` | high |
| OM security master | `optionm.securd` (secid, cusip, ticker, index_flag, issue_type); `optionm.secnmd` is the *name-history* file | medium |
| Zero-coupon curve | `optionm.zerocd` (date, days, rate) | high |
| Projected dividends | `optionm.distrd` (secid, ex_date, amount, distr_type) | medium |
| OM historical volatility | brief says `optionm.hvol`; likely `optionm.hvold` and possibly year-split | **low** |
| **OM ↔ CRSP link** | `wrdsapps.opcrsphist` (secid, permno, score, sdate, edate) | **medium — verify first** |

**Checkpoint 0 — do not proceed until all are true:**
- `wrds.Connection()` succeeds and `db.list_tables(library='optionm')` returns.
- Every table above resolves to a real name, written down with its verified column list.
- `wrdsapps.opcrsphist` (or its current equivalent) is located and its `score` semantics are
  understood — specifically, which score values are trustworthy links.
- One sanity query returns rows: AAPL options on 2019-01-02.
- **Resolve the OM dividend question** (see Open Question Q5): confirm whether a per-security
  continuous dividend yield field exists for *equities*, or whether you must build it from
  `optionm.distrd`. My expectation is the latter — the continuous-yield file in IvyDB is for
  *indices* (`optionm.idxdvd`), not single names.

---

### Stage 1 — Point-in-time universe

Two-step, to avoid pulling 8 years of daily data for 1,000+ names:

```sql
-- 1a. Membership spans, with a 12-month burn-in before 2017 for vol estimation
SELECT permno, start AS from_date, ending AS thru_date
FROM   crsp.dsp500list
WHERE  ending >= DATE '2016-01-01'
  AND  start  <= DATE '2023-12-31';
```

```sql
-- 1b. Monthly market cap for members only (msf is small; dsf would be 100x larger)
SELECT m.permno,
       m.date,
       ABS(m.prc) * m.shrout AS mktcap_k          -- shrout in thousands ⟹ mktcap in $000s
FROM   crsp.msf m
JOIN   crsp.dsp500list s
       ON  s.permno = m.permno
       AND m.date BETWEEN s.start AND s.ending    -- ← the point-in-time join; this is the
                                                  --   entire survivorship-bias fix
WHERE  m.date BETWEEN DATE '2016-12-01' AND DATE '2023-12-31'
  AND  m.prc IS NOT NULL;
```

Then in pandas: for each month-end, rank members by `mktcap_k` descending, take top 150. That gives
a `universe` frame of (year_month, permno, rank, mktcap_k). A name enters and exits as its rank and
membership change — do not carry it forward.

**Checkpoint 1:**
- Member count per month is 495–510 in every month (S&P 500 runs slightly over 500 due to
  multi-class names). A month showing 300 or 700 means the join is wrong.
- **Survivorship spot-check:** TSLA (permno 93436) must be **absent** before 2020-12-21 and present
  after. This is the single best one-line proof your point-in-time logic works. Add a second:
  a name that *exited* mid-sample and must vanish (e.g. a 2018–2020 deletion you pick from the
  membership table itself).
- Top-150 on 2020-03-02 contains AAPL, MSFT, AMZN, GOOGL. On 2017-01-03 it contains XOM and GE
  (both large then, both much smaller by 2023) — a good "the ranking is genuinely time-varying"
  check.
- No permno appears twice within a month; ranks are 1…150 with no gaps.
- Total distinct permnos across the sample should be meaningfully more than 150 (expect ~250–320).
  If it's ~150, your universe is static and you have reintroduced survivorship bias.

---

### Stage 2a — CRSP daily prices + the link

```sql
-- Daily prices for the union of all permnos that were ever top-150.
-- NOTE the end date: December-2023 entries are held into January 2024.
SELECT permno, date, prc, ret, cfacpr, cfacshr, shrout, bidlo, askhi, vol, openprc
FROM   crsp.dsf
WHERE  permno IN %(permnos)s
  AND  date BETWEEN DATE '2016-01-01' AND DATE '2024-02-29';
```

Cleaning, in this order:
1. `prc < 0` means CRSP stored the bid-ask *average* (no closing trade). Take `ABS(prc)` and set a
   `price_is_quote_avg` flag; do not silently drop.
2. Adjusted price `S_adj = ABS(prc) / cfacpr`. Adjusted shares `shrout * cfacshr`.
3. **Normalize to the entry-date factor per position**, so entry-date S and K are as-traded:
   `S_used(t) = ABS(prc_t)/cfacpr_t * cfacpr_entry`. See Open Question Q11 — this is the classic
   split bug and it is worth writing down explicitly in `methodology.md`.

```sql
-- The OM ↔ CRSP link. VERIFY the schema/table name in Stage 0 before running this.
SELECT secid, permno, score, sdate, edate
FROM   wrdsapps.opcrsphist
WHERE  permno IN %(permnos)s
  AND  edate >= DATE '2017-01-01'
  AND  sdate <= DATE '2024-02-29';
```

Apply the link as a **date-ranged join**, exactly like the membership join:
`ON l.permno = u.permno AND u.date BETWEEN l.sdate AND l.edate`. Then assert one secid per
(permno, date).

**Checkpoint 2a:**
- **Split test:** AAPL 4-for-1 on 2020-08-31 and NVDA 4-for-1 on 2021-07-20 show **no** ~75% jump in
  the adjusted series. Plot both; look, don't just assert.
- **Return reconciliation:** cumulative product of `(1 + ret)` vs. cumulative adjusted *price*
  return over 2019 for AAPL — they should differ only by the dividend contribution (~1%/yr for
  AAPL), and the gap should be a smooth staircase, not noise. If they diverge wildly, your
  adjustment is wrong.
- Link coverage: what fraction of (permno, month) cells in the universe get a secid? Expect >98%
  for large-cap S&P names. Investigate any name that fails — it's usually a share-class or ticker
  issue, and a systematic gap here silently deletes part of your cross-section.
- Zero permno-months with two concurrent secids (or a documented tie-break rule if there are).
- No duplicate (permno, date) rows.

---

### Stage 2b — Option chains (two passes)

**Key realization that shrinks this stage by ~95%:** the paper's π formula (M1) needs the option
price **only at entry**. The terminal value is *intrinsic* (computed from S_T and K), and the deltas
are *ours*, recomputed from physical vol. So the daily option quote panel is **not required for the
P&L**. Pull it anyway, but only for the ~25,000 selected `optionid`s — for diagnostics, for the
implied-vol placebo, and because Phase 2 will want mark-to-market. Do not pull daily chains for the
full universe; that is tens of millions of rows per year.

**Pass A — selection candidates.** One query per year table. Entry dates are the first trading day
of each month, taken from the CRSP calendar built in Stage 2a (~12 dates/year), which is what makes
this cheap.

```sql
SELECT o.secid, o.date, o.exdate, o.optionid, o.cp_flag,
       o.strike_price / 1000.0            AS strike,        -- OM stores strike × 1000
       o.best_bid, o.best_offer,
       (o.best_bid + o.best_offer) / 2.0  AS mid,
       o.best_offer - o.best_bid          AS spread,
       o.volume, o.open_interest,
       o.impl_volatility, o.delta, o.vega, o.gamma,
       o.cfadj, o.ss_flag, o.contract_size, o.exercise_style,
       (o.exdate - o.date)                AS dte            -- Postgres date diff → integer days
FROM   optionm.opprcd2019 o
WHERE  o.secid IN %(secids)s
  AND  o.date  IN %(entry_dates)s                           -- ~12 dates, huge reduction
  AND  (o.exdate - o.date) BETWEEN 20 AND 40                -- brief's window; do not stretch
  AND  o.best_bid  > 0                                      -- brief
  AND  o.best_offer > o.best_bid                            -- brief: excludes crossed quotes
  AND  o.open_interest > 0                                  -- brief
  AND  o.impl_volatility IS NOT NULL                        -- brief
  AND  o.ss_flag = '0'                                      -- standard settlement only
  AND  o.contract_size = 100                                -- drop adjusted/non-standard deliverables
  AND  (   (o.cp_flag = 'C' AND o.delta BETWEEN  0.35 AND  0.65)
        OR (o.cp_flag = 'P' AND o.delta BETWEEN -0.65 AND -0.35) );
```

Notes on this query:
- `ss_flag` may be `char` or `smallint` depending on vintage — verify in Stage 0 and adjust the
  literal.
- The delta band is a **pre-filter** to keep the result set small. Final selection (nearest to
  ±0.50) happens in pandas. Widen to ±0.35 if coverage turns out thin; do not narrow it, or you
  will silently truncate the cross-section on high-skew names.
- Add BK's data screens (M11) here or immediately after: `impl_volatility BETWEEN 0.01 AND 1.00`
  is theirs. For single names in 2020, a 100% IV cap may be **too tight** — see Open Question Q8.

Then, in pandas, per (secid, entry_date, cp_flag):
1. Pick the expiration minimizing `|dte − 30|`; ties → the shorter one.
2. Within that expiration, pick the contract minimizing `|delta ∓ 0.50|`.
3. Tie-break on higher `open_interest`.
4. If no qualifying contract exists → drop the stock-month entirely (brief's rule) and **log it**;
   coverage is a result, not a nuisance.

**Pass B — holding paths, selected contracts only.**

```sql
SELECT secid, date, optionid, best_bid, best_offer, impl_volatility,
       delta, vega, open_interest, volume, cfadj
FROM   optionm.opprcd2019
WHERE  optionid IN %(selected_optionids)s
  AND  date BETWEEN %(entry_date)s AND %(exdate)s;
```

**Trap:** a December entry expires in January of the next year, so its path spans two yearly
tables. Pass B must `UNION ALL` across `opprcd{Y}` and `opprcd{Y+1}` for December cohorts. This is
a guaranteed silent-truncation bug if missed — it will just quietly drop December from your sample.

**Supporting pulls:**

```sql
-- Risk-free curve (annualized, continuously compounded, in percent → divide by 100)
SELECT date, days, rate FROM optionm.zerocd
WHERE date BETWEEN DATE '2017-01-01' AND DATE '2024-02-29';
-- Linearly interpolate in `days` to each option's remaining maturity, daily (M4).

-- Discrete projected dividends, for the escrowed-dividend adjustment (M9)
SELECT secid, ex_date, amount, distr_type
FROM   optionm.distrd
WHERE  secid IN %(secids)s
  AND  ex_date BETWEEN DATE '2017-01-01' AND DATE '2024-02-29';
-- distr_type '1' = ordinary cash dividend; verify the code list in Stage 0.
```

**Checkpoint 2b (AAPL, 2019):**
- Exactly ≤ 12 entry months, and you can name why any month is missing.
- Every selected contract: `20 ≤ dte ≤ 40` and `|delta| ∈ [0.35, 0.65]`, with the *median* `|delta|`
  within ~0.02 of 0.50. If the median sits at 0.55, your delta-sign or moneyness convention is off.
- **Put-call parity on entry dates:** for the selected call and put (same expiry, but different
  strikes, so use the parity-implied forward rather than a naive check) — confirm the implied
  forward from each pair is within a few bp of `S e^{(r−q)τ}`. This catches strike-scaling errors
  (the ÷1000), sign errors on delta, and rate-units errors all at once.
- Pass B row count ≈ (number of positions) × (~21 trading days), within ±10%. A large shortfall
  means the December/year-boundary UNION is missing or contracts stop quoting mid-life.
- Selected strike is within ±10% of spot (should follow automatically from ~0.50 delta; if not,
  something is wrong with the delta field).

---

### Stage 3 — Physical volatility estimation
*The brief folds this into Stage 4; separate it. It has its own failure modes and its own test.*

Two estimators, per the paper (M7, M8), computed on **daily log price returns** of the
split-adjusted series, using the 2016 burn-in:

- `VOL^h_t` — rolling sample s.d., Eq. (29), annualized ×√252, over the prior 30 calendar days
  (BK use "the 30 calendar day period prior to t," Sec. 5).
- `VOL^g_t` — GARCH(1,1) per stock, Eqs. (25)–(28), parameters **re-estimated annually on the
  trailing year** (M6), then Eq. (28)'s trailing average of fitted σ̂. Use `arch`.

**Checkpoint 3:**
- On synthetic GBM with known σ = 25%, both estimators recover 25% ± sampling error.
- Real data: AAPL 2019 `VOL^h` sits in a plausible 15–35% band; the 2020-03 spike reaches 60–90%
  for typical large caps. If your March 2020 vol is 20%, your annualization or window is wrong.
- GARCH converges for **every** stock-year. Log failures and define the fallback (fall back to
  `VOL^h`, don't drop the stock). Expect a handful of non-convergences across 300 names × 8 years.
- Correlation between `VOL^h` and `VOL^g` across the panel should be high (>0.85). A low
  correlation means one of them is broken.
- Cross-check a sample against `optionm.hvold` if that table resolves — an independent
  implementation agreeing with yours is cheap validation.

---

### Stage 4 — The delta-hedging engine

Implement `delta_hedged_gain` per the contract in §2, following M1–M5 exactly.

Per rebalance date t_n:
- `Δ_n = delta_fn(S_n, K, τ_n, σ̂_n, r_n, q_n)` — σ̂ from Stage 3, **never** `impl_volatility`.
- Hedge P&L increment: `−Δ_n (S_{n+1} − S_n)`.
- Financing increment: `−r_n (C_t − Δ_n S_n)(τ/N)`, r_n updated daily (M4).
- Terminal: `max(S_T − K, 0)` for calls, `max(K − S_T, 0)` for puts (M1, M2).
- Missing stock day inside the hold (halt, etc.): carry Δ forward, no rebalance, and record it in
  `diagnostics`.

**Checkpoint 4 — three independent validations. This is the most important checkpoint in the
project and the best interview material in it.**

**(a) The zero-VRP synthetic test (`tests/test_hedging.py`).** Simulate 10,000 GBM paths with
constant σ = 30%, r = 2%, q = 0, price the option with Black–Scholes at the *same* σ, and run it
through the engine. Proposition 1 / Eq. (16) says mean π must be zero up to O(1/N). **Assert
|mean π| is small relative to its standard error** (i.e. t-stat within ±2), and confirm the mean
shrinks as you increase rebalancing frequency N.

This is the whole ballgame. If your engine has a sign error, a financing error, or an off-by-one in
the rebalance loop, it shows up here as a spurious nonzero mean — in a world where the paper proves
the answer is zero. And "I validated the engine against the paper's own null hypothesis, in
simulation, before touching real data" is a genuinely strong thing to be able to say in an
interview. Do this *before* Stage 4(b).

**(b) Single position, hand-checkable.** One AAPL position in 2019. Print the full daily path.
Verify by eye: Δ starts near 0.50; Δ → 1 or 0 as expiry approaches and the option moves ITM/OTM;
the financing term is small and the right sign; `cum_pnl` on the last row equals the scalar `pnl`.
Recompute two consecutive days by hand in a spreadsheet.

**(c) The implied-vol placebo.** Re-run the same positions with `delta_fn` fed `impl_volatility`
instead of σ̂. Per the brief's own economic argument, hedging with implied vol should partially
hedge away the premium, so the measured loss should **shrink**. If it doesn't move, your vol input
is not actually reaching the delta calculation. This is a free wiring test *and* a robustness table
you'll want in the writeup anyway.

---

### Stage 4.5 — SPX anchor
*Not in the brief. Cheap, high value — add it.*

Run the finished engine on **SPX options (secid 108105) over 2017–2023**: one secid, ATM ~30-day
calls, monthly. This is the paper's own asset class with your own code. You should recover a
significantly negative mean π/S of roughly the paper's order of magnitude.

Why it's worth the half-day: it separates "my extension found something" from "my engine has a
bug." If SPX comes out *positive*, you have an engine problem, not a finding. And in an interview,
"I anchored my implementation on the paper's own underlying before running the extension" is the
answer to "how do you know your code is right?"

Note the honest caveat: SPX options are European and cash-settled, so this validates the engine but
does **not** validate your American-option handling for single names.

**Checkpoint 4.5:** SPX mean π/S significantly negative; magnitude within a factor of ~3 of the
paper's −0.10% (different sample period, so don't demand a match); more negative in 2018 and 2020
than in 2017 and 2019.

---

### Stage 5 — Statistical analysis

Unit of observation: one (permno, entry_month, cp_flag) position — roughly 150 × 84 × 2 ≈ 25,000,
less attrition. Report, mirroring Table 1 (M13):

| Column | Note |
|---|---|
| Mean π (dollars) | per contract, ×100 for per-option-contract dollars if you prefer |
| Mean π/S (%) | **headline** — theoretically motivated by Lemma 1 / Eq. (19) (M14) |
| Mean π/C (%) | the "% of option value" number, comparable to the paper's −3.88%/−12.18% |
| Fraction with π < 0 | paper's `1_{π<0}`; robust to outliers, hard to fake |
| Mean spread / mean \|π\| | **M18** — is the loss bigger than the cost of trading it? |
| N | |

**Inference — this is where the extension diverges hardest from the paper, and where the project is
most likely to overclaim.** BK have one underlying; you have 150 that all load on the same market
volatility factor. Two positions in the same month are *not* independent draws. A naive pooled
t-stat over 25,000 observations will produce something absurd like −40 and it will be wrong. BK
themselves flag the weaker version of this problem (M15).

Report three, in this order, and lead with the conservative one:
1. **Month-level portfolio time series** (primary): average π/S across all names within each entry
   month → 84 monthly observations → mean, Newey–West t-stat (lag 6). Your effective N is 84, not
   25,000. Say so.
2. **Two-way clustered** (secondary): cluster by permno and by entry month.
3. **Naive pooled** (reported only for comparison, and explicitly labeled as overstated).

If (1) and (3) differ by an order of magnitude — they will — that gap *is* a finding worth a
paragraph in `methodology.md`. Being the candidate who noticed it is worth more than the point
estimate.

**Vol-regime split (M16).** Two versions, because they answer different questions:
- **Time-series regime:** VIX terciles at entry date. Tests "are hedged losses worse in high-vol
  markets" — the direct analogue of the paper's Table 3.
- **Cross-sectional regime:** each stock's own `VOL^h` terciles *within* month. Tests "do high-vol
  *stocks* lose more" — this has no analogue in the paper because the paper has one underlying, and
  it is the genuinely new question the extension can ask. Flag it as such.

**Time-series regression (M17).** Panel version of Eq. (33):
`GAINS_{i,t} = Ω0 + Ω1 VOL_{i,t} + Ω2 GAINS_{i,t−1} + ε`, with stock fixed effects and standard
errors clustered by month. Null Ω1 = 0; predicted Ω1 < 0. Also run the pure time-series version on
the monthly cross-sectional average, matching the paper's specification exactly (Newey–West, lag
12) so there is one number directly comparable to their −0.032.

**Mishedging robustness (M19).** Add contemporaneous underlying return as a regressor; check whether
Ω3 > 0, which would mean BS underhedges and π is biased *upward* — i.e., against your finding, which
strengthens it.

**Checkpoint 5:**
- Sign is negative and the month-clustered t-stat is significant. **If it isn't, that is a
  publishable-to-your-README result, not a failure** — write it up honestly and investigate whether
  spreads, the 2017/2019 low-vol years, or single-name idiosyncratic vol explain it.
- The mean loss exceeds half the mean bid-ask spread. If not, say so plainly.
- 2018 and 2020 subperiods show more negative gains than 2017 and 2019.
- Results survive dropping the top and bottom 1% of π (outlier check); the *median* is also
  negative (the paper's Table 3 medians are more negative than its means — check whether yours
  behave the same way).
- Calls and puts give the same sign independently.

---

## 4. Open questions, ambiguities, and risks

Ordered roughly by how much they change the build. **Q1, Q2, Q3 and Q8 are RESOLVED** (decisions
recorded in the planning session of 2026-09-01); the rest have defensible defaults proposed below.

### Q1 — Unit of observation: separate calls and puts, or the straddle? ✅ DECIDED
**Decision: separate call and put observations.** Paper-faithful (BK analyze calls in Table 1 and
puts in Table 2, never combined), doubles N, and gives two independent checks on the sign. The
combined position goes in a secondary table. Rationale retained below.

<details><summary>Original discussion</summary>
The brief's Stage 3 is titled "straddle/contract construction" and selects both a call and a put per
stock-month, but Stage 5 says "mean delta-hedged P&L" without specifying whether the observation is
the call, the put, or their sum. These are different analyses. BK analyze **calls and puts
separately** (Tables 1 and 2), never as a combined straddle. Selecting the ~0.50-delta call and
~−0.50-delta put gives you a *strangle*, not a straddle, since they'll usually sit at different
strikes.
</details>

### Q2 — Delta model: American or European? ✅ DECIDED
**Decision: Black–Scholes European delta with an escrowed-dividend adjustment, as the primary
hedging delta (paper's Eq. 30).** CRR binomial American delta added later as a robustness column on
a subsample. OptionMetrics' own binomial delta is still used for *contract selection*, per the
brief. Rationale retained below.

<details><summary>Original discussion</summary>
Single-name equity options are **American**; SPX options are European. This matters twice:
- **Selection:** OptionMetrics' `delta` field comes from a binomial (Cox–Ross–Rubinstein) American
  model with discrete projected dividends. Using it for *selection* (per the brief) is fine and
  arguably better than a BS delta.
- **Hedging:** here you must recompute at physical vol, and you choose. **(a)** BS European delta
  with an escrowed-dividend adjustment — paper-faithful (Eq. 30), ~20 lines, but slightly
  mis-specified for American options. **(b)** CRR binomial American delta with discrete dividends —
  more correct, ~150 lines, ~50× slower (still fine at this scale), and harder to defend as "what
  the paper did."

**My recommendation: (a) as primary — the paper's specification is the thing you're extending, and
for ~0.50-delta 30-day options with modest dividends the early-exercise premium is small — with (b)
as a robustness column on a subsample.** Note there's a real interview answer here either way, but
you must not be caught unaware that these are American options.
</details>

### Q3 — Physical volatility: rolling realized or GARCH as primary? ✅ DECIDED
**Decision: build VOL^h (rolling realized) first so Stage 4 unblocks, then add GARCH(1,1) refit
annually and report GARCH as the primary hedge-ratio input to match the paper's Eq. (30); VOL^h
becomes the robustness column.** Window for VOL^h: **30 calendar days**, matching the paper's Sec. 5
("computed over the 30 calendar day period prior to t"), with 60-day reported as a stability check
since single names are noisier than an index. Rationale retained below.

<details><summary>Original discussion</summary>
BK compute both and use **GARCH for the hedge ratio** (M5), while noting `VOL^h` has higher
explanatory power in the regressions (M8). Rolling realized is trivial; GARCH is 300 names × 8
annual refits ≈ 2,400 fits, all of which can fail in interesting ways.
**My recommendation: build `VOL^h` first (Stage 3 ships faster, Stage 4 unblocks), then add GARCH
and report it as the primary hedge-ratio input to match the paper, with `VOL^h` as the robustness
column.** Also: what window for `VOL^h`? BK use 30 calendar days (Sec. 5); 60 days is more stable
for single names.
</details>

### Q4 — Are we comfortable that the headline t-stat is effectively N=84? ⚠ worth a decision
See §3 Stage 5. Just flagging that the honest number will be much smaller than the naive one, and
you should decide *now* that you'll lead with the conservative version, before you see a t-stat of
−40 and get attached to it.

### Q5 — The OptionMetrics dividend yield field may not exist for equities
The brief says "use OptionMetrics' dividend yield field in Black-Scholes delta calculations." I
believe IvyDB provides a continuous dividend-yield series for **indices** (`optionm.idxdvd`) but for
individual equities provides **discrete projected dividends** (`optionm.distrd`) instead.
Conveniently, discrete-and-escrowed is also exactly what the paper does (M9). **Proposed default:
build `q` from `distrd` by subtracting the PV of dividends with ex-dates inside the option's life
from S, and additionally expose an equivalent continuous `q = −ln(1 − PVD/S)/τ` for the BS
formula.** Verify in Stage 0; if a per-equity yield field does exist, use it and note the choice.

### Q6 — Risk-free rate source (deliberate deviation from the paper)
BK back r out of put-call parity (M10). For 150 single names with strike-mismatched pairs that's
noisy and slow. **Proposed default: `optionm.zerocd`, linearly interpolated in days to each option's
remaining maturity, updated daily.** This is standard practice in the single-name literature and
strictly cleaner data. Document it in `methodology.md` as a conscious deviation, with the reason.

### Q7 — Delisting, M&A, and halts mid-hold
2017–2023 contains plenty of S&P 500 deletions (acquisitions, spinoffs). If a name is acquired 10
days into a 30-day hold, the option is early-terminated or converted and CRSP prices stop. **Proposed
default: exclude any position whose underlying delists (`crsp.dsedelist`) before its expiry, and
report the count.** With ~25,000 positions the count should be tiny; if it's material, revisit. The
alternative — closing at the delisting price — introduces its own assumptions.

### Q8 — BK's implied-vol screen: literal threshold vs. original intent ✅ DECIDED
**Decision: cap at 300% (floor at 1%) as primary; the literal 100% cap reported as a robustness
row.** Both thresholds are fixed here, in the plan, before any results exist.

The reasoning, which is the interview answer:

BK's screen reads (Sec. 2, p. 538) — "**to minimize the impact of recording errors**, we discard all
options that have Black–Scholes implied volatilites exceeding 100%, or less than 1%." It is a
**data-error filter**, not an economic screen. On SPX in 1988–1995 the threshold and the intent
coincided: a 100% implied vol on the index was essentially impossible as a genuine quote, so
anything above it was bad data.

On individual equities in 2017–2023 the threshold and the intent **come apart**. In March 2020,
30-day ATM implied vol above 100% was a real, correctly-recorded market observation on plenty of
large caps. Applying 100% literally would therefore delete genuine data BK never intended to
delete — and delete it **asymmetrically**, only on the high-vol side, biasing the measured premium
toward zero in precisely the month that carries the most information about it.

So "closest to the paper" is ambiguous, and the two readings disagree:
- **Literal fidelity** → keep 100%.
- **Fidelity to purpose** → pick the single-name analogue of "this is a recording error," which is
  far above 100%.

We take the second and report the first, so the reader sees both and the choice was never made
after seeing the answer. Note this cuts *against* the hypothesis rather than for it: we are keeping
the observations that are hardest to hedge well, not discarding inconvenient ones.

**Report in the robustness table:** N dropped and mean π/S under each cap, broken out by year, so
the March-2020 effect is visible rather than asserted.

### Q9 — Coverage rate is a result, not a nuisance
The 20–40 day window plus the liquidity filters will silently drop stock-months. If the drop rate is
5%, ignore it. If it's 30%, and the drops concentrate in high-vol months or smaller names, you have
a selection bias that could produce your whole result. **Track and report the coverage rate by year
and by market-cap quintile.** Make this a table in the writeup.

### Q10 — Entry timing within the month
"Stock-month" is unspecified. **Proposed default: first trading day of each calendar month.** The
alternative (a fixed number of days before monthly expiration) gives cleaner ~30-day maturities but
correlates entry dates with the expiration cycle. First-of-month is simpler and easier to defend;
just be consistent and say which you chose.

### Q11 — The split-adjustment trap
A 4-for-1 split mid-hold changes S, and OptionMetrics adjusts the contract (strike, `cfadj`,
`contract_size`). Mixing adjusted stock prices with as-traded strikes produces a catastrophic fake
P&L. **Proposed default: normalize the whole position to the entry-date adjustment factor** —
`S_used(t) = |prc_t|/cfacpr_t × cfacpr_entry` — so entry-date S and K are as-traded and the series
is internally consistent. Cross-check against `cfadj` on the option side. Add a unit test using
AAPL's 2020-08-31 split as a fixture; this is exactly the bug that quietly produces a spectacular
false result.

### Q12 — Transaction costs and the entry mark
The brief marks at mid, which matches the paper's use of closing prices. That's the right primary
spec. But for single names the spread is the whole economic question (M18). **Proposed addition: a
robustness column entering at the ask (the cost of actually being long the option).** If the loss
survives paying the full spread, the finding is strong; if it doesn't, that's an honest and
interesting result about why the premium isn't arbitraged in single names.

### Q13 — Early exercise vs. the intrinsic-value terminal
M1's `max(S_T − K, 0)` assumes European settlement. You're *long* the option, so early exercise is
your right, not a risk imposed on you — but for a deep-ITM call just before a large ex-dividend
date, holding to expiry is genuinely suboptimal and the intrinsic-value terminal understates the
position's value. At ~0.50 delta and ~30 days this is second-order. **Proposed default: keep the
intrinsic-value terminal (paper-faithful), and check how many positions finish deep ITM with an
ex-date inside the final week.** If it's a handful, note it and move on.

### Q14 — Minor items, defaults proposed, no answer needed
- **Realized-vol input:** log *price* returns (`prc/cfacpr`), not total returns — BS σ is the
  diffusion of the price process. Cross-check against `log(1+ret)`.
- **Quote timing:** OptionMetrics quotes are ~15:59 ET; CRSP `prc` is the close. Sub-minute
  non-synchronicity, immaterial at daily frequency. Note it and move on.
- **Reproducibility:** cache every WRDS query to parquet under `data/raw/` keyed by a hash of the
  SQL. WRDS sessions are slow and rate-limited; you will re-run Stage 4 many times and you should
  never re-pull for it.
- **`crsp.dsp500list` column names** (`start`/`ending`) — verify; some vintages differ.

---

## 5. Suggested sequencing

| Stage | Rough effort | Gate |
|---|---|---|
| 0 — env + schema recon | half day | all table names verified and written down |
| 1 — universe | half day | TSLA survivorship test passes |
| 2a — CRSP + link | 1 day | split test passes, link coverage >98% |
| 2b — options | 1–2 days | AAPL 2019 selection clean, parity check passes |
| 3 — volatility | 1 day | recovers known σ on synthetic; GARCH converges |
| 4 — hedging engine | 1–2 days | **synthetic zero-VRP test passes** |
| 4.5 — SPX anchor | half day | SPX π/S significantly negative |
| 5 — analysis | 1–2 days | three t-stat flavors reported; spread comparison in main table |

Write `docs/methodology.md` incrementally at each gate, not at the end. It is the document you'll
actually reread the night before an interview, and it's where every "why did you do it that way"
answer lives.
