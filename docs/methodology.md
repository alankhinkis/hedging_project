# Methodology

The "why did you do it that way" document. Written incrementally at each stage gate, not at the
end. Plan references (M#, Q#, Checkpoint N) point at `PHASE1_PLAN.md`; paper references are to
Bakshi & Kapadia (2003, RFS 16(2), 527–566), `bakshi_kapadia_2003_rfs.pdf`.

**Sections are added as stages complete.** Stages 0 and 1 are written up below.

---

## 0. Framing

BK test delta-hedged gains on **S&P 500 index options only**, 1988:01–1995:12 — 36,237 calls and
35,030 puts on a single underlying. Their Section 7 (p. 561) says:

> "There are two natural extensions to this article. First, given that volatilities of individual
> stocks and the market index comove highly, one could examine whether the volatility risk premium
> is negative in individual equity options."

This project is that extension. It is **not** a replication of BK, and it is not novel: the
extension was carried out in the published literature, closest of all by **Cao & Han (2013, JFE)**,
"Cross section of option returns and idiosyncratic stock volatility," which runs BK-style
delta-hedged gains on individual equity options, and **Goyal & Saretto (2009, JFE)**, which trades a
related IV−RV signal on single names. *(Both citations are from general knowledge and should be
verified against the papers themselves before being repeated in an interview.)*

How this setup differs, in one sentence: **2017–2023, a point-in-time top-150 S&P 500 universe, one
ATM contract per stock-month held to expiry, hedged at physical volatility** — a narrower, cleaner,
more liquidity-controlled sample than the broad cross-sections those papers use.

### Target magnitudes — for bug detection, not for reproduction

Different underlying, different decade. These numbers say whether the engine is broken, not
whether the finding is right.

| Measure | Paper value | Source |
|---|---|---|
| Mean π/S, ATM | **−0.10% to −0.11%** — the stable anchor | Table 1 |
| Mean π/C, ATM buckets | −3.88%, −7.59% | Table 1 |
| Fraction of ATM observations negative | 68% | Table 1 |

π/S is the headline because Lemma 1 / Eq. (19) *prove* π scales with S (M14). π/C is quoted
alongside it but never alone: the option price sits in the denominator and collapses for OTM
options, which is why the paper's full-sample π/C (−12.18%) is three times its ATM figure.

---

## Stage 0 — Environment and schema reconnaissance

### Why this stage exists

It is not in the original brief. Every stage downstream is written against table and column names
that were, until this stage runs, **guesses**. CRSP and OptionMetrics have both been through schema
migrations on WRDS (the CRSP "v2" tables rename nearly every column; OptionMetrics has moved
between `optionm` and `optionm_all`), so a plan that hard-codes `crsp.dsp500list.start` is one
migration away from silently pulling nothing.

### How it is implemented

`config.yaml:schema_targets` lists each *need* with a primary guess, a confidence level, and the
key columns that need must have. `schema_fallbacks` lists alternate names to try. Stage 0 resolves
each need against the live server and writes two artefacts:

* `docs/wrds_schema_notes.md` — human-readable: exact `schema.table`, full column list with types,
  row count, date range, and the date verified.
* `docs/wrds_schema_resolved.json` — machine-readable `{need → table}`.

Every later stage calls `vrp.schema.table_for("Daily stock prices")` rather than naming a table.
If Stage 0 has not run, that call logs a loud warning and falls back to the unverified guess — it
degrades rather than crashing, but it tells you.

Column names are resolved the same way, per pull: `_resolve_columns` maps a logical name
(`"start"`) onto whichever of `start`, `mbrstartdt`, `begdt` actually exists on the resolved table.
This is what makes the code survive a CRSP v1→v2 migration.

### Checkpoint 0

Encoded in `vrp.schema.checkpoint_0` and printed by the script:

1. every schema target resolves to a real table;
2. every declared key column exists on it;
3. the OM↔CRSP link table is located **and its `score` distribution is recorded** — Checkpoint 0
   requires knowing which score values are trustworthy, and the Stage 2a filter should be chosen
   from that distribution rather than from folklore;
4. the live sanity query returns rows (AAPL options on 2019-01-02), routed through the link so it
   exercises permno → secid → option chain end to end;
5. **Q5 is answered from the server**: the probe lists every dividend/yield-like table in `optionm`
   and the `distr_type` code distribution. The expectation is that IvyDB provides a continuous
   dividend *yield* for indices (`idxdvd`) and **discrete projected dividends** for single names
   (`distrd`), which is also exactly what the paper does (M9) — but that is now checked, not
   assumed.

The probe also records `ss_flag`'s dtype and observed values and `contract_size`'s values, because
the Stage 2b selection SQL compares against literals whose type differs by vintage (`'0'` vs `0`).

### Reproducibility: the query cache

Every WRDS pull goes through `cached_query`, which writes parquet under `data/raw/` keyed by
`sha256(normalized SQL + bound parameters)`, with a JSON sidecar recording the SQL, the parameters,
the row count and the pull timestamp. Two properties are unit-tested (`tests/test_wrds_cache.py`):

* reformatting the SQL (whitespace) or reordering the parameters does **not** invalidate the cache,
  so cosmetic edits do not trigger a re-pull;
* changing the SQL or any parameter value **always** does.

This matters because Stage 4 will be re-run many times and WRDS sessions are slow and
rate-limited. It also means the whole build is auditable after the fact: `vrp.wrds_conn.cache_manifest()`
lists every query that ever produced a number in the output.

Credentials: `WRDS_USERNAME` from `.env`; the password lives only in `~/.pgpass` and is never read,
logged, or committed by this code.

### What Stage 0 actually found (run 2026-09-02, all 12 targets resolved)

Five of these change what later stages must do. They are the reason this stage exists.

**1. `exercise_style` does not exist on the option price file.** `optionm.opprcd2019` carries
`am_settlement` (0/1) and `expiry_indicator` instead. Nothing downstream needs an exercise-style
flag — US single-name equity options are American regardless, and Q2 already fixed the hedging
delta as BS-European with an escrowed-dividend adjustment. The plan's Pass A query
(`PHASE1_PLAN.md:370`) selects this column and would have failed on first run.

**2. `ss_flag` is `VARCHAR(1)` with observed value `'0'`; `contract_size` is `DOUBLE`.** The plan
flagged this as vintage-dependent and needing verification. Verified: the selection SQL wants
`ss_flag = '0'` **quoted** and `contract_size = 100` **unquoted**.

**3. `expiry_indicator` marks weeklies.** Value `'w'` on the AAPL probe; NULL on standard
monthlies. The plan does not mention this column, but with a 20–40 day window and monthly entry
dates, weeklies will be in the candidate set. Whether to keep them is a Stage 2b decision — they
are genuine, liquid contracts on large caps, but they change the expiry-cycle composition of the
sample. Flagged, not yet decided.

**4. The OM↔CRSP link's `score` semantics.** Distribution across `wrdsapps.opcrsphist`:

| score | rows | distinct permno | reading |
|---:|---:|---:|---|
| 1 | 28,336 | 27,997 | best match |
| 2 | 190 | 181 | |
| 3 | 8 | 8 | |
| 4 | 660 | 404 | |
| 5 | 5,687 | 3,382 | |
| 6 | 86,892 | **0** | **no CRSP match at all — permno is null** |

Score 6 is not a weak link, it is the absence of one, and it is by far the largest bucket. The
Stage 2a filter must therefore be `score <= 5 AND permno IS NOT NULL`; score 1 alone covers 99.3%
of matched rows. Selecting on `score` without excluding 6 would look like a working join and
silently produce nothing.

**5. Q5 is resolved, and the answer is better than the plan expected.** There is no continuous
dividend yield for equities (`idxdvd` is the index file, as predicted). But there are *two*
discrete dividend sources, not one:

* `optionm.distrd` — `(secid, ex_date, amount, distr_type, cancel_flag, approx_flag, ...)`:
  **announced** distributions. AAPL 2019 shows four quarterly ordinary dividends (`distr_type='1'`).
* `optionm.distrprojd{YYYY}` — `(secid, date, exdate, amount)`: the dividend schedule **as
  projected on each `date`**. Year-split, like the price files.

The projection file is the right input for the escrowed-dividend adjustment (M9), because it says
what was expected on the entry date rather than what was later declared. For 30-day holds on large
caps the two will rarely differ — dividends are announced well in advance — but the projection file
removes the look-ahead by construction rather than by argument, at no extra cost. **Decision: use
`distrprojd{YYYY}`, and use `distrd` as a cross-check on the realised amounts.** Note the column
is `exdate` there and `ex_date` in `distrd`.

Two smaller items: `optionm.hvold` is year-split as `optionm.hvold2019` etc. (the plan rated this
guess "low confidence" and was right to), and the AAPL probe returned 1,590 rows across 14
expiries with strikes 2.5–425, confirming permno 14593 → secid 101594 end to end.

---

## Stage 1 — Point-in-time universe

### The survivorship fix

A name is in the universe in month *m* only if *m* falls inside one of its `dsp500list` membership
spans. That is the entire fix, and it lives in one predicate in `members_by_month`:

```python
inside = (merged["date"] >= merged["from_date"]) & (merged["date"] <= merged["thru_date"])
```

Nothing is carried forward and nothing is back-filled from today's index composition. The
synthetic version of this is unit-tested two ways: a name with market-cap data all year that joins
the index mid-year must be absent before its add date, and a name that is deleted mid-sample must
vanish after it — the TSLA and deletion spot-checks from Checkpoint 1, in miniature, with no WRDS
connection required.

### Ranking lags the entry month by one — a deviation from the plan

The plan says: rank members at each month-end, take the top 150. Taken literally, positions opened
on the **first trading day of month *m*** would be selected using **month-end *m*** market caps,
which are not known on that day. That is look-ahead bias. Small in effect — market-cap rankings are
persistent — but indefensible, and exactly the kind of thing a careful reader looks for.

So the universe carries two columns: `rank_month` (the month-end whose data produced the ranking)
and `entry_month = rank_month + 1` (the month whose first trading day the position is opened on).
Stage 2b selects contracts for month *m* using `entry_month == m`. The cost is one month of
burn-in; the December-2016 ranking governs January-2017 entries, which the burn-in window already
covers.

### Market cap from `msf`, not `dsf`

Ranking happens once a month, and `msf` is roughly 20× smaller than `dsf`. `shrout` is in
thousands, so `|prc| × shrout` is market cap in $000s — the units only matter for readability
since the ranking is scale-invariant.

A negative `prc` is CRSP's flag that the field holds a **bid-ask average** rather than a closing
trade. The magnitude is still the right price, so `ABS(prc)` is taken and the observation is kept;
dropping it would silently delete illiquid month-ends. (Stage 2a carries the same convention plus
an explicit `price_is_quote_avg` flag, where it actually affects P&L.)

### Multi-class names are not consolidated

The S&P 500 holds slightly more than 500 permnos — GOOG/GOOGL, BRK.B, FOX/FOXA. Checkpoint 1
therefore expects a member count of **495–510** per month; exactly 500 would be suspicious. Ranking
is per permno and the top-150 cut is applied afterwards, so a dual-class name can occupy two slots.
That is the honest reading of "top 150 names by market cap" when the tradable unit — the thing an
option is written on — is the permno, not the issuer.

Ties are broken by `rank(method="first")` so that ranks are always contiguous `1…N`; a
`method="min"` tie would produce a gap and fail the checkpoint on a legitimate tie.

### Checkpoint 1

Encoded in `vrp.data.universe.checkpoint_1`:

1. **member count per month ∈ [480, 520]** (the band is in config; 495–510 is the expected range,
   with slack for the odd transition month). 300 or 700 means the point-in-time join is wrong.
2. **TSLA (permno 93436) absent 30 days before its 2020-12-21 index add, present 30 days after.**
   The single best one-line proof the point-in-time logic works.
3. **A mid-sample deletion vanishes.** Rather than hard-coding a name — which rots — the check
   picks the largest-cap name whose membership span ends inside the sample, straight from the
   membership table, and asserts it has no universe rows after that month.
4. **The ranking is genuinely time-varying**: XOM and GE in the January-2017 top-150 (both large
   then, both much smaller by 2023); AAPL, MSFT, AMZN in March 2020. Tickers are resolved
   point-in-time from the CRSP name history, so a renamed permno picks up the ticker it had on the
   ranking date.
5. **Structural integrity**: no permno twice in a month; ranks contiguous `1…N`.
6. **Turnover**: distinct permnos across the sample must exceed 200. If it comes out near 150, the
   universe is static and survivorship bias has been reintroduced — this is the check that catches
   the failure mode the whole stage exists to prevent.

`output/tables/universe_turnover.csv` reports entries and exits per month, and
`universe_member_counts.csv` the raw member count, so (1) and (6) are inspectable rather than
merely asserted.

### What Stage 1 actually produced (run 2026-09-02, all 7 checks pass)

697 membership spans covering 693 distinct permnos over 2016–2023; 58,724 month-end market caps.
The universe is 14,400 (month, permno) rows across 96 entry months, of which **84 entry months ×
150 names = 12,600 fall inside the 2017–2023 sample**. The extra months at each end are the
burn-in: the ranking runs from December 2016 (which governs the January 2017 entries) through
December 2023 (which governs January 2024, unused). Stage 2b filters to the 84.

Turnover is 2–5 names per month, and **241 distinct permnos appear in the in-sample top-150**
(252 including burn-in). The plan expected 250–320; we are at the low end of that, which is what a
top-*150* cut should give — the plan's range was calibrated on the full index, and the largest 150
names turn over more slowly than the tail.

The checks that carry real information:

* **Member count 502–505 in every month.** Inside the expected 495–510, and the tightness is
  itself reassuring — a join that drifted would not hold a 4-name range across 96 months.
* **The deletion check auto-selected RAI (Reynolds American)**, whose membership span ends
  2017-07-24 — acquired by BAT that month. Zero universe rows after. This is the check picking a
  real corporate event out of the data with nothing hard-coded.
* **The ranking is visibly time-varying.** January 2017's top 8 is AAPL, MSFT, **XOM**, AMZN, JNJ,
  JPM, **GE**, WFC; December 2023's is AAPL, MSFT, AMZN, **NVDA**, GOOGL, GOOG, **TSLA**, META.
  XOM at #3 and GE at #7 in 2017 are exactly the names a static present-day universe would omit.
* **GOOG and GOOGL both sit in the March 2020 top 10**, confirming live the multi-class decision
  above: the cut is per permno, and a dual-class issuer legitimately occupies two slots.

### One extra table the plan did not list

`crsp.dsenames` (name history) was added to the schema targets. It is not needed for any
computation — it exists so the Checkpoint 1 ticker assertions and every diagnostic table are
readable by a human rather than being lists of permnos.

---

## Stage 2a — CRSP prices and the OM↔CRSP link

479,878 daily rows for 252 permnos, 2016-01-04 to 2024-02-29. All six Checkpoint 2a
conditions pass.

### The three CRSP conventions, handled rather than assumed

**Negative prices are quote averages, not errors.** CRSP writes `prc < 0` when there was no
closing trade and the field holds the bid-ask midpoint. We take the magnitude and set
`price_is_quote_avg`. Dropping those rows would delete precisely the illiquid days a hedging
study should care about; taking the raw negative would flip the sign of the position. In this
panel it happens **once** in 479,878 rows, and two further rows have no price at all — so the
convention is nearly irrelevant here, which is worth knowing rather than guessing.

**`cfacpr == 0` is nulled, not divided by.** CRSP writes it in rare degenerate cases and it
would otherwise produce infinities. None occur in this panel.

**Splits are normalised to the entry-date factor (Q11).** `normalize_to_entry_factor` expresses
every price in a position's life in the share units that prevailed on its entry date, so
entry-date S and K are as-traded whichever side of a split the position opens on. The unit
tests build AAPL's 2020-08-31 4-for-1 as a fixture and check both directions, including an
explicit test that the *unfixed* version produces the fake −74% overnight move — the failure
mode is pinned, not just the fix.

Live confirmation, from `output/figures/split_adjustment.png` and the checkpoint:

| | raw before → after | raw log change | adjusted log change |
|---|---|---:|---:|
| AAPL 4-for-1, 2020-08-31 | 499.23 → 129.04 | −1.353 | **+0.033** |
| NVDA 4-for-1, 2021-07-20 | 751.19 → 186.12 | −1.395 | **−0.009** |

The adjusted moves are the real returns for those days; the raw ones are the split.

### A sharper reconciliation than the plan specified

The plan proposes comparing cumulated `ret` against the adjusted price return and expecting
them to differ "by the dividend contribution (~1%/yr)" — a soft check with a fuzzy tolerance.
`crsp.dsf` also carries **`retx`**, the return *excluding* dividends, which should reproduce
the adjusted price return essentially exactly. So the check became a two-part one:

* `retx` vs adjusted price return — AAPL 2019: 1.85950 vs 1.85951, **relative error 3.4e-06**.
  A tolerance of 1e-3 now means something.
* `ret` vs `retx` — implied dividend yield **1.51%**, a plausible figure for AAPL in 2019, which
  independently confirms the two return fields are what we think they are.

### The link tie-break is decided by option activity, not by span

This is the one Checkpoint 2a condition that failed on the first run, and the failure was real.

Coverage is **100%** of the 14,400 universe (permno, month) cells — no name is unlinked. But 14
permnos carry more than one secid concurrently, producing 19,093 ambiguous (permno, date) cells.
The plan permits "zero conflicts *or* a documented tie-break rule", so a rule was needed.

The obvious rule — lowest `score`, then the longer span — is **wrong**. Counting actual option
rows for all 29 candidate secids shows that in every one of the 14 cases exactly one secid has
any option data at all and the rest are empty. For 13 names the score-1 secid is the live one.
For **USB (permno 66157) both secids score 1**:

| secid | span | option rows, 2019 |
|---|---|---:|
| 111306 | 1996-01-01 → 2025-12-31 | **0** |
| 104866 | 2001-02-27 → 2025-12-31 | 161,208 |

A span-based tie-break picks 111306 — the dormant one — and USB silently contributes no
tradable contracts for the entire sample. Nothing in the link table distinguishes them.

**Rule adopted: lowest `score`, then most option activity, then longer span, then lower secid.**
Activity is counted only for the ~29 candidate secids on ambiguous permnos, so it costs one
small query. The checkpoint condition was rewritten to match: rather than demanding zero
conflicts (which would fail forever on a legitimate data feature), it asserts that **every
ambiguous link resolves to a secid that actually has option data**. That is the property that
matters. The USB configuration is pinned as a unit test, including an assertion that the
span-based rule picks the wrong secid, so the reasoning cannot be lost.

Note the limit of this check: it verifies the *ambiguous* names resolve to live chains. A name
with a single secid that happens to be dormant would not be caught here — that is Stage 2b's
coverage report (Q9), where missing chains become a first-class result rather than a silent gap.

### Delistings

25 delisting events fall in the window, all on universe names. Per Q7, a position whose
underlying delists before its expiry is excluded and counted — the option is early-terminated or
converted and CRSP prices stop, so there is no honest way to carry the hedge to expiry. 25 events
against ~12,600 stock-months is immaterial, but it is now a number rather than an assumption.

### Entry calendar

The first trading day of each month is derived from the CRSP calendar rather than assumed
(Q10): 98 dates from 2016-01-04 to 2024-02-01, of which the 84 in 2017-01…2023-12 are the entry
dates Stage 2b will use. This is what makes Stage 2b's option pull cheap — 12 dates a year
instead of 250.

---

## Decisions already fixed (from the planning session, 2026-09-01)

Recorded here so they are visibly pre-committed, before any results exist.

| # | Decision | Rationale |
|---|---|---|
| Q1 | **Calls and puts are separate observations**, not a combined straddle. | Paper-faithful (BK's Tables 1 and 2 are separate); doubles N; gives two independent checks on the sign. A ~0.50-delta call and a ~−0.50-delta put usually sit at different strikes, so combining them is a *strangle*, not a straddle. |
| Q2 | **BS European delta with an escrowed-dividend adjustment** as the primary hedge ratio (paper's Eq. 30); CRR American delta as a robustness column on a subsample; OptionMetrics' own binomial delta used for *contract selection*. | The paper's specification is the thing being extended, and for ~0.50-delta 30-day options with modest dividends the early-exercise premium is small. Single-name options *are* American; this is a conscious approximation, not an oversight. |
| Q3 | **VOL^h (30-calendar-day rolling realized) built first; GARCH(1,1), refit annually, reported as the primary hedge-ratio input** to match Eq. (30); VOL^h as the robustness column. 60-day VOL^h as a stability check. | 30 days matches BK Sec. 5. Building the simple estimator first unblocks Stage 4. |
| Q6 | **Risk-free rate from `optionm.zerocd`**, linearly interpolated in days to each option's remaining maturity, updated daily — a deliberate deviation from BK's put-call-parity rate (M10). | For 150 single names with strike-mismatched pairs, parity-implied rates are noisy and slow. The zero curve is standard practice in the single-name literature and strictly cleaner data. |
| Q8 | **Implied-vol screen capped at 300%** (floor 1%) as primary; the paper's literal 100% cap reported as a robustness row. | BK's screen is stated as a *recording-error* filter ("to minimize the impact of recording errors," Sec. 2 p. 538). On SPX in 1988–1995, 100% and "impossible quote" coincided. On single names in March 2020 they do not: 30-day ATM IV above 100% was a real, correctly-recorded observation. Applying 100% literally would delete genuine data **asymmetrically, only on the high-vol side**, biasing the measured premium toward zero in precisely the month carrying the most information. Note this cuts *against* the hypothesis — it keeps the observations hardest to hedge well. |
| Q10 | **Entry on the first trading day of each calendar month.** | Simpler and easier to defend than a fixed offset from monthly expiration, which correlates entry dates with the expiration cycle. |
| Q4 | **The headline t-stat will be led by the month-level portfolio time series (effective N = 84), not the naive pooled t-stat.** Decided before any results exist. | 150 names all load on the same market volatility factor; two positions in the same month are not independent draws. A naive pooled t-stat over ~25,000 observations will produce something absurd. BK flag the weaker version of this themselves (M15). |
