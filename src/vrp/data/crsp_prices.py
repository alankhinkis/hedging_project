"""Stage 2a -- CRSP daily prices, split adjustment, and delistings.

The whole stage exists to produce one trustworthy series: the underlying price the hedge
sees on each day of a position's life. Three things can silently corrupt it, and each is
handled explicitly here rather than by convention.

**Negative prices.** CRSP stores `prc < 0` when there was no closing trade and the field
holds the bid-ask midpoint instead. The magnitude is still the right price. We take the
absolute value and set `price_is_quote_avg`, because dropping those days would delete
exactly the illiquid days a hedging study should care about, and taking the raw negative
would flip the sign of the position.

**Splits.** A 4-for-1 split mid-hold changes S by 75% overnight while the option contract
is adjusted separately by OptionMetrics. Mixing an adjusted stock price with an as-traded
strike produces a spectacular fake P&L. `normalize_to_entry_factor` is the fix (Q11): every
price in a position's life is expressed in the share units that prevailed on its entry
date, so entry-date S and K are as-traded and the series is internally consistent.

**Non-trading days.** `prc` can be null (halt, suspension). Rows are kept and flagged; the
hedging engine carries the previous delta forward rather than rebalancing (M-series, Stage 4).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..schema import table_for
from ..wrds_conn import cached_query, get_connection, resolve_columns, split_table

log = logging.getLogger(__name__)

_DSF_COLS = {
    "permno": ["permno"],
    "date": ["date", "dlycaldt"],
    "prc": ["prc", "dlyprc"],
    "ret": ["ret", "dlyret"],
    "retx": ["retx", "dlyretx"],
    "cfacpr": ["cfacpr", "dlyfacprc"],
    "cfacshr": ["cfacshr", "dlyfacshr"],
    "shrout": ["shrout"],
    "bidlo": ["bidlo", "dlylow"],
    "askhi": ["askhi", "dlyhigh"],
    "bid": ["bid", "dlybid"],
    "ask": ["ask", "dlyask"],
    "vol": ["vol", "dlyvol"],
    "openprc": ["openprc", "dlyopen"],
}
# Columns that are nice to have but not worth failing over on an unusual vintage.
_DSF_OPTIONAL = ("retx", "bid", "ask", "openprc", "bidlo", "askhi", "vol")

_DELIST_COLS = {
    "permno": ["permno"],
    "dlstdt": ["dlstdt"],
    "dlstcd": ["dlstcd"],
    "dlret": ["dlret"],
    "dlprc": ["dlprc"],
    "nwperm": ["nwperm"],
}


# ---------------------------------------------------------------------------
# Pulls
# ---------------------------------------------------------------------------

def fetch_daily_prices(
    permnos: Sequence[int],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
) -> pd.DataFrame:
    """Daily CRSP prices for every permno that was ever in the universe.

    The window runs from the burn-in start (volatility estimation needs a year of returns
    before the first entry) to `hold_buffer_end` -- December-2023 entries are held into
    January 2024, so stopping at the sample end would truncate the last cohort.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    table = table_for("Daily stock prices", cfg)
    cols = resolve_columns(table, _DSF_COLS, conn=conn, optional=_DSF_OPTIONAL)
    schema, tbl = split_table(table)

    select = ",\n               ".join(f"{src} AS {logical}" for logical, src in cols.items())
    sql = f"""
        SELECT {select}
        FROM   {schema}.{tbl}
        WHERE  {cols['permno']} IN %(permnos)s
          AND  {cols['date']} BETWEEN %(a)s AND %(b)s
    """
    df = cached_query(
        sql,
        {
            "permnos": tuple(int(p) for p in sorted(set(permnos))),
            "a": str(cfg.burnin_start),
            "b": str(cfg.hold_buffer_end),
        },
        name="dsf_prices",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("date",),
    )
    df["permno"] = df["permno"].astype("int64")
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


def fetch_delistings(
    permnos: Sequence[int],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
) -> pd.DataFrame:
    """Delisting events (Q7). A position whose underlying delists before expiry is excluded.

    The option is early-terminated or converted and CRSP prices stop, so there is no honest
    way to carry the hedge to expiry. The count is reported rather than assumed negligible.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    table = table_for("Delisting returns", cfg)
    cols = resolve_columns(table, _DELIST_COLS, conn=conn, optional=("nwperm", "dlprc"))
    schema, tbl = split_table(table)

    select = ",\n               ".join(f"{src} AS {logical}" for logical, src in cols.items())
    sql = f"""
        SELECT {select}
        FROM   {schema}.{tbl}
        WHERE  {cols['permno']} IN %(permnos)s
          AND  {cols['dlstdt']} BETWEEN %(a)s AND %(b)s
    """
    df = cached_query(
        sql,
        {
            "permnos": tuple(int(p) for p in sorted(set(permnos))),
            "a": str(cfg.burnin_start),
            "b": str(cfg.hold_buffer_end),
        },
        name="dsedelist",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("dlstdt",),
    )
    if len(df):
        df["permno"] = df["permno"].astype("int64")
    return df


# ---------------------------------------------------------------------------
# Cleaning (pure pandas -- unit-testable without WRDS)
# ---------------------------------------------------------------------------

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the three CRSP conventions, in the order they must happen.

    Adds:
      price_is_quote_avg -- CRSP had no closing trade and stored a bid-ask midpoint
      prc_abs            -- the usable price level, as traded on that date
      s_adj              -- split-adjusted price, comparable across a split
      shrout_adj         -- split-adjusted shares outstanding
      has_price          -- False on halts/suspensions, where prc is null
    """
    out = df.copy()

    # 1. Negative price = bid-ask average, not a bad number.
    out["price_is_quote_avg"] = out["prc"] < 0
    out["prc_abs"] = out["prc"].abs()
    out["has_price"] = out["prc_abs"].notna() & (out["prc_abs"] > 0)

    # 2. Split adjustment. CRSP writes cfacpr = 0 in rare degenerate cases; treating that as
    #    a divisor would produce infinities, so it is nulled and flagged instead.
    cfacpr = out["cfacpr"].replace(0, np.nan)
    out["cfacpr_is_bad"] = out["cfacpr"].notna() & (out["cfacpr"] == 0)
    out["s_adj"] = out["prc_abs"] / cfacpr

    if "cfacshr" in out.columns:
        out["shrout_adj"] = out["shrout"] * out["cfacshr"]

    return out


def normalize_to_entry_factor(
    prices: pd.DataFrame,
    entry_date,
    *,
    permno: int | None = None,
) -> pd.Series:
    """Express every price in the share units that prevailed on `entry_date` (Q11).

        S_used(t) = |prc_t| / cfacpr_t * cfacpr_entry

    This is the split trap. OptionMetrics adjusts the contract (strike, `cfadj`,
    `contract_size`) when the underlying splits; CRSP adjusts the price series. Mixing an
    adjusted price with an as-traded strike is the bug that quietly produces a spectacular
    false result. Normalising to the entry-date factor makes entry-date S and K as-traded
    and keeps the whole path internally consistent, whichever side the split lands on.

    Returns a Series indexed like `prices`.
    """
    frame = prices if permno is None else prices.loc[prices["permno"] == int(permno)]
    entry_date = pd.Timestamp(entry_date)

    at_entry = frame.loc[frame["date"] == entry_date, "cfacpr"]
    if at_entry.empty:
        raise KeyError(f"no price row on entry date {entry_date.date()}"
                       + (f" for permno {permno}" if permno is not None else ""))
    cfacpr_entry = float(at_entry.iloc[0])
    if not np.isfinite(cfacpr_entry) or cfacpr_entry == 0:
        raise ValueError(f"cfacpr on {entry_date.date()} is {cfacpr_entry}; cannot normalise")

    cfacpr = frame["cfacpr"].replace(0, np.nan)
    return frame["prc"].abs() / cfacpr * cfacpr_entry


def price_panel(
    prices: pd.DataFrame,
    permno: int,
    start,
    end,
) -> pd.DataFrame:
    """One permno's cleaned daily rows over [start, end], inclusive -- a position's path."""
    mask = (
        (prices["permno"] == int(permno))
        & (prices["date"] >= pd.Timestamp(start))
        & (prices["date"] <= pd.Timestamp(end))
    )
    return prices.loc[mask].sort_values("date").reset_index(drop=True)


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """The CRSP trading calendar implied by the panel -- Stage 2b's entry dates come from here."""
    return pd.DatetimeIndex(sorted(prices["date"].unique()))


def first_trading_days(prices: pd.DataFrame) -> pd.DataFrame:
    """First trading day of each calendar month.

    The plan's original Q10 rule. Retained because it is the robustness comparison for
    `post_expiry_entry_days`, and because the coverage gap between the two is a reportable
    result rather than a discarded experiment.
    """
    cal = pd.Series(trading_calendar(prices), name="date")
    month = cal.dt.to_period("M")
    firsts = cal.groupby(month).min().rename("entry_date").reset_index()
    firsts.columns = ["entry_month", "entry_date"]
    return firsts


def third_friday(year: int, month: int) -> pd.Timestamp:
    """The standard US equity-option monthly expiration date for a month."""
    start = pd.Timestamp(year=int(year), month=int(month), day=1)
    fridays = pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2]


def post_expiry_entry_days(prices: pd.DataFrame) -> pd.DataFrame:
    """First trading day AFTER each month's monthly expiration (the Q10 rule in force).

    Why this rather than the first trading day of the month. Entering on the 1st puts the
    near monthly expiry ~18 days out and the following one ~46 -- neither inside a 20-40 day
    window. Only names with weekly options have anything in the window, so a first-of-month
    rule silently selects on the presence of weeklies, which tracks size: measured coverage
    was 64.2% in the smallest within-month cap quintile against 99.6% in the largest.

    Entering the day after expiration instead puts the *next* monthly expiration ~28 days
    out for every name, weeklies or not. Measured coverage is 100.0% and flat across
    quintiles, with the 20-40 day window unchanged.

    The cost, which the plan named: entry dates now correlate with the expiration cycle.
    Here that is the mechanism, not a side effect -- it is what makes maturity uniform
    across the cross-section instead of conditional on option-market development.
    """
    cal = trading_calendar(prices)
    if not len(cal):
        return pd.DataFrame(columns=["entry_month", "entry_date"])
    months = pd.period_range(cal.min().to_period("M"), cal.max().to_period("M"), freq="M")

    rows = []
    for period in months:
        expiry = third_friday(period.year, period.month)
        after = cal[cal > expiry]
        if len(after):
            rows.append({"entry_month": period, "entry_date": after[0]})
    return pd.DataFrame(rows)


def entry_calendar(prices: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Entry dates under the configured rule (config.yaml: selection.entry_rule)."""
    cfg = cfg or load_config()
    rule = str(cfg["selection"].get("entry_rule", "first_trading_day_after_monthly_expiry"))
    if rule == "first_trading_day_of_month":
        return first_trading_days(prices)
    if rule == "first_trading_day_after_monthly_expiry":
        return post_expiry_entry_days(prices)
    raise ValueError(f"unknown selection.entry_rule {rule!r}")


def delisted_before(
    delistings: pd.DataFrame,
    permno: int,
    before,
) -> bool:
    """Did `permno` delist strictly before `before`? Drives the Q7 position exclusion."""
    if not len(delistings):
        return False
    hits = delistings.loc[
        (delistings["permno"] == int(permno))
        & (delistings["dlstdt"] < pd.Timestamp(before))
    ]
    return bool(len(hits))


# ---------------------------------------------------------------------------
# Checkpoint 2a
# ---------------------------------------------------------------------------

def split_jump(prices: pd.DataFrame, permno: int, split_date) -> dict[str, float]:
    """Overnight log change in the raw and adjusted series across a known split date.

    The raw series must jump by roughly -log(ratio); the adjusted series must not jump at
    all. Reporting both is what makes this a test rather than an assertion about a number
    nobody looked at.
    """
    panel = prices.loc[prices["permno"] == int(permno)].sort_values("date")
    split_date = pd.Timestamp(split_date)
    after = panel.loc[panel["date"] >= split_date].head(1)
    before = panel.loc[panel["date"] < split_date].tail(1)
    if before.empty or after.empty:
        return {}
    b, a = before.iloc[0], after.iloc[0]
    out = {
        "prev_date": b["date"],
        "split_date": a["date"],
        "raw_before": float(b["prc_abs"]),
        "raw_after": float(a["prc_abs"]),
        "adj_before": float(b["s_adj"]),
        "adj_after": float(a["s_adj"]),
        "raw_log_change": float(np.log(a["prc_abs"] / b["prc_abs"])),
        "adj_log_change": float(np.log(a["s_adj"] / b["s_adj"])),
    }
    if "ret" in panel.columns and pd.notna(a["ret"]):
        out["crsp_ret"] = float(a["ret"])
    return out


def return_reconciliation(prices: pd.DataFrame, permno: int, year: int) -> dict[str, float]:
    """Cross-check the adjusted price series against CRSP's own return fields.

    `retx` is the return *excluding* dividends, so it should reproduce the adjusted price
    return almost exactly -- a far sharper test than comparing against `ret`, which differs
    by the dividend contribution. The `ret`-vs-`retx` gap is reported alongside as the
    implied dividend yield, which should look like a plausible annual figure for the name.
    """
    panel = prices.loc[
        (prices["permno"] == int(permno)) & (prices["date"].dt.year == int(year))
    ].sort_values("date")
    panel = panel.loc[panel["has_price"]]
    if len(panel) < 2 or "retx" not in panel.columns:
        return {}

    price_growth = float(panel["s_adj"].iloc[-1] / panel["s_adj"].iloc[0])
    # Compound from the second row: the first row's return belongs to the prior year.
    retx_growth = float((1.0 + panel["retx"].iloc[1:].astype(float)).prod())
    ret_growth = float((1.0 + panel["ret"].iloc[1:].astype(float)).prod())
    return {
        "n_days": int(len(panel)),
        "price_growth": price_growth,
        "retx_growth": retx_growth,
        "ret_growth": ret_growth,
        "price_vs_retx_rel_error": abs(price_growth - retx_growth) / retx_growth,
        "implied_div_yield": ret_growth / retx_growth - 1.0,
    }


def checkpoint_2a(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    link: pd.DataFrame,
    linked: pd.DataFrame,
    delistings: pd.DataFrame,
    cfg: Config | None = None,
    *,
    activity: pd.Series | None = None,
) -> list[tuple[str, bool, str]]:
    """Checkpoint 2a from PHASE1_PLAN.md, as an explicit pass/fail list."""
    from .linking import link_conflicts, link_coverage

    cfg = cfg or load_config()
    sanity = cfg["sanity"]
    results: list[tuple[str, bool, str]] = []

    # --- 1. no duplicate (permno, date) rows -------------------------------
    dupes = int(prices.duplicated(subset=["permno", "date"]).sum())
    results.append((
        "no duplicate (permno, date) price rows",
        dupes == 0,
        f"{dupes} duplicates over {len(prices):,} rows",
    ))

    # --- 2. split test -----------------------------------------------------
    detail, ok_all = [], True
    for spec in sanity.get("splits", []):
        jump = split_jump(prices, spec["permno"], spec["date"])
        if not jump:
            ok_all = False
            detail.append(f"{spec['ticker']}: no data around {spec['date']}")
            continue
        expected_raw = -np.log(float(spec["ratio"]))
        raw_ok = abs(jump["raw_log_change"] - expected_raw) < 0.15
        adj_ok = abs(jump["adj_log_change"]) < 0.15
        ok_all &= raw_ok and adj_ok
        detail.append(
            f"{spec['ticker']} {spec['date']}: raw {jump['raw_before']:.2f}->{jump['raw_after']:.2f} "
            f"(logchg {jump['raw_log_change']:+.3f}), adj {jump['adj_before']:.2f}->"
            f"{jump['adj_after']:.2f} (logchg {jump['adj_log_change']:+.3f})"
        )
    results.append((
        "known splits show no jump in the adjusted series",
        ok_all and bool(sanity.get("splits")),
        "; ".join(detail) if detail else "no split fixtures configured",
    ))

    # --- 3. return reconciliation -----------------------------------------
    recon = return_reconciliation(
        prices, int(sanity["recon_permno"]), int(sanity["recon_year"])
    )
    if not recon:
        results.append(("adjusted price return reconciles with CRSP retx", False,
                        "insufficient data for the reconciliation fixture"))
    else:
        ok = recon["price_vs_retx_rel_error"] < 1e-3
        results.append((
            "adjusted price return reconciles with CRSP retx",
            ok,
            f"price {recon['price_growth']:.4f} vs retx {recon['retx_growth']:.4f} "
            f"(rel err {recon['price_vs_retx_rel_error']:.2e}); "
            f"implied dividend yield {recon['implied_div_yield']:.2%}",
        ))

    # --- 4. link coverage --------------------------------------------------
    cov = link_coverage(universe, linked)
    floor = float(sanity.get("link_coverage_min", 0.98))
    results.append((
        f"link coverage of universe (permno, month) cells > {floor:.0%}",
        cov["coverage"] > floor,
        f"{cov['coverage']:.2%} of {cov['n_cells']:,} cells; "
        f"{cov['n_permnos_missing']} of {cov['n_permnos']} permnos unlinked"
        + (f" e.g. {cov['missing_permnos'][:5]}" if cov["missing_permnos"] else ""),
    ))

    # --- 5. concurrent secids are resolved to a LIVE chain -----------------
    # The plan allows either zero conflicts or a documented tie-break. There are conflicts
    # (14 permnos carry a dormant second secid), so what has to hold is that the tie-break
    # always lands on the secid that actually has option data -- otherwise a name silently
    # gets no tradable contracts for the entire sample.
    panel = prices[["permno", "date"]].drop_duplicates()
    conflicts = link_conflicts(panel, link)
    if activity is None:
        results.append((
            "every ambiguous link resolves to a secid with option data",
            False,
            f"{len(conflicts):,} conflicted cells but no activity data supplied to adjudicate",
        ))
    else:
        act = pd.Series(activity, dtype="float64")
        ambiguous = set(conflicts["permno"].unique())
        chosen = linked.loc[linked["permno"].isin(ambiguous) & linked["secid"].notna()]
        chosen = chosen[["permno", "secid"]].drop_duplicates()
        chosen["n_rows"] = chosen["secid"].map(act).fillna(0.0)
        dead = chosen.loc[chosen["n_rows"] <= 0]
        results.append((
            "every ambiguous link resolves to a secid with option data",
            dead.empty,
            f"{len(conflicts):,} conflicted cells across {len(ambiguous)} permnos; "
            + ("all resolved to live chains" if dead.empty
               else f"{len(dead)} resolved to an EMPTY secid: "
                    f"{dead[['permno', 'secid']].to_dict('records')[:5]}"),
        ))

    # --- 6. delistings are a countable exclusion, not a surprise -----------
    n_delist = int(len(delistings))
    in_universe = (
        int(delistings["permno"].isin(set(universe["permno"])).sum()) if n_delist else 0
    )
    results.append((
        "delisting events counted for the Q7 exclusion",
        True,  # informational: a count, not a threshold
        f"{n_delist} delisting rows in window, {in_universe} on universe names",
    ))

    return results
