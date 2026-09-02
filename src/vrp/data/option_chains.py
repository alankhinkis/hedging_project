"""Stage 2b -- option chains, in two passes.

**Pass A** pulls the candidate chain on each entry date. **Pass B** pulls the daily path of
the selected contracts only.

The stage is cheap because of one observation: the paper's P&L (M1) needs the option price
only *at entry*. The terminal value is intrinsic, computed from S_T and K, and the deltas are
ours, recomputed from physical volatility. So a daily option quote panel is not required for
the P&L at all. Pass B exists for diagnostics, the implied-vol placebo (Stage 4c), and
Phase 2's mark-to-market -- not for the headline number.

**What is filtered where, and why.** Only volume reducers go in the SQL:

    secid IN (...)                  the universe
    date  IN (entry dates)          ~12 dates a year instead of 250 -- the ~95% reduction
    exdate - date BETWEEN 15 AND 50 deliberately WIDER than the 20-40 rule

Every economic screen -- bid/offer, open interest, implied-vol bounds, arbitrage bounds,
`ss_flag`, `contract_size`, the delta band -- is applied in pandas, each one logging what it
dropped. The plan's Pass A query puts those in the WHERE clause, but the plan also requires
reporting coverage by year and market-cap quintile (Q9) and an IV-cap robustness row at both
100% and 300% (Q8). Neither is computable from rows that never arrived. Filtering server-side
would mean re-querying WRDS to answer questions the cache should already answer.

The DTE window is pulled wider than the selection rule for the same reason: the 20-40 rule can
then be varied without a re-pull.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from ..config import Config, load_config
from ..schema import available_year_tables, opprcd_table_for_year
from ..wrds_conn import cached_query, get_connection, split_table

log = logging.getLogger(__name__)

# Pulled on every candidate row. `exercise_style` is deliberately absent: Stage 0 verified
# IvyDB's price file does not have it (US single-name equity options are American anyway).
CANDIDATE_COLUMNS = [
    "secid", "date", "exdate", "optionid", "cp_flag",
    "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "delta", "gamma", "vega", "theta",
    "cfadj", "ss_flag", "contract_size", "am_settlement", "expiry_indicator",
    "forward_price",
]

PATH_COLUMNS = [
    "secid", "date", "exdate", "optionid", "cp_flag",
    "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "delta", "vega", "cfadj", "contract_size",
]


def build_entry_spec(
    universe: pd.DataFrame,
    calendar: pd.DataFrame,
    linked: pd.DataFrame,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """One row per (entry_month, permno): the entry date and the secid to pull.

    Joins the universe (which names, in which month) to the CRSP trading calendar (which
    date is "the first trading day of the month", Q10) to the resolved link (which secid).
    Restricted to entry months inside the sample -- the universe deliberately extends past
    both ends for burn-in.
    """
    cfg = cfg or load_config()
    uni = universe.copy()
    if not isinstance(uni["entry_month"].dtype, pd.PeriodDtype):
        uni["entry_month"] = pd.PeriodIndex(uni["entry_month"], freq="M")

    cal = calendar.copy()
    if not isinstance(cal["entry_month"].dtype, pd.PeriodDtype):
        cal["entry_month"] = pd.PeriodIndex(cal["entry_month"], freq="M")

    lo = pd.Period(cfg.start_date, "M")
    hi = pd.Period(cfg.end_date, "M")
    uni = uni.loc[(uni["entry_month"] >= lo) & (uni["entry_month"] <= hi)]

    spec = uni.merge(cal[["entry_month", "entry_date"]], on="entry_month", how="inner")
    spec = spec.merge(
        linked[["permno", "date", "secid"]].rename(columns={"date": "entry_date"}),
        on=["permno", "entry_date"],
        how="left",
    )
    spec["has_secid"] = spec["secid"].notna()
    return spec[
        ["entry_month", "entry_date", "permno", "secid", "rank", "mktcap_k", "has_secid"]
    ].sort_values(["entry_month", "rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pass A -- candidates on entry dates
# ---------------------------------------------------------------------------

def fetch_candidates_year(
    year: int,
    secids: Sequence[int],
    entry_dates: Sequence[pd.Timestamp],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
    dte_min: int | None = None,
    dte_max: int | None = None,
) -> pd.DataFrame:
    """Candidate chain for one year's entry dates. Volume reducers only -- no screens."""
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    sel = cfg["selection"]
    # Pull wider than the selection rule so the rule can move without a re-pull.
    dte_min = int(dte_min if dte_min is not None else sel["dte_min"] - 5)
    dte_max = int(dte_max if dte_max is not None else sel["dte_max"] + 10)

    schema, tbl = split_table(opprcd_table_for_year(year, cfg))
    cols = ",\n               ".join(
        c if c != "strike_price" else c for c in CANDIDATE_COLUMNS
    )
    sql = f"""
        SELECT {cols},
               strike_price / 1000.0 AS strike,
               (exdate - date)        AS dte
        FROM   {schema}.{tbl}
        WHERE  secid IN %(secids)s
          AND  date  IN %(dates)s
          AND  (exdate - date) BETWEEN %(dte_min)s AND %(dte_max)s
    """
    return cached_query(
        sql,
        {
            "secids": tuple(int(s) for s in sorted(set(secids))),
            "dates": tuple(str(pd.Timestamp(d).date()) for d in sorted(set(entry_dates))),
            "dte_min": dte_min,
            "dte_max": dte_max,
        },
        name=f"opt_candidates_{year}",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("date", "exdate"),
    )


def fetch_candidates(
    spec: pd.DataFrame,
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
) -> pd.DataFrame:
    """Pass A across every year in `spec`, one cached query per year table."""
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    frames = []
    have = spec.loc[spec["has_secid"]]
    for year, grp in have.groupby(have["entry_date"].dt.year):
        secids = sorted(grp["secid"].astype("int64").unique())
        dates = sorted(grp["entry_date"].unique())
        log.info("Pass A %s: %d secids x %d entry dates", year, len(secids), len(dates))
        df = fetch_candidates_year(int(year), secids, dates, cfg, conn=conn, force=force)
        log.info("  %s rows", f"{len(df):,}")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[*CANDIDATE_COLUMNS, "strike", "dte"])
    out = pd.concat(frames, ignore_index=True)
    for col in ("secid", "optionid"):
        out[col] = out[col].astype("int64")
    return out


# ---------------------------------------------------------------------------
# Pass B -- daily paths of the selected contracts
# ---------------------------------------------------------------------------

def fetch_paths(
    selected: pd.DataFrame,
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
) -> pd.DataFrame:
    """Daily quotes for the selected contracts, entry date through expiry.

    **The December trap.** A position entered in December expires in January of the next
    year, and IvyDB splits option prices by year -- so that cohort's path spans two tables.
    Querying only `opprcd{Y}` would return the December rows and silently drop January,
    which looks like contracts that stopped quoting rather than like a missing table. Every
    year therefore unions `opprcd{Y}` with `opprcd{Y+1}`, and the year-boundary coverage is
    asserted in Checkpoint 2b.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    if not len(selected):
        return pd.DataFrame(columns=PATH_COLUMNS)

    frames = []
    for year, grp in selected.groupby(selected["entry_date"].dt.year):
        optionids = tuple(int(o) for o in sorted(grp["optionid"].unique()))
        start = str(grp["entry_date"].min().date())
        end = str(grp["exdate"].max().date())
        spans_year_end = grp["exdate"].dt.year.max() > year

        wanted = [int(year)] + ([int(year) + 1] if spans_year_end else [])
        available = available_year_tables(
            opprcd_table_for_year(int(year), cfg), wanted, conn=conn, cfg=cfg
        )
        missing = sorted(set(wanted) - {y for y, _ in available})
        if missing:
            # Loud, because a missing next-year table silently truncates the December
            # cohort -- the exact failure the union exists to prevent.
            log.warning("Pass B %s: option price tables missing for %s", year, missing)
        parts = []
        years = [y for y, _ in available]
        for y in years:
            schema, tbl = split_table(opprcd_table_for_year(y, cfg))
            parts.append(
                f"SELECT {', '.join(PATH_COLUMNS)}, strike_price / 1000.0 AS strike "
                f"FROM {schema}.{tbl} "
                f"WHERE optionid IN %(optionids)s AND date BETWEEN %(a)s AND %(b)s"
            )
        sql = " UNION ALL ".join(parts)
        log.info(
            "Pass B %s: %d contracts, %s..%s%s",
            year, len(optionids), start, end,
            f" (unioned with {int(year) + 1})" if spans_year_end else "",
        )
        df = cached_query(
            sql,
            {"optionids": optionids, "a": start, "b": end},
            name=f"opt_paths_{year}",
            cfg=cfg,
            conn=conn,
            force=force,
            date_cols=("date", "exdate"),
        )
        log.info("  %s rows", f"{len(df):,}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    for col in ("secid", "optionid"):
        out[col] = out[col].astype("int64")
    out = out.drop_duplicates(subset=["optionid", "date"])

    # Trim each contract to its OWN life. The pull uses one date window per year for all of
    # that year's contracts, but an optionid exists in the chain long before we enter it --
    # it was listed months earlier -- so the raw result carries pre-entry quotes. Without
    # this the paths look ~60% longer than a 30-day hold should be, which is exactly what
    # Checkpoint 2b's path-length condition caught.
    lives = selected[["optionid", "entry_date", "exdate"]].drop_duplicates("optionid").copy()
    lives["optionid"] = lives["optionid"].astype("int64")
    out = out.merge(lives, on="optionid", how="inner", suffixes=("", "_sel"))
    in_life = (out["date"] >= out["entry_date"]) & (out["date"] <= out["exdate_sel"])
    out = out.loc[in_life].drop(columns=["exdate_sel"])
    return out.sort_values(["optionid", "date"]).reset_index(drop=True)
