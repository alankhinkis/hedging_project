"""Stage 1 -- point-in-time S&P 500 universe, ranked to the top N by market cap.

The whole survivorship-bias fix lives in one line: a name is a member in month *m* only if
*m* falls inside one of its `crsp.dsp500list` membership spans. Nothing is carried forward,
nothing is back-filled from today's index composition.

Two subtleties that are easy to get wrong and both matter:

**Lagged ranking.** Positions are opened on the first trading day of a month, so the ranking
that selects them must be knowable *before* that day. We rank on month-end *m* and use that
ranking for entries in month *m+1*. `rank_month` is the data month; `entry_month` is the
month it governs. Ranking month *m* and trading month *m* would be look-ahead bias -- small
in effect, indefensible in an interview.

**Multi-class names.** The S&P 500 carries slightly more than 500 permnos (GOOG/GOOGL,
BRK.B, FOX/FOXA...), so a member count of 495-510 is correct and 500 exactly would be
suspicious. Ranking is per permno; no share-class consolidation is attempted, and the
top-N cut is applied after ranking so a dual-class name can occupy two slots. That is the
honest reading of "top 150 names by market cap" when the tradable unit is the permno.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..schema import table_for
from ..wrds_conn import cached_query, get_connection, resolve_columns, split_table

log = logging.getLogger(__name__)

# Column-name candidates, in preference order. CRSP vintages differ (the v2 tables rename
# almost everything), so every pull resolves its columns against the live table first.
_MEMBERSHIP_COLS = {
    "permno": ["permno"],
    "start": ["start", "mbrstartdt", "mbrstart", "begdt"],
    "end": ["ending", "mbrenddt", "mbrend", "enddt"],
}
_MSF_COLS = {
    "permno": ["permno"],
    "date": ["date", "mthcaldt", "caldt"],
    "prc": ["prc", "mthprc"],
    "shrout": ["shrout", "shrout_adj"],
}
_NAMES_COLS = {
    "permno": ["permno"],
    "namedt": ["namedt", "namestartdt"],
    "nameendt": ["nameendt", "nameenddt"],
    "ticker": ["ticker", "htick"],
    "comnam": ["comnam", "conm"],
}


def _resolve_columns(table: str, wanted: dict[str, list[str]], conn) -> dict[str, str]:
    """Thin wrapper kept for readability at the call sites in this module."""
    return resolve_columns(table, wanted, conn=conn)


# ---------------------------------------------------------------------------
# Pulls
# ---------------------------------------------------------------------------

def fetch_membership(cfg: Config | None = None, *, conn=None, force: bool = False) -> pd.DataFrame:
    """S&P 500 membership spans overlapping [burnin_start, end_date].

    Returns permno / from_date / thru_date. Open-ended memberships come back from CRSP with
    a far-future `ending`; they are left as-is and clipped at the sample end downstream.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    table = table_for("S&P 500 point-in-time membership", cfg)
    cols = _resolve_columns(table, _MEMBERSHIP_COLS, conn)
    schema, tbl = split_table(table)

    sql = f"""
        SELECT {cols['permno']} AS permno,
               {cols['start']}  AS from_date,
               {cols['end']}    AS thru_date
        FROM   {schema}.{tbl}
        WHERE  {cols['end']}   >= %(burnin)s
          AND  {cols['start']} <= %(end)s
    """
    df = cached_query(
        sql,
        {"burnin": str(cfg.burnin_start), "end": str(cfg.end_date)},
        name="sp500_membership",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("from_date", "thru_date"),
    )
    df["permno"] = df["permno"].astype("int64")
    return df.sort_values(["permno", "from_date"]).reset_index(drop=True)


def fetch_monthly_mktcap(
    permnos: Sequence[int],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
) -> pd.DataFrame:
    """Month-end market cap for the given permnos over the burn-in..sample window.

    `msf` is used rather than `dsf` because it is ~20x smaller and month-end is the only
    date we rank on. `shrout` is in thousands, so `|prc| * shrout` is market cap in $000s.
    A negative `prc` is CRSP's flag for a bid-ask average rather than a closing trade; the
    magnitude is still the right price, so we take `ABS(prc)` and keep the observation.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    table = table_for("Monthly market cap", cfg)
    cols = _resolve_columns(table, _MSF_COLS, conn)
    schema, tbl = split_table(table)

    sql = f"""
        SELECT {cols['permno']} AS permno,
               {cols['date']}   AS date,
               ABS({cols['prc']}) * {cols['shrout']} AS mktcap_k,
               {cols['prc']}    AS prc,
               {cols['shrout']} AS shrout
        FROM   {schema}.{tbl}
        WHERE  {cols['permno']} IN %(permnos)s
          AND  {cols['date']} BETWEEN %(a)s AND %(b)s
          AND  {cols['prc']} IS NOT NULL
          AND  {cols['shrout']} IS NOT NULL
    """
    df = cached_query(
        sql,
        {
            "permnos": tuple(int(p) for p in sorted(set(permnos))),
            "a": str(cfg.burnin_start),
            "b": str(cfg.end_date),
        },
        name="msf_mktcap",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("date",),
    )
    df["permno"] = df["permno"].astype("int64")
    return df


def fetch_names(
    permnos: Sequence[int],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
) -> pd.DataFrame:
    """Date-ranged ticker/company-name history, used only for human-readable checks."""
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    table = table_for("CRSP name history", cfg)
    cols = _resolve_columns(table, _NAMES_COLS, conn)
    schema, tbl = split_table(table)

    sql = f"""
        SELECT {cols['permno']}   AS permno,
               {cols['namedt']}   AS namedt,
               {cols['nameendt']} AS nameendt,
               {cols['ticker']}   AS ticker,
               {cols['comnam']}   AS comnam
        FROM   {schema}.{tbl}
        WHERE  {cols['permno']} IN %(permnos)s
    """
    df = cached_query(
        sql,
        {"permnos": tuple(int(p) for p in sorted(set(permnos)))},
        name="crsp_names",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("namedt", "nameendt"),
    )
    df["permno"] = df["permno"].astype("int64")
    return df


# ---------------------------------------------------------------------------
# Point-in-time construction (pure pandas -- unit-testable without WRDS)
# ---------------------------------------------------------------------------

def is_member(membership: pd.DataFrame, permno: int, when) -> bool:
    """Was `permno` in the index on `when`? The one-line survivorship test."""
    when = pd.Timestamp(when)
    spans = membership.loc[membership["permno"] == int(permno)]
    return bool(((spans["from_date"] <= when) & (spans["thru_date"] >= when)).any())


def members_by_month(membership: pd.DataFrame, mktcap: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time join of month-end market caps onto membership spans.

    This is the survivorship-bias fix. A month-end observation survives only if that date
    falls inside a membership span for that permno.
    """
    merged = mktcap.merge(membership, on="permno", how="inner")
    inside = (merged["date"] >= merged["from_date"]) & (merged["date"] <= merged["thru_date"])
    merged = merged.loc[inside].copy()

    # A permno can have several (re-entry) spans; the join can therefore duplicate a
    # month-end. Collapse back to one row per (permno, month-end).
    merged = merged.drop_duplicates(subset=["permno", "date"], keep="first")
    merged["rank_month"] = merged["date"].dt.to_period("M")
    return merged[["permno", "date", "rank_month", "mktcap_k", "prc", "shrout"]].reset_index(drop=True)


def build_universe(
    membership: pd.DataFrame,
    mktcap: pd.DataFrame,
    *,
    top_n: int = 150,
) -> pd.DataFrame:
    """Top-`top_n` members by month-end market cap, with the entry month they govern.

    Output columns: rank_month (data month), entry_month (rank_month + 1, the month whose
    first trading day the position is opened on), permno, rank, mktcap_k.
    """
    members = members_by_month(membership, mktcap)
    members = members.sort_values(["rank_month", "mktcap_k"], ascending=[True, False])
    members["rank"] = members.groupby("rank_month")["mktcap_k"].rank(
        method="first", ascending=False
    ).astype(int)

    universe = members.loc[members["rank"] <= top_n].copy()
    universe["entry_month"] = universe["rank_month"] + 1
    return (
        universe[["rank_month", "entry_month", "permno", "rank", "mktcap_k", "date"]]
        .rename(columns={"date": "rank_date"})
        .sort_values(["entry_month", "rank"])
        .reset_index(drop=True)
    )


def member_counts(membership: pd.DataFrame, mktcap: pd.DataFrame) -> pd.DataFrame:
    """Members per month-end -- the join-sanity diagnostic (expect 495-510)."""
    members = members_by_month(membership, mktcap)
    return (
        members.groupby("rank_month").size().rename("n_members").reset_index()
    )


def attach_tickers(universe: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time ticker for each (permno, rank_date), for readable diagnostics."""
    merged = universe.merge(names, on="permno", how="left")
    inside = (merged["namedt"] <= merged["rank_date"]) & (
        merged["nameendt"].isna() | (merged["nameendt"] >= merged["rank_date"])
    )
    merged = merged.loc[inside].drop_duplicates(subset=["rank_month", "permno"], keep="last")
    keep = list(universe.columns) + ["ticker", "comnam"]
    out = universe.merge(
        merged[["rank_month", "permno", "ticker", "comnam"]],
        on=["rank_month", "permno"],
        how="left",
    )
    return out[keep]


# ---------------------------------------------------------------------------
# Checkpoint 1
# ---------------------------------------------------------------------------

def _fmt(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def checkpoint_1(
    membership: pd.DataFrame,
    mktcap: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: Config | None = None,
    *,
    names: pd.DataFrame | None = None,
) -> list[tuple[str, bool, str]]:
    """Checkpoint 1 from PHASE1_PLAN.md, as an explicit pass/fail list."""
    cfg = cfg or load_config()
    u_cfg = cfg["universe"]
    sanity = cfg["sanity"]
    results: list[tuple[str, bool, str]] = []

    # --- 1. member count per month ----------------------------------------
    counts = member_counts(membership, mktcap)
    in_sample = counts.loc[counts["rank_month"] >= pd.Period(cfg.start_date, "M")]
    lo, hi = int(u_cfg["expected_members_min"]), int(u_cfg["expected_members_max"])
    bad = in_sample.loc[(in_sample["n_members"] < lo) | (in_sample["n_members"] > hi)]
    results.append((
        f"member count per month in [{lo}, {hi}]",
        bad.empty,
        f"min={in_sample['n_members'].min()}, max={in_sample['n_members'].max()}"
        + ("" if bad.empty else f"; {len(bad)} bad months e.g. {bad.iloc[0].to_dict()}"),
    ))

    # --- 2. TSLA survivorship spot-check ----------------------------------
    tsla = int(sanity["tsla_permno"])
    add_date = pd.Timestamp(sanity["tsla_index_add_date"])
    before = is_member(membership, tsla, add_date - pd.Timedelta(days=30))
    after = is_member(membership, tsla, add_date + pd.Timedelta(days=30))
    results.append((
        "TSLA absent before 2020-12-21 index add, present after",
        (not before) and after,
        f"member 30d before={before}, 30d after={after}",
    ))

    # --- 3. a name that exited mid-sample must vanish ----------------------
    # Picked from the membership table itself rather than hard-coded, so the test cannot
    # rot: the largest name whose span ends strictly inside the sample.
    exits = membership.loc[
        (membership["thru_date"] >= pd.Timestamp(cfg.start_date))
        & (membership["thru_date"] <= pd.Timestamp(cfg.end_date) - pd.Timedelta(days=180))
    ]
    if exits.empty:
        results.append(("a mid-sample deletion disappears from the universe", False,
                        "no membership span ends inside the sample -- suspicious"))
    else:
        # choose the exiting name with the largest market cap at its exit
        exit_caps = []
        for _, row in exits.iterrows():
            cap = mktcap.loc[
                (mktcap["permno"] == row["permno"])
                & (mktcap["date"] <= row["thru_date"])
            ]["mktcap_k"]
            if len(cap):
                exit_caps.append((cap.iloc[-1], int(row["permno"]), row["thru_date"]))
        exit_caps.sort(key=lambda t: t[0], reverse=True)
        if not exit_caps:
            results.append((
                "a mid-sample deletion disappears from the universe",
                False,
                f"{len(exits)} spans end inside the sample but none has market-cap data -- "
                "the mktcap pull does not cover the deleted names",
            ))
        else:
            _, ex_permno, ex_date = exit_caps[0]
            after_exit = universe.loc[
                (universe["permno"] == ex_permno)
                & (universe["rank_month"] > pd.Period(ex_date, "M"))
            ]
            results.append((
                "a mid-sample deletion disappears from the universe",
                after_exit.empty,
                f"permno {ex_permno} exited {ex_date.date()}; "
                f"{len(after_exit)} universe rows after that month",
            ))

    # --- 4. ranking is genuinely time-varying -----------------------------
    expected = sanity.get("expected_top_members") or {}
    if names is None:
        results.append((
            "expected large caps present in their months",
            False,
            "ticker history not supplied; cannot check",
        ))
    else:
        tagged = attach_tickers(universe, names)
        misses: list[str] = []
        for month, tickers in expected.items():
            present = set(
                tagged.loc[tagged["entry_month"] == pd.Period(month, "M"), "ticker"].dropna()
            )
            missing = [t for t in tickers if t not in present]
            if missing:
                misses.append(f"{month}: missing {missing}")
        results.append((
            "expected large caps present in their months (XOM/GE in 2017, AAPL/MSFT/AMZN in 2020)",
            not misses,
            "; ".join(misses) if misses else "all present",
        ))

    # --- 5. structural integrity of the ranking ---------------------------
    dupes = universe.duplicated(subset=["rank_month", "permno"]).sum()
    results.append((
        "no permno appears twice within a month",
        dupes == 0,
        f"{dupes} duplicate (month, permno) rows",
    ))

    top_n = int(u_cfg["top_n"])
    sizes = universe.groupby("rank_month").size()
    full_months = sizes.loc[sizes.index >= pd.Period(cfg.start_date, "M")]
    ranks_ok = True
    detail = ""
    for month, grp in universe.groupby("rank_month"):
        expected_ranks = set(range(1, min(top_n, len(grp)) + 1))
        if set(grp["rank"]) != expected_ranks:
            ranks_ok = False
            detail = f"{month} has ranks {sorted(grp['rank'])[:5]}..."
            break
    results.append((
        f"ranks are 1..{top_n} with no gaps in every month",
        ranks_ok,
        detail or f"all months have contiguous ranks (sizes {full_months.min()}-{full_months.max()})",
    ))

    # --- 6. turnover: distinct permnos must exceed top_n -------------------
    distinct = universe["permno"].nunique()
    floor = int(u_cfg["expected_distinct_permnos_min"])
    results.append((
        f"distinct permnos across the sample > {floor} (a static ~{top_n} means survivorship bias)",
        distinct > floor,
        f"{distinct} distinct permnos",
    ))

    return results


def universe_summary(universe: pd.DataFrame) -> pd.DataFrame:
    """Per-year turnover summary -- names, entries, exits. Coverage is a result (Q9)."""
    by_month = (
        universe.groupby("entry_month")["permno"].apply(frozenset).sort_index()
    )
    rows: list[dict[str, Any]] = []
    prev: frozenset[int] | None = None
    for month, members in by_month.items():
        rows.append(
            {
                "entry_month": str(month),
                "n": len(members),
                "entries": np.nan if prev is None else len(members - prev),
                "exits": np.nan if prev is None else len(prev - members),
            }
        )
        prev = members
    return pd.DataFrame(rows)
