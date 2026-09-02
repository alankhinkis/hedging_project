"""Stage 2a -- the OptionMetrics <-> CRSP link.

`wrdsapps.opcrsphist` maps secid to permno over date ranges. Two things about it matter,
both established by the Stage 0 probe rather than assumed:

**Score 6 is not a weak link, it is the absence of one.** The distribution is:

    score 1: 28,336 rows, 27,997 distinct permno   <- best match
    score 2:    190
    score 3:      8
    score 4:    660
    score 5:  5,687
    score 6: 86,892 rows, 0 distinct permno        <- permno is NULL

Score 6 is the largest bucket by far and carries no permno at all. Filtering on `score`
without excluding it looks like a working join and returns nothing, so
`fetch_link` requires `score <= max_score AND permno IS NOT NULL` at the server.

**The join is date-ranged, exactly like the membership join.** A secid maps to a permno
only for the interval [sdate, edate]. Merging on permno alone would attach the wrong secid
across a re-issue or share-class change.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from ..config import Config, load_config
from ..schema import opprcd_table_for_year, table_for
from ..wrds_conn import cached_query, get_connection, split_table

log = logging.getLogger(__name__)

# Score 6 means "no CRSP match". Everything at or below MAX_USABLE_SCORE is a real link;
# score 1 alone covers 99.3% of matched rows, so the choice is barely consequential -- but
# it is a choice, and it is made here rather than being implicit in a WHERE clause.
MAX_USABLE_SCORE = 5
NO_MATCH_SCORE = 6


def fetch_link(
    permnos: Sequence[int],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
    max_score: int = MAX_USABLE_SCORE,
) -> pd.DataFrame:
    """Link rows for the given permnos, excluding the no-match bucket."""
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    table = table_for("OM <-> CRSP link", cfg)
    schema, tbl = split_table(table)

    sql = f"""
        SELECT secid, permno, score, sdate, edate
        FROM   {schema}.{tbl}
        WHERE  permno IN %(permnos)s
          AND  permno IS NOT NULL
          AND  score <= %(max_score)s
          AND  edate >= %(a)s
          AND  sdate <= %(b)s
    """
    df = cached_query(
        sql,
        {
            "permnos": tuple(int(p) for p in sorted(set(permnos))),
            "max_score": int(max_score),
            "a": str(cfg.burnin_start),
            "b": str(cfg.hold_buffer_end),
        },
        name="opcrsphist",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("sdate", "edate"),
    )
    if len(df):
        df["permno"] = df["permno"].astype("int64")
        df["secid"] = df["secid"].astype("int64")
        df["score"] = df["score"].astype("int64")
    return df.sort_values(["permno", "sdate"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Applying the link (pure pandas -- unit-testable without WRDS)
# ---------------------------------------------------------------------------

def ambiguous_permnos(link: pd.DataFrame) -> list[int]:
    """Permnos carrying more than one secid -- the only ones a tie-break can matter for."""
    if not len(link):
        return []
    counts = link.groupby("permno")["secid"].nunique()
    return sorted(int(p) for p in counts[counts > 1].index)


def secid_activity(
    secids: Sequence[int],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
    years: Sequence[int] | None = None,
) -> pd.Series:
    """Rows of option data per secid across the sample years, as a liveness measure.

    Used to break link ties. When a permno carries two secids, one of them is almost always
    a dormant shell -- a legacy identifier or a corporate-action series that never traded --
    and the distinction is invisible in the link table itself. Counting actual option rows
    settles it from data.

    Returns a Series secid -> row count (zero for secids with no rows anywhere).
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    secids = tuple(int(s) for s in sorted(set(secids)))
    if not secids:
        return pd.Series(dtype="int64", name="n_rows")

    years = list(years or range(cfg.start_date.year, cfg.end_date.year + 1))
    parts = []
    for year in years:
        schema, tbl = split_table(opprcd_table_for_year(year, cfg))
        parts.append(f"SELECT secid, COUNT(*) AS n FROM {schema}.{tbl} "
                     f"WHERE secid IN %(secids)s GROUP BY secid")
    sql = " UNION ALL ".join(parts)

    df = cached_query(
        sql,
        {"secids": secids},
        name="secid_activity",
        cfg=cfg,
        conn=conn,
        force=force,
    )
    if not len(df):
        return pd.Series(0, index=list(secids), dtype="int64", name="n_rows")
    totals = df.groupby("secid")["n"].sum().astype("int64")
    totals.index = totals.index.astype("int64")
    return totals.reindex(list(secids), fill_value=0).rename("n_rows")


def apply_link(
    panel: pd.DataFrame,
    link: pd.DataFrame,
    *,
    date_col: str = "date",
    how: str = "left",
    activity: pd.Series | dict[int, int] | None = None,
) -> pd.DataFrame:
    """Attach `secid` to a (permno, date) panel via a date-ranged join.

    Tie-break, in order: lowest `score`, then **most option activity**, then the longer
    overlapping span, then the smaller secid.

    Activity outranks span deliberately. USB (permno 66157) carries two score-1 secids,
    111306 (spanning from 1996, and completely empty) and 104866 (161,208 option rows in
    2019 alone). A span-based tie-break picks the dead one and would silently give that
    name no tradable contracts for the whole sample. Score alone does not separate them;
    only the data does.
    """
    if not len(link) or "sdate" not in link.columns:
        out = panel.copy()
        out["secid"] = pd.NA
        return out

    # Resolve the best link per (permno, date) on the *matched* rows only...
    candidates = panel.merge(link, on="permno", how="inner")
    inside = (candidates[date_col] >= candidates["sdate"]) & (
        candidates[date_col] <= candidates["edate"]
    )
    candidates = candidates.loc[inside].copy()
    candidates["_span"] = (candidates["edate"] - candidates["sdate"]).dt.days
    if activity is None:
        candidates["_activity"] = 0
    else:
        act = pd.Series(activity, dtype="float64")
        candidates["_activity"] = candidates["secid"].map(act).fillna(0.0)
    best = (
        candidates.sort_values(
            ["permno", date_col, "score", "_activity", "_span", "secid"],
            ascending=[True, True, True, False, False, True],
        )
        .drop_duplicates(subset=["permno", date_col], keep="first")
        .drop(columns=["_span", "_activity"])
    )

    # ...then join back onto the panel, so a date that falls outside every span keeps its
    # row with a null secid instead of vanishing. Silent row loss here would delete part of
    # the cross-section and show up only as an unexplained coverage gap much later.
    out = panel.merge(best, on=["permno", date_col], how=how)
    assert len(out) == len(panel) or how != "left", "apply_link must not change the row count"
    return out.reset_index(drop=True)


def link_conflicts(panel: pd.DataFrame, link: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """(permno, date) cells that match two or more *distinct* secids simultaneously.

    Checkpoint 2a wants this to be empty, or the tie-break rule documented. Distinctness is
    on secid, not on link rows: the same secid appearing in two adjacent spans is not a
    conflict.
    """
    merged = panel.merge(link, on="permno", how="inner")
    inside = (merged[date_col] >= merged["sdate"]) & (merged[date_col] <= merged["edate"])
    merged = merged.loc[inside]
    counts = (
        merged.groupby(["permno", date_col])["secid"].nunique().rename("n_secid").reset_index()
    )
    return counts.loc[counts["n_secid"] > 1]


def link_coverage(universe: pd.DataFrame, linked: pd.DataFrame) -> dict[str, object]:
    """What fraction of universe (permno, entry_month) cells got a secid?

    A systematic gap here silently deletes part of the cross-section, so this is reported,
    not assumed. Failures are usually share-class or ticker issues on a handful of names.
    """
    have = linked.loc[linked["secid"].notna(), ["permno"]].drop_duplicates()
    have_set = set(have["permno"].astype("int64"))
    cells = universe[["entry_month", "permno"]].drop_duplicates()
    covered = cells["permno"].isin(have_set)

    missing = sorted(set(cells.loc[~covered, "permno"].astype("int64")))
    return {
        "n_cells": int(len(cells)),
        "n_covered": int(covered.sum()),
        "coverage": float(covered.mean()) if len(cells) else 0.0,
        "n_permnos": int(cells["permno"].nunique()),
        "n_permnos_missing": len(missing),
        "missing_permnos": missing[:25],
    }
