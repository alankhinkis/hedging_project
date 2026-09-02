"""Stage 1 unit tests.

The point-in-time join and the ranking are pure pandas, so they are tested here on
synthetic fixtures with no WRDS connection. The survivorship-bias fix is the single most
consequential line in the data build; it gets a test that fails loudly if it regresses.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vrp.data.universe import (
    attach_tickers,
    build_universe,
    is_member,
    member_counts,
    members_by_month,
    universe_summary,
)


def month_ends(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="ME")


@pytest.fixture
def membership() -> pd.DataFrame:
    """Four names:

    A -- member throughout.
    B -- member throughout, smaller.
    C -- joins mid-sample (the TSLA analogue).
    D -- leaves mid-sample (the deletion analogue).
    """
    return pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "from_date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-07-15", "2019-01-01"]),
            "thru_date": pd.to_datetime(["2020-12-31", "2020-12-31", "2020-12-31", "2019-05-20"]),
        }
    )


@pytest.fixture
def mktcap() -> pd.DataFrame:
    """Market caps that make the ranking time-varying: A shrinks past B halfway through."""
    dates = month_ends("2019-01-31", 12)
    rows = []
    for i, d in enumerate(dates):
        rows += [
            {"permno": 1, "date": d, "mktcap_k": 1000 - 40 * i, "prc": 10.0, "shrout": 100},
            {"permno": 2, "date": d, "mktcap_k": 500 + 40 * i, "prc": 5.0, "shrout": 100},
            {"permno": 3, "date": d, "mktcap_k": 900, "prc": 9.0, "shrout": 100},
            {"permno": 4, "date": d, "mktcap_k": 800, "prc": 8.0, "shrout": 100},
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The survivorship fix
# ---------------------------------------------------------------------------

def test_late_joiner_absent_before_its_add_date(membership, mktcap):
    """C has market-cap data all year but joins the index on 2019-07-15.

    Before that date it must not appear, even though it would rank second by size. This is
    the synthetic version of the TSLA spot-check in Checkpoint 1.
    """
    members = members_by_month(membership, mktcap)
    early = members.loc[members["rank_month"] < pd.Period("2019-07", "M")]
    assert 3 not in set(early["permno"])
    late = members.loc[members["rank_month"] >= pd.Period("2019-08", "M")]
    assert 3 in set(late["permno"])


def test_deleted_name_vanishes_after_its_exit(membership, mktcap):
    members = members_by_month(membership, mktcap)
    after = members.loc[
        (members["permno"] == 4) & (members["rank_month"] > pd.Period("2019-05", "M"))
    ]
    assert after.empty


def test_is_member_boundaries(membership):
    assert is_member(membership, 3, "2019-07-15")      # inclusive start
    assert not is_member(membership, 3, "2019-07-14")
    assert is_member(membership, 4, "2019-05-20")      # inclusive end
    assert not is_member(membership, 4, "2019-05-21")


def test_reentry_spans_do_not_duplicate_a_month():
    """A name that leaves and rejoins has two spans; a month must still yield one row."""
    membership = pd.DataFrame(
        {
            "permno": [1, 1],
            "from_date": pd.to_datetime(["2019-01-01", "2019-06-01"]),
            "thru_date": pd.to_datetime(["2019-03-31", "2019-12-31"]),
        }
    )
    mktcap = pd.DataFrame(
        {
            "permno": 1,
            "date": month_ends("2019-01-31", 12),
            "mktcap_k": 100.0,
            "prc": 1.0,
            "shrout": 100,
        }
    )
    members = members_by_month(membership, mktcap)
    assert not members.duplicated(subset=["permno", "date"]).any()
    # Jan-Mar plus Jun-Dec = 10 months; Apr and May are out of both spans.
    assert len(members) == 10
    assert pd.Period("2019-04", "M") not in set(members["rank_month"])


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_ranking_is_time_varying(membership, mktcap):
    """A starts largest and B overtakes it; the top-1 universe must switch names."""
    top1 = build_universe(membership, mktcap, top_n=1)
    first = top1.loc[top1["rank_month"] == pd.Period("2019-01", "M"), "permno"].iloc[0]
    last = top1.loc[top1["rank_month"] == pd.Period("2019-12", "M"), "permno"].iloc[0]
    assert first == 1
    assert last == 2


def test_entry_month_lags_rank_month_by_one(membership, mktcap):
    """Positions opened on the first trading day of month m are ranked on month-end m-1.

    Ranking and trading in the same month would be look-ahead bias.
    """
    universe = build_universe(membership, mktcap, top_n=2)
    lag = {(e - r).n for e, r in zip(universe["entry_month"], universe["rank_month"])}
    assert lag == {1}


def test_ranks_are_contiguous_and_unique(membership, mktcap):
    universe = build_universe(membership, mktcap, top_n=3)
    for _, grp in universe.groupby("rank_month"):
        assert sorted(grp["rank"]) == list(range(1, len(grp) + 1))
        assert grp["permno"].is_unique


def test_top_n_truncates(membership, mktcap):
    universe = build_universe(membership, mktcap, top_n=2)
    assert universe.groupby("rank_month").size().max() == 2


def test_ties_do_not_produce_duplicate_ranks():
    """Equal market caps must still yield distinct ranks (method='first'), or the
    'ranks are 1..N with no gaps' checkpoint would fail on a legitimate tie."""
    membership = pd.DataFrame(
        {"permno": [1, 2], "from_date": pd.to_datetime(["2019-01-01"] * 2),
         "thru_date": pd.to_datetime(["2019-12-31"] * 2)}
    )
    mktcap = pd.DataFrame(
        {
            "permno": [1, 2],
            "date": [pd.Timestamp("2019-01-31")] * 2,
            "mktcap_k": [100.0, 100.0],
            "prc": [1.0, 1.0],
            "shrout": [100, 100],
        }
    )
    universe = build_universe(membership, mktcap, top_n=2)
    assert sorted(universe["rank"]) == [1, 2]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_member_counts_match_the_join(membership, mktcap):
    counts = member_counts(membership, mktcap)
    jan = counts.loc[counts["rank_month"] == pd.Period("2019-01", "M"), "n_members"].iloc[0]
    dec = counts.loc[counts["rank_month"] == pd.Period("2019-12", "M"), "n_members"].iloc[0]
    assert jan == 3   # A, B, D  (C has not joined)
    assert dec == 3   # A, B, C  (D has left)


def test_universe_summary_reports_entries_and_exits(membership, mktcap):
    universe = build_universe(membership, mktcap, top_n=10)
    summary = universe_summary(universe)
    june = summary.loc[summary["entry_month"] == "2019-06"].iloc[0]
    assert june["exits"] == 1        # D left after its 2019-05-20 deletion
    august = summary.loc[summary["entry_month"] == "2019-09"].iloc[0]
    assert august["entries"] == 0    # C already entered in the 2019-08 ranking month


def test_attach_tickers_is_point_in_time(membership, mktcap):
    """A permno that was renamed mid-sample must pick up the ticker of the ranking date."""
    universe = build_universe(membership, mktcap, top_n=10)
    names = pd.DataFrame(
        {
            "permno": [1, 1],
            "namedt": pd.to_datetime(["2015-01-01", "2019-07-01"]),
            "nameendt": pd.to_datetime(["2019-06-30", "2030-12-31"]),
            "ticker": ["OLD", "NEW"],
            "comnam": ["Old Co", "New Co"],
        }
    )
    tagged = attach_tickers(universe, names)
    one = tagged.loc[tagged["permno"] == 1].set_index("rank_month")["ticker"]
    assert one[pd.Period("2019-03", "M")] == "OLD"
    assert one[pd.Period("2019-09", "M")] == "NEW"
    # Names the lookup does not cover come back as NaN rather than silently dropping the row.
    assert tagged["permno"].tolist() == universe["permno"].tolist()
    assert tagged.loc[tagged["permno"] == 2, "ticker"].isna().all()
