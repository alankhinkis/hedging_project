"""OM <-> CRSP link tests.

The date-ranged join is the same shape as the membership join in Stage 1, and it fails the
same way: merging on permno alone silently attaches the wrong secid across a re-issue. The
tie-break is pinned here so it stays deterministic.
"""

from __future__ import annotations

import pandas as pd

from vrp.data.linking import apply_link, link_conflicts, link_coverage


def panel(permno: int, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"permno": permno, "date": pd.to_datetime(dates)})


def test_link_is_date_ranged_not_permno_only():
    """One permno, two secids in sequence: each date must get the secid of its own era."""
    link = pd.DataFrame(
        {
            "permno": [1, 1],
            "secid": [100, 200],
            "score": [1, 1],
            "sdate": pd.to_datetime(["2015-01-01", "2020-01-01"]),
            "edate": pd.to_datetime(["2019-12-31", "2030-12-31"]),
        }
    )
    out = apply_link(panel(1, ["2018-06-01", "2021-06-01"]), link)
    got = dict(zip(out["date"], out["secid"]))
    assert got[pd.Timestamp("2018-06-01")] == 100
    assert got[pd.Timestamp("2021-06-01")] == 200


def test_dates_outside_every_span_get_no_secid_but_keep_the_row():
    """Coverage must be measurable, so an unlinked date survives with a null secid."""
    link = pd.DataFrame(
        {
            "permno": [1],
            "secid": [100],
            "score": [1],
            "sdate": pd.to_datetime(["2020-01-01"]),
            "edate": pd.to_datetime(["2020-12-31"]),
        }
    )
    out = apply_link(panel(1, ["2019-06-01", "2020-06-01"]), link)
    assert len(out) == 2
    assert out.loc[out["date"] == pd.Timestamp("2019-06-01"), "secid"].isna().all()


def test_lowest_score_wins_on_overlapping_spans():
    link = pd.DataFrame(
        {
            "permno": [1, 1],
            "secid": [999, 111],
            "score": [5, 1],
            "sdate": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "edate": pd.to_datetime(["2020-12-31", "2020-12-31"]),
        }
    )
    out = apply_link(panel(1, ["2020-06-01"]), link)
    assert out["secid"].iloc[0] == 111


def test_activity_outranks_span_the_usb_case():
    """The real USB (permno 66157) configuration: two score-1 secids, one of them dead.

    111306 spans from 1996 and has no option rows at all; 104866 starts in 2001 and has
    161,208 rows in 2019 alone. A span-based tie-break picks 111306 and silently leaves the
    name with no tradable contracts for the entire sample. Activity must win.
    """
    link = pd.DataFrame(
        {
            "permno": [66157, 66157],
            "secid": [111306, 104866],
            "score": [1, 1],
            "sdate": pd.to_datetime(["1996-01-01", "2001-02-27"]),
            "edate": pd.to_datetime(["2025-12-31", "2025-12-31"]),
        }
    )
    activity = pd.Series({111306: 0, 104866: 161_208})

    # Without activity, the longer span wins -- and it is the wrong answer.
    assert apply_link(panel(66157, ["2019-06-03"]), link)["secid"].iloc[0] == 111306
    # With activity, the live chain wins.
    out = apply_link(panel(66157, ["2019-06-03"]), link, activity=activity)
    assert out["secid"].iloc[0] == 104866


def test_score_still_outranks_activity():
    """A better-scored link is not overridden by a busier but worse-matched secid."""
    link = pd.DataFrame(
        {
            "permno": [1, 1],
            "secid": [10, 20],
            "score": [1, 5],
            "sdate": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "edate": pd.to_datetime(["2020-12-31", "2020-12-31"]),
        }
    )
    activity = pd.Series({10: 5, 20: 999_999})
    out = apply_link(panel(1, ["2020-06-01"]), link, activity=activity)
    assert out["secid"].iloc[0] == 10


def test_secids_absent_from_the_activity_map_score_zero():
    link = pd.DataFrame(
        {
            "permno": [1, 1],
            "secid": [10, 20],
            "score": [1, 1],
            "sdate": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "edate": pd.to_datetime(["2020-12-31", "2020-06-30"]),
        }
    )
    out = apply_link(panel(1, ["2020-06-01"]), link, activity=pd.Series({20: 100}))
    assert out["secid"].iloc[0] == 20      # 10 is unmapped -> activity 0


def test_ambiguous_permnos_finds_only_multi_secid_names():
    link = pd.DataFrame(
        {
            "permno": [1, 1, 2, 3, 3],
            "secid": [10, 20, 30, 40, 40],   # permno 3 repeats ONE secid: not ambiguous
            "score": [1, 1, 1, 1, 1],
            "sdate": pd.to_datetime(["2020-01-01"] * 5),
            "edate": pd.to_datetime(["2020-12-31"] * 5),
        }
    )
    from vrp.data.linking import ambiguous_permnos
    assert ambiguous_permnos(link) == [1]


def test_tie_on_score_is_broken_by_longer_span_deterministically():
    link = pd.DataFrame(
        {
            "permno": [1, 1],
            "secid": [222, 333],
            "score": [1, 1],
            "sdate": pd.to_datetime(["2020-05-01", "2020-01-01"]),
            "edate": pd.to_datetime(["2020-07-01", "2020-12-31"]),
        }
    )
    out = apply_link(panel(1, ["2020-06-01"]), link)
    assert out["secid"].iloc[0] == 333          # the longer overlapping span
    assert len(out) == 1                        # exactly one row per (permno, date)


def test_one_row_per_permno_date_even_with_many_candidates():
    link = pd.DataFrame(
        {
            "permno": [1] * 3,
            "secid": [1, 2, 3],
            "score": [1, 1, 2],
            "sdate": pd.to_datetime(["2020-01-01"] * 3),
            "edate": pd.to_datetime(["2020-12-31"] * 3),
        }
    )
    out = apply_link(panel(1, ["2020-06-01", "2020-06-02"]), link)
    assert len(out) == 2
    assert not out.duplicated(subset=["permno", "date"]).any()


def test_link_conflicts_flags_concurrent_distinct_secids():
    link = pd.DataFrame(
        {
            "permno": [1, 1],
            "secid": [10, 20],
            "score": [1, 1],
            "sdate": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "edate": pd.to_datetime(["2020-12-31", "2020-12-31"]),
        }
    )
    conflicts = link_conflicts(panel(1, ["2020-06-01"]), link)
    assert len(conflicts) == 1
    assert conflicts["n_secid"].iloc[0] == 2


def test_same_secid_in_adjacent_spans_is_not_a_conflict():
    """A renewed span for the same secid must not be reported as an ambiguous link."""
    link = pd.DataFrame(
        {
            "permno": [1, 1],
            "secid": [10, 10],
            "score": [1, 1],
            "sdate": pd.to_datetime(["2020-01-01", "2020-06-01"]),
            "edate": pd.to_datetime(["2020-12-31", "2021-12-31"]),
        }
    )
    assert link_conflicts(panel(1, ["2020-07-01"]), link).empty


def test_link_coverage_counts_universe_cells():
    universe = pd.DataFrame(
        {
            "entry_month": pd.PeriodIndex(["2020-01", "2020-01", "2020-02"], freq="M"),
            "permno": [1, 2, 1],
        }
    )
    linked = pd.DataFrame({"permno": [1, 2], "secid": [10, pd.NA]})
    cov = link_coverage(universe, linked)
    assert cov["n_cells"] == 3
    assert cov["n_covered"] == 2               # permno 1 in both months
    assert cov["n_permnos_missing"] == 1
    assert cov["missing_permnos"] == [2]
