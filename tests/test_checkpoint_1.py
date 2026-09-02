"""Exercise Checkpoint 1 itself on synthetic data.

`checkpoint_1` is Stage 1's deliverable, and it is the code that decides whether the build
may proceed. It should not run for the first time against live WRDS data -- a bug in the
checkpoint is indistinguishable from a bug in the universe. So it is run here against a
fixture designed to pass, and against mutations designed to fail each check individually.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vrp.config import Config
from vrp.data.universe import build_universe, checkpoint_1


N_NAMES = 6
TSLA = 99  # the late joiner, standing in for TSLA


@pytest.fixture
def cfg() -> Config:
    """A miniature config: 6 names, top-3, a 2019 sample, a late joiner added mid-year."""
    return Config(
        raw={
            "sample": {
                "start_date": "2019-01-01",
                "end_date": "2020-12-31",
                "burnin_start": "2018-01-01",
                "hold_buffer_end": "2021-02-28",
            },
            "universe": {
                "top_n": 3,
                "expected_members_min": 4,
                "expected_members_max": 6,
                "expected_distinct_permnos_min": 3,
            },
            "sanity": {
                "tsla_permno": TSLA,
                "tsla_index_add_date": "2019-07-15",
                "expected_top_members": {"2019-03": ["AAA"]},
            },
            "paths": {},
        },
        root=Path("."),
    )


@pytest.fixture
def membership() -> pd.DataFrame:
    """Five permanent members plus a late joiner; one permanent member exits mid-2019."""
    rows = [
        {"permno": p, "from_date": "2018-01-01", "thru_date": "2020-12-31"}
        for p in range(1, 6)
    ]
    rows[4]["thru_date"] = "2019-09-20"          # permno 5 is deleted mid-sample
    rows.append({"permno": TSLA, "from_date": "2019-07-15", "thru_date": "2020-12-31"})
    df = pd.DataFrame(rows)
    df["from_date"] = pd.to_datetime(df["from_date"])
    df["thru_date"] = pd.to_datetime(df["thru_date"])
    return df


@pytest.fixture
def mktcap() -> pd.DataFrame:
    """Caps chosen so the top-3 actually turns over.

    Permno 1 is always first. Permno 5 is second until it is deleted in September 2019;
    the late joiner takes that slot afterwards. So the universe touches four distinct
    permnos over the sample even though only three are held at a time -- which is what
    check 6 (turnover) is looking for.
    """
    caps = {1: 1000.0, 2: 900.0, 3: 800.0, 4: 700.0, 5: 950.0, TSLA: 940.0}
    dates = pd.date_range("2018-01-31", "2020-12-31", freq="ME")
    rows = [
        {"permno": p, "date": d, "mktcap_k": cap, "prc": 10.0, "shrout": 100}
        for d in dates
        for p, cap in caps.items()
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def names() -> pd.DataFrame:
    rows = []
    for p, tic in [(1, "AAA"), (2, "BBB"), (3, "CCC"), (4, "DDD"), (5, "EEE"), (TSLA, "TSLA")]:
        rows.append(
            {
                "permno": p,
                "namedt": pd.Timestamp("2010-01-01"),
                "nameendt": pd.Timestamp("2030-12-31"),
                "ticker": tic,
                "comnam": tic + " Corp",
            }
        )
    return pd.DataFrame(rows)


def as_dict(results):
    return {name: (ok, detail) for name, ok, detail in results}


def test_all_checks_pass_on_a_clean_fixture(cfg, membership, mktcap, names):
    universe = build_universe(membership, mktcap, top_n=3)
    results = checkpoint_1(membership, mktcap, universe, cfg, names=names)
    failures = [(n, d) for n, ok, d in results if not ok]
    assert not failures, failures
    # Guards against a check being silently dropped from the checkpoint.
    assert len(results) == 7


def test_member_count_check_catches_a_broken_join(cfg, membership, mktcap, names):
    """Drop two names' market caps and the per-month member count falls below the band."""
    thin = mktcap.loc[~mktcap["permno"].isin([4, 5])]
    universe = build_universe(membership, thin, top_n=3)
    results = as_dict(checkpoint_1(membership, thin, universe, cfg, names=names))
    key = next(k for k in results if k.startswith("member count"))
    assert results[key][0] is False


def test_survivorship_check_catches_a_carried_forward_member(cfg, membership, mktcap, names):
    """Backdate the late joiner's membership to the start -- the TSLA check must fail."""
    broken = membership.copy()
    broken.loc[broken["permno"] == TSLA, "from_date"] = pd.Timestamp("2018-01-01")
    universe = build_universe(broken, mktcap, top_n=3)
    results = as_dict(checkpoint_1(broken, mktcap, universe, cfg, names=names))
    key = next(k for k in results if k.startswith("TSLA"))
    assert results[key][0] is False


def test_turnover_check_catches_a_static_universe(cfg, membership, mktcap, names):
    """A universe of exactly top_n permanent names has no turnover and must fail check 6."""
    static_membership = membership.loc[membership["permno"].isin([1, 2, 3])].copy()
    static_membership["thru_date"] = pd.Timestamp("2020-12-31")
    static_mktcap = mktcap.loc[mktcap["permno"].isin([1, 2, 3])]
    universe = build_universe(static_membership, static_mktcap, top_n=3)
    results = as_dict(checkpoint_1(static_membership, static_mktcap, universe, cfg, names=names))
    key = next(k for k in results if k.startswith("distinct permnos"))
    assert results[key][0] is False


def test_expected_members_check_catches_a_missing_name(cfg, membership, mktcap, names):
    """Demote AAA below the top-3 cut; the 'expected large caps present' check must fail."""
    demoted = mktcap.copy()
    demoted.loc[demoted["permno"] == 1, "mktcap_k"] = 1.0
    universe = build_universe(membership, demoted, top_n=3)
    results = as_dict(checkpoint_1(membership, demoted, universe, cfg, names=names))
    key = next(k for k in results if k.startswith("expected large caps"))
    assert results[key][0] is False


def test_missing_ticker_history_is_reported_not_silently_passed(cfg, membership, mktcap):
    """Without a name history the ticker check cannot run -- it must fail, not pass."""
    universe = build_universe(membership, mktcap, top_n=3)
    results = as_dict(checkpoint_1(membership, mktcap, universe, cfg, names=None))
    key = next(k for k in results if k.startswith("expected large caps"))
    assert results[key][0] is False
