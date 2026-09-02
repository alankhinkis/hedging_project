"""Stage 2a price-cleaning tests.

The split normalisation is the single most dangerous line in the data build: mixing an
adjusted stock price with an as-traded strike produces a large, plausible-looking, entirely
fake P&L. It is tested here on a fixture built to mimic AAPL's 2020-08-31 4-for-1 split,
with a position spanning the split in both directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrp.data.crsp_prices import (
    clean_prices,
    delisted_before,
    first_trading_days,
    normalize_to_entry_factor,
    price_panel,
    return_reconciliation,
    split_jump,
    trading_calendar,
)


@pytest.fixture
def split_panel() -> pd.DataFrame:
    """A 4-for-1 split on 2020-08-31, in CRSP's conventions.

    CRSP's `cfacpr` is cumulative to the end of file, so it is 4.0 *before* the split and
    1.0 after. The raw price quarters overnight; the adjusted series is continuous.
    """
    dates = pd.to_datetime(
        ["2020-08-26", "2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01", "2020-09-02"]
    )
    raw = [506.0, 500.0, 500.0, 129.0, 134.0, 131.0]
    cfacpr = [4.0, 4.0, 4.0, 1.0, 1.0, 1.0]
    return pd.DataFrame(
        {
            "permno": 14593,
            "date": dates,
            "prc": raw,
            "cfacpr": cfacpr,
            "cfacshr": cfacpr,
            "shrout": 4_300_000.0,
            "ret": [0.0, -0.0119, 0.0, 0.0339, 0.0398, -0.0207],
            "retx": [0.0, -0.0119, 0.0, 0.0339, 0.0398, -0.0207],
        }
    )


# ---------------------------------------------------------------------------
# CRSP conventions
# ---------------------------------------------------------------------------

def test_negative_price_is_a_quote_average_not_a_bad_row():
    df = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
            "prc": [100.0, -98.0],       # second day: no closing trade, bid-ask midpoint
            "cfacpr": [1.0, 1.0],
            "cfacshr": [1.0, 1.0],
            "shrout": [1000.0, 1000.0],
        }
    )
    out = clean_prices(df)
    assert list(out["price_is_quote_avg"]) == [False, True]
    assert list(out["prc_abs"]) == [100.0, 98.0]
    assert out["has_price"].all()            # the row is kept, not dropped
    assert list(out["s_adj"]) == [100.0, 98.0]


def test_missing_price_is_flagged_and_kept():
    df = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
            "prc": [100.0, np.nan],      # trading halt
            "cfacpr": [1.0, 1.0],
            "cfacshr": [1.0, 1.0],
            "shrout": [1000.0, 1000.0],
        }
    )
    out = clean_prices(df)
    assert list(out["has_price"]) == [True, False]
    assert len(out) == 2


def test_zero_cfacpr_is_nulled_not_divided_by():
    df = pd.DataFrame(
        {
            "permno": [1],
            "date": pd.to_datetime(["2019-01-02"]),
            "prc": [100.0],
            "cfacpr": [0.0],
            "cfacshr": [1.0],
            "shrout": [1000.0],
        }
    )
    out = clean_prices(df)
    assert bool(out["cfacpr_is_bad"].iloc[0])
    assert pd.isna(out["s_adj"].iloc[0])     # NaN, not inf


# ---------------------------------------------------------------------------
# The split trap (Q11)
# ---------------------------------------------------------------------------

def test_adjusted_series_is_continuous_across_the_split(split_panel):
    out = clean_prices(split_panel)
    jump = split_jump(out, 14593, "2020-08-31")
    assert jump["raw_log_change"] == pytest.approx(np.log(129.0 / 500.0), abs=1e-9)
    assert abs(jump["raw_log_change"]) > 1.0          # the raw series really does jump
    assert abs(jump["adj_log_change"]) < 0.05         # the adjusted one does not


def test_normalize_to_entry_before_the_split_keeps_entry_units(split_panel):
    """Entering before the split: prices stay in PRE-split (as-traded) units all the way.

    The strike on that contract was struck pre-split, so the path must be pre-split too.
    """
    out = clean_prices(split_panel)
    s = normalize_to_entry_factor(out, "2020-08-27", permno=14593)
    assert s.iloc[1] == pytest.approx(500.0)          # entry day is untouched
    # Post-split raw 129 is 516 in pre-split units -- the same economic level, x4.
    assert s.iloc[3] == pytest.approx(129.0 * 4.0)
    # No discontinuity: the overnight move is the real +3.4%, not -74%.
    assert abs(np.log(s.iloc[3] / s.iloc[2])) < 0.05


def test_normalize_to_entry_after_the_split_keeps_post_split_units(split_panel):
    out = clean_prices(split_panel)
    s = normalize_to_entry_factor(out, "2020-09-01", permno=14593)
    assert s.iloc[4] == pytest.approx(134.0)
    assert s.iloc[1] == pytest.approx(500.0 / 4.0)    # pre-split day expressed post-split
    assert abs(np.log(s.iloc[3] / s.iloc[2])) < 0.05


def test_normalizing_without_the_fix_would_produce_a_fake_75pct_move(split_panel):
    """Guards the reason the fix exists: raw prices across a split look like a crash."""
    out = clean_prices(split_panel)
    naive = out["prc_abs"]
    assert np.log(naive.iloc[3] / naive.iloc[2]) < -1.0     # a fake -74% overnight
    fixed = normalize_to_entry_factor(out, "2020-08-27", permno=14593)
    assert abs(np.log(fixed.iloc[3] / fixed.iloc[2])) < 0.05


def test_normalize_rejects_an_entry_date_with_no_price_row(split_panel):
    out = clean_prices(split_panel)
    with pytest.raises(KeyError):
        normalize_to_entry_factor(out, "2020-08-29", permno=14593)   # a Saturday


# ---------------------------------------------------------------------------
# Reconciliation and calendar helpers
# ---------------------------------------------------------------------------

def test_return_reconciliation_matches_retx_exactly_when_consistent():
    """Build a series from retx and confirm the check recovers it to machine precision."""
    dates = pd.bdate_range("2019-01-01", periods=60)
    rng = np.random.default_rng(0)
    retx = rng.normal(0.0004, 0.012, len(dates))
    retx[0] = 0.0
    prc = 100.0 * np.cumprod(1.0 + retx)
    df = pd.DataFrame(
        {
            "permno": 1,
            "date": dates,
            "prc": prc,
            "cfacpr": 1.0,
            "cfacshr": 1.0,
            "shrout": 1000.0,
            "retx": retx,
            "ret": retx + 0.00008,        # a small steady dividend contribution
        }
    )
    recon = return_reconciliation(clean_prices(df), 1, 2019)
    assert recon["price_vs_retx_rel_error"] < 1e-9
    assert recon["implied_div_yield"] > 0     # ret compounds above retx


def test_return_reconciliation_catches_a_broken_adjustment():
    """Corrupt cfacpr midway; the price series and retx must then disagree."""
    dates = pd.bdate_range("2019-01-01", periods=60)
    retx = np.full(len(dates), 0.001)
    retx[0] = 0.0
    prc = 100.0 * np.cumprod(1.0 + retx)
    cfacpr = np.where(np.arange(len(dates)) < 30, 2.0, 1.0)   # a split the price ignores
    df = pd.DataFrame(
        {
            "permno": 1, "date": dates, "prc": prc, "cfacpr": cfacpr,
            "cfacshr": 1.0, "shrout": 1000.0, "retx": retx, "ret": retx,
        }
    )
    recon = return_reconciliation(clean_prices(df), 1, 2019)
    assert recon["price_vs_retx_rel_error"] > 0.5


def test_first_trading_days_picks_the_first_open_day_of_each_month():
    # 2019-09-01 is a Sunday and 2019-09-02 is Labor Day, so September opens on the 3rd.
    dates = pd.to_datetime(
        ["2019-08-30", "2019-09-03", "2019-09-04", "2019-10-01", "2019-10-02"]
    )
    prices = pd.DataFrame({"permno": 1, "date": dates})
    firsts = first_trading_days(prices)
    got = dict(zip(firsts["entry_month"].astype(str), firsts["entry_date"]))
    assert got["2019-09"] == pd.Timestamp("2019-09-03")
    assert got["2019-10"] == pd.Timestamp("2019-10-01")


def test_trading_calendar_is_sorted_and_unique():
    prices = pd.DataFrame(
        {"permno": [1, 2, 1], "date": pd.to_datetime(["2019-01-03", "2019-01-02", "2019-01-02"])}
    )
    cal = trading_calendar(prices)
    assert list(cal) == [pd.Timestamp("2019-01-02"), pd.Timestamp("2019-01-03")]


def test_price_panel_slices_one_name_inclusively(split_panel):
    out = clean_prices(split_panel)
    panel = price_panel(out, 14593, "2020-08-27", "2020-08-31")
    assert len(panel) == 3
    assert panel["date"].iloc[0] == pd.Timestamp("2020-08-27")
    assert panel["date"].iloc[-1] == pd.Timestamp("2020-08-31")


def test_delisted_before_is_strict():
    delist = pd.DataFrame(
        {"permno": [7], "dlstdt": pd.to_datetime(["2019-06-15"]), "dlstcd": [233]}
    )
    assert delisted_before(delist, 7, "2019-06-16")
    assert not delisted_before(delist, 7, "2019-06-15")     # strictly before
    assert not delisted_before(delist, 8, "2020-01-01")
    assert not delisted_before(pd.DataFrame(), 7, "2020-01-01")


# ---------------------------------------------------------------------------
# Entry-timing rule (Q10, revised)
# ---------------------------------------------------------------------------

def test_third_friday_is_the_monthly_expiration():
    from vrp.data.crsp_prices import third_friday
    assert third_friday(2019, 3) == pd.Timestamp("2019-03-15")
    assert third_friday(2020, 8) == pd.Timestamp("2020-08-21")
    # A month starting on a Friday still yields the third Friday, not the second.
    assert third_friday(2021, 1) == pd.Timestamp("2021-01-15")


def test_post_expiry_entry_is_the_next_trading_day_after_expiration():
    from vrp.data.crsp_prices import post_expiry_entry_days
    cal = pd.bdate_range("2019-03-01", "2019-04-30")
    prices = pd.DataFrame({"permno": 1, "date": cal})
    entries = post_expiry_entry_days(prices).set_index("entry_month")["entry_date"]
    # 2019-03-15 is the third Friday, so March's entry is Monday the 18th.
    assert entries[pd.Period("2019-03", "M")] == pd.Timestamp("2019-03-18")


def test_post_expiry_entry_skips_a_holiday_after_expiration():
    """If the Monday after expiration is closed, entry moves to the next open day."""
    from vrp.data.crsp_prices import post_expiry_entry_days
    cal = [d for d in pd.bdate_range("2019-03-01", "2019-03-29")
           if d != pd.Timestamp("2019-03-18")]
    prices = pd.DataFrame({"permno": 1, "date": pd.DatetimeIndex(cal)})
    entries = post_expiry_entry_days(prices).set_index("entry_month")["entry_date"]
    assert entries[pd.Period("2019-03", "M")] == pd.Timestamp("2019-03-19")


def test_post_expiry_entry_leaves_about_thirty_days_to_the_next_expiration():
    """The whole point of the rule: the NEXT monthly expiry lands inside the 20-40 window
    for every name, whether or not it has weekly options."""
    from vrp.data.crsp_prices import post_expiry_entry_days, third_friday
    cal = pd.bdate_range("2019-01-01", "2023-12-31")
    prices = pd.DataFrame({"permno": 1, "date": cal})
    entries = post_expiry_entry_days(prices)
    gaps = []
    for _, row in entries.iterrows():
        nxt = row["entry_month"] + 1
        gaps.append((third_friday(nxt.year, nxt.month) - row["entry_date"]).days)
    gaps = pd.Series(gaps)
    # The containment is the property that matters. The gap alternates 25/32 days
    # depending on where the two third-Fridays fall, so the median is not ~30 -- but every
    # value clears the window, which is what full coverage depends on.
    assert gaps.between(20, 40).all(), f"out-of-window gaps: {sorted(set(gaps))}"
    assert set(gaps.unique()) <= {25, 32}
    assert 24 <= gaps.median() <= 32


def test_entry_calendar_honours_the_configured_rule():
    from vrp.config import Config
    from vrp.data.crsp_prices import entry_calendar
    from pathlib import Path
    prices = pd.DataFrame({"permno": 1, "date": pd.bdate_range("2019-03-01", "2019-03-29")})

    def cfg_for(rule):
        return Config(raw={"selection": {"entry_rule": rule}, "paths": {}}, root=Path("."))

    first = entry_calendar(prices, cfg_for("first_trading_day_of_month"))
    assert first["entry_date"].iloc[0] == pd.Timestamp("2019-03-01")
    post = entry_calendar(prices, cfg_for("first_trading_day_after_monthly_expiry"))
    assert post["entry_date"].iloc[0] == pd.Timestamp("2019-03-18")
    with pytest.raises(ValueError):
        entry_calendar(prices, cfg_for("nonsense"))
