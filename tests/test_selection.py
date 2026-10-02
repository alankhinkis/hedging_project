"""Stage 2b selection tests.

The selection rule has a subtle ordering requirement: the expiration is chosen first, and
only then the strike within it. Picking the globally-nearest-0.50-delta contract can land on
a different expiry than the nearest-30-day one, which would quietly break the "one ~30-day
contract per stock-month" design. That ordering is pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrp.config import load_config
from vrp.selection import apply_screens, coverage_report, select_contracts


def chain(rows: list[dict]) -> pd.DataFrame:
    """Build a candidate chain with sane defaults, overriding per row."""
    base = {
        "secid": 1,
        "date": pd.Timestamp("2019-03-01"),
        "exdate": pd.Timestamp("2019-03-29"),
        "optionid": 0,
        "cp_flag": "C",
        "strike": 100.0,
        "best_bid": 4.0,
        "best_offer": 4.2,
        "volume": 100.0,
        "open_interest": 500.0,
        "impl_volatility": 0.30,
        "delta": 0.50,
        "gamma": 0.02,
        "vega": 0.10,
        "theta": -0.05,
        "cfadj": 1.0,
        "ss_flag": "0",
        "contract_size": 100.0,
        "am_settlement": 0.0,
        "expiry_indicator": None,
        "forward_price": 100.0,
        "dte": 28,
    }
    out = []
    for i, r in enumerate(rows):
        row = dict(base)
        row["optionid"] = 1000 + i
        row.update(r)
        out.append(row)
    return pd.DataFrame(out)


@pytest.fixture
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------

def test_picks_the_expiration_nearest_thirty_days(cfg):
    c = chain([
        {"exdate": pd.Timestamp("2019-03-22"), "dte": 21, "delta": 0.50},
        {"exdate": pd.Timestamp("2019-03-31"), "dte": 30, "delta": 0.50},
        {"exdate": pd.Timestamp("2019-04-10"), "dte": 40, "delta": 0.50},
    ])
    picked = select_contracts(c, cfg)
    assert len(picked) == 1
    assert picked["dte"].iloc[0] == 30


def test_expiration_tie_goes_to_the_shorter_maturity(cfg):
    """25 and 35 days are equidistant from 30; the plan says take the shorter."""
    c = chain([
        {"exdate": pd.Timestamp("2019-03-26"), "dte": 25, "delta": 0.50},
        {"exdate": pd.Timestamp("2019-04-05"), "dte": 35, "delta": 0.50},
    ])
    picked = select_contracts(c, cfg)
    assert picked["dte"].iloc[0] == 25


def test_expiration_is_chosen_before_the_strike(cfg):
    """The 30-day expiry has only a 0.44-delta contract; a 21-day one sits at exactly 0.50.

    Choosing globally on delta would pick the 21-day contract. The rule is expiry first.
    """
    c = chain([
        {"exdate": pd.Timestamp("2019-03-22"), "dte": 21, "delta": 0.500, "strike": 100.0},
        {"exdate": pd.Timestamp("2019-03-31"), "dte": 30, "delta": 0.440, "strike": 103.0},
        {"exdate": pd.Timestamp("2019-03-31"), "dte": 30, "delta": 0.610, "strike": 97.0},
    ])
    picked = select_contracts(c, cfg)
    assert picked["dte"].iloc[0] == 30
    assert picked["delta"].iloc[0] == pytest.approx(0.44)


def test_within_the_expiration_picks_nearest_half_delta(cfg):
    c = chain([
        {"delta": 0.38, "strike": 106.0},
        {"delta": 0.52, "strike": 99.0},
        {"delta": 0.63, "strike": 95.0},
    ])
    picked = select_contracts(c, cfg)
    assert picked["delta"].iloc[0] == pytest.approx(0.52)


def test_puts_use_absolute_delta(cfg):
    c = chain([
        {"cp_flag": "P", "delta": -0.38},
        {"cp_flag": "P", "delta": -0.51},
        {"cp_flag": "P", "delta": -0.64},
    ])
    picked = select_contracts(c, cfg)
    assert picked["delta"].iloc[0] == pytest.approx(-0.51)
    assert picked["abs_delta"].iloc[0] == pytest.approx(0.51)


def test_calls_and_puts_are_selected_independently(cfg):
    c = chain([
        {"cp_flag": "C", "delta": 0.49},
        {"cp_flag": "C", "delta": 0.60},
        {"cp_flag": "P", "delta": -0.52},
        {"cp_flag": "P", "delta": -0.40},
    ])
    picked = select_contracts(c, cfg).sort_values("cp_flag")
    assert len(picked) == 2
    assert picked.loc[picked.cp_flag == "C", "delta"].iloc[0] == pytest.approx(0.49)
    assert picked.loc[picked.cp_flag == "P", "delta"].iloc[0] == pytest.approx(-0.52)


def test_delta_tie_breaks_on_open_interest(cfg):
    c = chain([
        {"delta": 0.55, "open_interest": 100.0, "optionid": 1},
        {"delta": 0.45, "open_interest": 9000.0, "optionid": 2},
    ])
    picked = select_contracts(c, cfg)
    assert picked["open_interest"].iloc[0] == 9000.0


def test_contracts_outside_the_delta_band_are_never_selected(cfg):
    c = chain([{"delta": 0.90}, {"delta": 0.15}])
    assert len(select_contracts(c, cfg)) == 0


def test_contracts_outside_the_dte_rule_are_dropped_even_though_pulled(cfg):
    """The pull uses a wider window than the rule; selection must still enforce the rule."""
    c = chain([
        {"exdate": pd.Timestamp("2019-03-17"), "dte": 16},
        {"exdate": pd.Timestamp("2019-04-18"), "dte": 48},
    ])
    assert len(select_contracts(c, cfg)) == 0


def test_selected_rows_carry_mid_and_spread(cfg):
    c = chain([{"best_bid": 4.0, "best_offer": 4.4}])
    picked = select_contracts(c, cfg)
    assert picked["mid"].iloc[0] == pytest.approx(4.2)
    assert picked["spread"].iloc[0] == pytest.approx(0.4)
    assert picked["rel_spread"].iloc[0] == pytest.approx(0.4 / 4.2)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def test_each_screen_drops_only_its_own_row(cfg):
    c = chain([
        {"optionid": 1},                                  # clean
        {"optionid": 2, "ss_flag": "1"},                  # non-standard settlement
        {"optionid": 3, "contract_size": 90.0},           # adjusted deliverable
        {"optionid": 4, "best_bid": 0.0},                 # no bid
        {"optionid": 5, "best_bid": 4.5, "best_offer": 4.4},   # crossed quote
        {"optionid": 6, "open_interest": 0.0},            # no open interest
        {"optionid": 7, "impl_volatility": np.nan},       # no IV
        {"optionid": 8, "impl_volatility": 5.0},          # IV above the 300% cap
        {"optionid": 9, "delta": np.nan},                 # no delta
        {"optionid": 10, "expiry_indicator": "w"},        # weekly, not a standard monthly
    ])
    kept, log = apply_screens(c, cfg)
    assert set(kept["optionid"]) == {1}
    # Every screen except the raw row and the skipped arbitrage check drops exactly one.
    dropped = log.loc[log["screen"].str.contains("raw|arbitrage") == False, "rows_dropped"]
    assert set(dropped) == {1}


def test_iv_cap_is_a_parameter_not_a_constant(cfg):
    """Q8: a 150% IV survives the 300% primary cap and is cut by BK's literal 100%."""
    c = chain([{"optionid": 1, "impl_volatility": 1.50}])
    assert len(apply_screens(c, cfg)[0]) == 1
    assert len(apply_screens(c, cfg, iv_cap=1.00)[0]) == 0


def test_screen_log_tracks_stock_months_not_just_rows(cfg):
    c = chain([
        {"secid": 1, "optionid": 1},
        {"secid": 2, "optionid": 2, "best_bid": 0.0},
    ])
    _, log = apply_screens(c, cfg)
    bid = log.loc[log["screen"] == "positive bid"].iloc[0]
    assert bid["stock_months_in"] == 2
    assert bid["stock_months_out"] == 1
    assert bid["stock_months_lost"] == 1


def test_arbitrage_bounds_drop_an_impossible_call(cfg):
    spot = pd.DataFrame({"secid": [1], "date": [pd.Timestamp("2019-03-01")], "S": [100.0]})
    c = chain([
        {"optionid": 1, "strike": 100.0, "best_bid": 4.0, "best_offer": 4.2},   # fine
        {"optionid": 2, "strike": 100.0, "best_bid": 120.0, "best_offer": 121.0},  # C > S
        {"optionid": 3, "strike": 80.0, "best_bid": 5.0, "best_offer": 5.2},    # C < S - K
    ])
    kept, log = apply_screens(c, cfg, spot=spot)
    assert set(kept["optionid"]) == {1}
    assert "SKIPPED" not in log.loc[log["screen"].str.contains("arbitrage"), "note"].iloc[0]


def test_arbitrage_screen_is_skipped_and_says_so_without_spot(cfg):
    c = chain([{"optionid": 1}])
    _, log = apply_screens(c, cfg)
    assert "SKIPPED" in log.loc[log["screen"].str.contains("arbitrage"), "note"].iloc[0]


def test_rows_without_a_spot_price_pass_the_arbitrage_screen(cfg):
    """A missing spot must not silently delete the contract; that is a coverage question."""
    spot = pd.DataFrame({"secid": [99], "date": [pd.Timestamp("2019-03-01")], "S": [100.0]})
    c = chain([{"optionid": 1, "secid": 1}])
    kept, _ = apply_screens(c, cfg, spot=spot)
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# Coverage (Q9)
# ---------------------------------------------------------------------------

def test_coverage_report_counts_missing_stock_months():
    spec = pd.DataFrame(
        {
            "entry_month": pd.PeriodIndex(["2019-01", "2019-01"], freq="M"),
            "entry_date": [pd.Timestamp("2019-01-02")] * 2,
            "secid": [1, 2],
            "rank": [1, 2],
            "mktcap_k": [100.0, 50.0],
            "has_secid": [True, True],
        }
    )
    selected = pd.DataFrame(
        {
            "secid": [1, 1],
            "entry_date": [pd.Timestamp("2019-01-02")] * 2,
            "cp_flag": ["C", "P"],
        }
    )
    cov = coverage_report(spec, selected, by="year")
    row = cov.iloc[0]
    assert row["stock_months"] == 2
    assert row["with_call"] == 1
    assert row["with_put"] == 1
    assert row["coverage"] == pytest.approx(0.5)


def test_coverage_ignores_stock_months_that_never_had_a_secid():
    """A cell with no secid is a linking gap, not a selection failure; it is excluded."""
    spec = pd.DataFrame(
        {
            "entry_month": pd.PeriodIndex(["2019-01", "2019-01"], freq="M"),
            "entry_date": [pd.Timestamp("2019-01-02")] * 2,
            "secid": [1, np.nan],
            "rank": [1, 2],
            "mktcap_k": [100.0, 50.0],
            "has_secid": [True, False],
        }
    )
    selected = pd.DataFrame(
        {"secid": [1], "entry_date": [pd.Timestamp("2019-01-02")], "cp_flag": ["C"]}
    )
    cov = coverage_report(spec, selected, by="year")
    assert cov["stock_months"].iloc[0] == 1
    assert cov["coverage"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Expiration cycle and position-level filters
# ---------------------------------------------------------------------------

def test_weeklies_are_screened_out_when_configured(cfg):
    """IvyDB marks weeklies with expiry_indicator='w'; standard monthlies are null."""
    c = chain([
        {"optionid": 1, "expiry_indicator": None},
        {"optionid": 2, "expiry_indicator": "w"},
    ])
    kept, log = apply_screens(c, cfg)
    assert set(kept["optionid"]) == {1}
    row = log.loc[log["screen"] == "standard monthly expiration"].iloc[0]
    assert row["rows_dropped"] == 1


def test_position_filters_drop_a_delisting_inside_the_hold():
    from vrp.selection import apply_position_filters
    sel = pd.DataFrame({
        "secid": [1, 2, 3],
        "optionid": [10, 20, 30],
        "entry_date": pd.to_datetime(["2019-01-22"] * 3),
        "exdate": pd.to_datetime(["2019-02-15"] * 3),
    })
    linked = pd.DataFrame({"permno": [101, 102, 103], "secid": [1, 2, 3],
                           "date": pd.to_datetime(["2019-01-22"] * 3)})
    prices = pd.DataFrame({
        "permno": [101, 102, 103],
        "date": pd.to_datetime(["2019-03-01"] * 3),
        "has_price": True,
    })
    delist = pd.DataFrame({
        "permno": [102, 103],
        # 102 delists mid-hold -> excluded; 103 delists after expiry -> kept.
        "dlstdt": pd.to_datetime(["2019-02-01", "2019-06-01"]),
    })
    kept, report = apply_position_filters(sel, prices, delist, linked)
    assert set(kept["secid"]) == {1, 3}
    q7 = report.loc[report["filter"].str.contains("delist")].iloc[0]
    assert q7["dropped"] == 1


def test_position_filters_drop_positions_with_no_price_at_expiry():
    from vrp.selection import apply_position_filters
    sel = pd.DataFrame({
        "secid": [1, 2],
        "optionid": [10, 20],
        "entry_date": pd.to_datetime(["2019-01-22"] * 2),
        "exdate": pd.to_datetime(["2019-02-15"] * 2),
    })
    linked = pd.DataFrame({"permno": [101, 102], "secid": [1, 2],
                           "date": pd.to_datetime(["2019-01-22"] * 2)})
    prices = pd.DataFrame({
        "permno": [101, 102],
        # permno 102's price series stops before expiry -> no S_T for the intrinsic value.
        "date": pd.to_datetime(["2019-03-01", "2019-02-01"]),
        "has_price": True,
    })
    kept, report = apply_position_filters(sel, prices, pd.DataFrame(), linked)
    assert set(kept["secid"]) == {1}
    px = report.loc[report["filter"].str.contains("price available")].iloc[0]
    assert px["dropped"] == 1


def test_position_filters_keep_a_delisting_outside_the_hold():
    """A delisting on the entry date or after expiry must not remove the position."""
    from vrp.selection import apply_position_filters
    sel = pd.DataFrame({
        "secid": [1], "optionid": [10],
        "entry_date": pd.to_datetime(["2019-01-22"]),
        "exdate": pd.to_datetime(["2019-02-15"]),
    })
    linked = pd.DataFrame({"permno": [101], "secid": [1],
                           "date": pd.to_datetime(["2019-01-22"])})
    prices = pd.DataFrame({"permno": [101], "date": pd.to_datetime(["2019-03-01"]),
                           "has_price": True})
    on_entry = pd.DataFrame({"permno": [101], "dlstdt": pd.to_datetime(["2019-01-22"])})
    assert len(apply_position_filters(sel, prices, on_entry, linked)[0]) == 1


def test_checkpoint_2b_defers_path_conditions_when_pass_b_has_not_run(cfg):
    """A deferred check must be OMITTED, never reported as passed or failed.

    Pass B is a remote call that can fail. Calling its conditions passed would assert
    something unverified; calling them failed would gate the build on a step the headline
    P&L does not need (M1). The non-path conditions must still be evaluated.
    """
    from vrp.selection import checkpoint_2b

    sel = select_contracts(chain([{"optionid": 1}]), cfg)
    sel["permno"] = 101
    spec = pd.DataFrame({
        "entry_month": pd.PeriodIndex(["2019-03"], freq="M"),
        "entry_date": [pd.Timestamp("2019-03-01")],
        "secid": [1], "rank": [1], "mktcap_k": [100.0], "has_secid": [True],
    })
    spot = pd.DataFrame({"secid": [1], "date": [pd.Timestamp("2019-03-01")], "S": [100.0]})
    curve = pd.DataFrame({"date": [pd.Timestamp("2019-03-01")], "days": [30.0], "r": [0.02]})

    deferred = checkpoint_2b(spec, chain([{"optionid": 1}]), sel, pd.DataFrame(),
                             spot, curve, cfg, paths_pending=True)
    names = [n for n, _, _ in deferred]
    assert not any("Pass B path length" in n for n in names)
    assert not any("December cohorts" in n for n in names)
    # The conditions that do not need paths still run.
    assert any("exactly one contract" in n for n in names)
    assert any("survives to expiry" in n for n in names)

    # Without the flag, an empty paths frame is a genuine failure, not a deferral.
    strict = checkpoint_2b(spec, chain([{"optionid": 1}]), sel, pd.DataFrame(),
                           spot, curve, cfg)
    assert any("Pass B path length" in n and not ok for n, ok, _ in strict)
