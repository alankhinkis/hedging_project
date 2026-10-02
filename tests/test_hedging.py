"""Stage 4 hedging-engine tests.

The headline test is Checkpoint 4(a): on simulated paths priced at the same volatility the
stock diffuses at, Proposition 1 says the mean delta-hedged gain is zero. A sign error, a
financing error or an off-by-one in the rebalance loop all surface there as a spuriously
non-zero mean, in a world where the answer is provably zero. Everything else here pins a
specific piece of the arithmetic so that when that test fails, it is findable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrp.hedging import (
    OptionPosition,
    checkpoint_4a,
    delta_hedged_gain,
    run_positions,
    simulate_zero_vrp,
)
from vrp.pricing import bs_delta, bs_price


def flat_setup(S_path, sigma=0.3, r=0.02, K=100.0, cp="C", entry_price=None, tau_days=30):
    """A position on a given price path, with constant volatility and rates."""
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
                              for i in range(len(S_path))])
    und = pd.DataFrame({"date": dates, "S": np.asarray(S_path, dtype="float64"), "pv_divs": 0.0})
    if entry_price is None:
        entry_price = float(bs_price(S_path[0], K, tau_days / 365.0, sigma, r, 0.0, cp))
    pos = OptionPosition(secid=1, permno=1, optionid=1, cp_flag=cp, strike=K,
                         entry_date=dates[0], expiry=dates[-1], entry_price=entry_price)
    return pos, und, pd.Series(sigma, index=dates), pd.Series(r, index=dates)


# ---------------------------------------------------------------------------
# Checkpoint 4(a) -- the whole ballgame
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_zero_vrp_synthetic_mean_is_zero():
    """Proposition 1 / Eq. (16) in simulation. This is the test that validates the engine
    against the paper's own null hypothesis before any real data is touched."""
    for name, ok, detail in checkpoint_4a(n_paths=800, seed=3):
        assert ok, f"{name}: {detail}"


def test_hedging_error_dispersion_falls_as_one_over_sqrt_n():
    """Boyle & Emanuel (1980): discrete delta-hedging error has standard deviation
    O(1/sqrt(N)). Quadrupling the rebalance count must roughly halve it.

    This replaces a mean-based version of the same check: the discretisation BIAS is far
    smaller than the Monte Carlo noise in the mean at any feasible path count, so comparing
    means tests sampling luck. The dispersion signal is large and clean, and an engine that
    mis-accumulates the hedge or financing legs will not reproduce it.
    """
    lo = float(simulate_zero_vrp(n_paths=600, n_steps=10, seed=11)["pnl"].std(ddof=1))
    hi = float(simulate_zero_vrp(n_paths=600, n_steps=40, seed=11)["pnl"].std(ddof=1))
    assert 1.5 < lo / hi < 2.7        # sqrt(4) = 2.0 expected


# ---------------------------------------------------------------------------
# The P&L identity, piece by piece
# ---------------------------------------------------------------------------

def test_flat_price_path_leaves_only_time_decay_and_financing():
    """If the stock never moves, the hedge P&L is exactly zero and the option simply decays
    to its intrinsic value. Isolates the terminal and financing legs from the hedge leg."""
    pos, und, vol, rate = flat_setup([100.0] * 31, K=100.0, cp="C")
    res = delta_hedged_gain(pos, und, vol, rate)
    assert res.diagnostics["hedge_pnl"] == pytest.approx(0.0, abs=1e-12)
    assert res.diagnostics["payoff"] == pytest.approx(0.0)      # finishes exactly ATM
    assert res.pnl < 0                                           # long option, all decay


def test_terminal_value_is_intrinsic_not_a_market_mark():
    pos, und, vol, rate = flat_setup([100.0] * 30 + [112.0], K=100.0, cp="C")
    res = delta_hedged_gain(pos, und, vol, rate)
    assert res.diagnostics["payoff"] == pytest.approx(12.0)


def test_put_terminal_value_uses_the_put_payoff():
    pos, und, vol, rate = flat_setup([100.0] * 30 + [88.0], K=100.0, cp="P")
    res = delta_hedged_gain(pos, und, vol, rate)
    assert res.diagnostics["payoff"] == pytest.approx(12.0)


def test_cum_pnl_last_row_equals_the_scalar_pnl():
    """Checkpoint 4(b)'s hand-check, automated: the path and the scalar must agree."""
    rng = np.random.default_rng(1)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 31)))
    pos, und, vol, rate = flat_setup(path)
    res = delta_hedged_gain(pos, und, vol, rate)
    assert float(res.daily["cum_pnl"].iloc[-1]) == pytest.approx(res.pnl, abs=1e-10)


def test_pnl_decomposes_into_its_three_legs():
    rng = np.random.default_rng(2)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, 31)))
    pos, und, vol, rate = flat_setup(path)
    d = delta_hedged_gain(pos, und, vol, rate).diagnostics
    total = d["option_pnl"] + d["hedge_pnl"] + d["financing_pnl"]
    assert total == pytest.approx(delta_hedged_gain(pos, und, vol, rate).pnl, abs=1e-10)


def test_hedge_leg_has_the_right_sign():
    """Long a call, short delta shares: a rising stock must produce a hedge LOSS."""
    pos, und, vol, rate = flat_setup(list(np.linspace(100.0, 115.0, 31)), cp="C")
    assert delta_hedged_gain(pos, und, vol, rate).diagnostics["hedge_pnl"] < 0


def test_put_hedge_is_a_long_stock_position():
    """A long put has negative delta, so the hedge is LONG stock (M2) and a rising stock
    produces a hedge GAIN. Catches a sign error that would otherwise only show up in the
    aggregate."""
    pos, und, vol, rate = flat_setup(list(np.linspace(100.0, 115.0, 31)), cp="P")
    res = delta_hedged_gain(pos, und, vol, rate)
    assert res.daily["delta"].iloc[0] < 0
    assert res.diagnostics["hedge_pnl"] > 0


def test_financing_is_charged_on_the_net_position():
    """Long the option costs cash, short delta*S raises it; only the difference is financed.
    With a positive net investment and r > 0 the financing leg must be a cost."""
    pos, und, vol, rate = flat_setup([100.0] * 31, K=200.0, cp="C", entry_price=5.0)
    res = delta_hedged_gain(pos, und, vol, rate)
    assert res.daily["delta"].iloc[0] < 0.05          # deep OTM, almost no stock short
    assert res.diagnostics["financing_pnl"] < 0       # so the net position is a cash outlay


def test_zero_rate_removes_the_financing_leg_entirely():
    pos, und, vol, rate = flat_setup([100.0] * 31)
    res = delta_hedged_gain(pos, und, vol, pd.Series(0.0, index=rate.index))
    assert res.diagnostics["financing_pnl"] == pytest.approx(0.0, abs=1e-15)


def test_call_and_put_hedged_gains_agree_by_put_call_parity():
    """A delta-hedged call and a delta-hedged put at the same strike are the same position
    net of a bond, and the bond is exactly what the financing term charges. Their gains must
    therefore coincide -- a strong joint check on the delta signs and the financing leg."""
    rng = np.random.default_rng(4)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 31)))
    c = delta_hedged_gain(*flat_setup(path, cp="C")).pnl
    p = delta_hedged_gain(*flat_setup(path, cp="P")).pnl
    assert c == pytest.approx(p, abs=0.02)


# ---------------------------------------------------------------------------
# Hedge inputs and robustness hooks
# ---------------------------------------------------------------------------

def test_the_hedge_uses_physical_volatility_not_the_entry_price():
    """Feeding a different sigma must change the hedge. If it does not, the volatility input
    is not reaching the delta -- which is exactly what the Stage 4c placebo tests for."""
    path = list(100.0 + np.sin(np.linspace(0, 6, 31)) * 6.0)
    low = delta_hedged_gain(*flat_setup(path, sigma=0.15)).pnl
    high = delta_hedged_gain(*flat_setup(path, sigma=0.60)).pnl
    assert abs(low - high) > 1e-6


def test_delta_fn_is_injectable():
    """The placebo and the American-delta column must be one-line changes, not edits here."""
    pos, und, vol, rate = flat_setup([100.0] * 31)
    called = {"n": 0}

    def fake_delta(S, K, tau, sigma, r, q, cp):
        called["n"] += 1
        return 0.0                                     # an unhedged position

    res = delta_hedged_gain(pos, und, vol, rate, delta_fn=fake_delta)
    assert called["n"] == 31
    assert res.diagnostics["hedge_pnl"] == pytest.approx(0.0)


def test_escrowed_dividends_lower_the_call_delta():
    pos, und, vol, rate = flat_setup([100.0] * 31, cp="C")
    und_div = und.copy()
    und_div["pv_divs"] = 3.0
    base = delta_hedged_gain(pos, und, vol, rate).daily["delta"].iloc[0]
    with_div = delta_hedged_gain(pos, und_div, vol, rate).daily["delta"].iloc[0]
    assert with_div < base


def test_signed_quantity_flips_the_sign_for_a_short_position():
    """Phase 2 shorts options; the engine must already support it."""
    rng = np.random.default_rng(6)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 31)))
    pos, und, vol, rate = flat_setup(path)
    long_pnl = delta_hedged_gain(pos, und, vol, rate).pnl
    short = OptionPosition(**{**pos.__dict__, "quantity": -1.0})
    assert delta_hedged_gain(short, und, vol, rate).pnl == pytest.approx(-long_pnl)


# ---------------------------------------------------------------------------
# Robustness of the loop
# ---------------------------------------------------------------------------

def test_missing_volatility_carries_the_previous_delta_forward():
    """A halt must not silently unhedge the position or shorten it."""
    pos, und, vol, rate = flat_setup([100.0] * 31)
    vol = vol.copy()
    vol.iloc[5:8] = np.nan
    res = delta_hedged_gain(pos, und, vol, rate)
    assert res.daily["delta"].notna().all()
    assert res.diagnostics["n_rebalances"] == 30


def test_a_position_with_too_few_price_rows_raises_rather_than_returning_junk():
    pos, und, vol, rate = flat_setup([100.0, 101.0])
    one_row = und.iloc[:1]
    with pytest.raises(ValueError, match="price rows"):
        delta_hedged_gain(pos, one_row, vol, rate)


def test_run_positions_records_failures_instead_of_aborting():
    """One unpriceable position must not abort a 27,000-position run."""
    rng = np.random.default_rng(8)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 31)))
    good, und, vol, rate = flat_setup(path)
    bad = OptionPosition(**{**good.__dict__, "optionid": 2, "permno": 999})
    res, _ = run_positions([good, bad], {1: und}, {1: vol}, rate, progress=False)
    assert len(res) == 2
    assert res["error"].isna().sum() == 1
    assert res["error"].notna().sum() == 1


def test_run_positions_adds_the_papers_three_normalisations():
    rng = np.random.default_rng(9)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 31)))
    pos, und, vol, rate = flat_setup(path)
    res, paths = run_positions([pos], {1: und}, {1: vol}, rate, keep_paths=True, progress=False)
    assert {"pnl", "pnl_over_S", "pnl_over_C"} <= set(res.columns)
    entry_S = float(path[0])          # the path is already shocked at t=0
    assert res["pnl_over_S"].iloc[0] == pytest.approx(res["pnl"].iloc[0] / entry_S)
    assert 1 in paths and len(paths[1]) == 31
