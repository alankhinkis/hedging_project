"""Stage 3 volatility tests.

The defining property of both estimators is that they are **backward** averages, not
forecasts (M7), and that they are causal: the value at date t uses no information after t.
A look-ahead here would leak directly into the hedge ratio and manufacture a result, so it
is tested explicitly rather than assumed from the formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrp.vol import (
    GarchParams,
    filter_variance,
    fit_garch,
    garch_vol,
    log_price_returns,
    realized_vol,
    simulate_garch_returns,
    simulate_gbm_returns,
)


def returns_frame(r, start="2019-01-01", permno=1) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(r))
    return pd.DataFrame({"permno": permno, "date": dates, "r": np.asarray(r, dtype="float64")})


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def test_returns_use_the_split_adjusted_series_not_the_raw_price():
    """A 4-for-1 split must not register as a -75% return."""
    prices = pd.DataFrame({
        "permno": 1,
        "date": pd.to_datetime(["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"]),
        "s_adj": [125.0, 125.0, 129.0, 134.0],       # continuous through the split
        "has_price": True,
    })
    r = log_price_returns(prices)
    assert len(r) == 3
    assert r["r"].abs().max() < 0.05


def test_halted_days_are_dropped_not_imputed_as_zero():
    """Imputing a zero return on a halt would bias volatility down exactly when it is
    highest, so the observation is dropped instead."""
    prices = pd.DataFrame({
        "permno": 1,
        "date": pd.bdate_range("2019-01-01", periods=4),
        "s_adj": [100.0, np.nan, 110.0, 111.0],
        "has_price": [True, False, True, True],
    })
    r = log_price_returns(prices)
    assert len(r) == 2
    assert 0.0 not in set(np.round(r["r"], 10))


# ---------------------------------------------------------------------------
# VOL^h
# ---------------------------------------------------------------------------

def test_realized_vol_recovers_a_known_sigma():
    sigma = 0.25
    df = returns_frame(simulate_gbm_returns(sigma, n=1500, seed=3))
    vol = realized_vol(df, 30)["vol_h"].dropna()
    assert abs(vol.median() - sigma) < 0.03


def test_realized_vol_window_is_calendar_days_not_observations():
    """A 30-CALENDAR-day window holds ~21 business days, per BK Sec. 5."""
    df = returns_frame(np.full(400, 0.01))
    out = realized_vol(df, 30)
    assert 19 <= out["n_obs"].iloc[-1] <= 23


def test_realized_vol_is_causal():
    """Appending future returns must not change any earlier value. Catches a centred or
    forward-shifted window, which would leak tomorrow's volatility into today's delta."""
    r = simulate_gbm_returns(0.3, n=300, seed=11)
    early = realized_vol(returns_frame(r[:200]), 30).set_index("date")["vol_h"]
    full = realized_vol(returns_frame(r), 30).set_index("date")["vol_h"]
    pd.testing.assert_series_equal(early, full.loc[early.index], check_names=False)


def test_realized_vol_reacts_to_a_volatility_regime_shift():
    calm = simulate_gbm_returns(0.15, n=200, seed=1)
    storm = simulate_gbm_returns(0.90, n=60, seed=2)
    out = realized_vol(returns_frame(np.concatenate([calm, storm])), 30).set_index("date")
    assert out["vol_h"].iloc[150] < 0.30
    assert out["vol_h"].iloc[-1] > 0.60


def test_short_windows_are_nulled_not_reported():
    """A standard deviation from three observations would feed straight into a hedge ratio."""
    df = returns_frame(simulate_gbm_returns(0.25, n=40, seed=5))
    out = realized_vol(df, 30, min_obs=10)
    assert out["vol_h"].iloc[:5].isna().all()
    assert out["vol_h"].iloc[-1] == pytest.approx(out["vol_h"].iloc[-1])  # later values exist


def test_annualisation_uses_252_not_365():
    """Daily sigma of 1% must annualise to ~15.9%, not ~19.1%."""
    df = returns_frame(np.tile([0.01, -0.01], 200))
    vol = realized_vol(df, 30)["vol_h"].dropna().iloc[-1]
    assert vol == pytest.approx(0.01 * np.sqrt(252), rel=0.02)


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

def test_garch_recovers_parameters_from_a_simulated_garch_series():
    omega, alpha, beta = 2e-6, 0.08, 0.90
    rng = np.random.default_rng(0)
    n = 3000
    r = np.zeros(n)
    v = omega / (1 - alpha - beta)
    for i in range(n):
        r[i] = rng.normal(0.0, np.sqrt(v))
        v = omega + alpha * r[i] ** 2 + beta * v
    p = fit_garch(r)
    assert p.converged
    assert p.alpha == pytest.approx(alpha, abs=0.05)
    assert p.beta == pytest.approx(beta, abs=0.07)


def test_garch_reports_non_convergence_instead_of_raising():
    """Stage 3's contract is to fall back to VOL^h and log it, never to drop a stock."""
    p = fit_garch(np.array([0.01, -0.01, 0.02]))
    assert not p.converged
    assert "observations" in p.note


def test_non_stationary_fit_is_rejected():
    """alpha + beta >= 1 has no finite unconditional variance, so the recursion cannot be
    seeded and the result would be meaningless rather than merely imprecise."""
    p = GarchParams(omega=1e-6, alpha=0.5, beta=0.6)
    assert not p.stationary
    assert np.isnan(p.long_run_var)
    assert np.isnan(filter_variance(np.zeros(10), p)).all()


def test_filter_variance_matches_the_recursion_by_hand():
    p = GarchParams(omega=1e-6, alpha=0.1, beta=0.85, mu=0.0)
    r = np.array([0.02, -0.01, 0.03])
    v = filter_variance(r, p)
    v0 = p.long_run_var
    assert v[0] == pytest.approx(v0)
    assert v[1] == pytest.approx(p.omega + p.alpha * 0.02**2 + p.beta * v0)
    assert v[2] == pytest.approx(p.omega + p.alpha * 0.01**2 + p.beta * v[1])


def test_filter_variance_is_causal():
    """sigma^2_t must depend only on returns strictly before t."""
    p = GarchParams(omega=1e-6, alpha=0.1, beta=0.85)
    a = np.array([0.01, -0.02, 0.015, 0.03])
    b = a.copy()
    b[-1] = 0.99                      # change only the LAST return
    assert np.allclose(filter_variance(a, p)[:-1], filter_variance(b, p)[:-1])


def test_fit_garch_rejects_a_fit_below_the_alpha_floor():
    """A GARCH whose ARCH term is switched off is a constant, and useless as a hedge ratio.
    fit_garch must report it as not-converged so the realised estimator takes over, rather
    than passing a flat series through.

    The floor is driven to 1.0 so the rejection path is exercised deterministically -- on
    real data whether the optimiser lands exactly on alpha = 0 depends on the sample (it did
    so for 26% of one-year single-name fits, but not for every low-volatility series).
    """
    r = simulate_garch_returns(0.25, n=1200, seed=9)
    assert fit_garch(r, min_alpha=0.001).converged
    rejected = fit_garch(r, min_alpha=1.0)
    assert not rejected.converged
    assert "degenerate" in rejected.note


def test_degeneracy_floor_is_tunable():
    p = GarchParams(omega=1e-6, alpha=0.005, beta=0.9)
    assert p.degenerate(min_alpha=0.01)
    assert not p.degenerate(min_alpha=0.001)


def test_seeding_survives_persistence_near_one():
    """With alpha+beta -> 1 the model's unconditional variance omega/(1-a-b) explodes; on the
    real panel it reached 49,020% annualised. Seeding at the training-sample variance keeps
    the recursion on a sane scale regardless."""
    p = GarchParams(omega=1e-9, alpha=0.02, beta=0.9799, train_var=(0.30**2) / 252)
    v = filter_variance(simulate_gbm_returns(0.30, n=300, seed=2), p)
    assert np.isfinite(v).all()
    assert 0.1 < np.sqrt(np.nanmedian(v) * 252) < 0.8


def test_garch_vol_recovers_a_known_sigma_and_logs_every_stock_year():
    df = returns_frame(simulate_garch_returns(0.25, n=1000, seed=4), start="2017-01-02")
    vol, fits = garch_vol(df, 30)
    assert set(fits.columns) >= {"permno", "year", "converged", "persistence", "burn_in"}
    assert len(fits) == df["date"].dt.year.nunique()
    # The first year has no trailing year to train on; it is recorded as burn-in, not as a
    # convergence failure, so the Checkpoint 3 failure rate is not polluted by it.
    assert fits["burn_in"].sum() == 1
    good = vol["vol_g"].dropna()
    assert len(good) > 0
    assert abs(good.median() - 0.25) < 0.08


def test_garch_vol_is_a_backward_average_not_a_forecast():
    """M7: Eq. (28) averages fitted variances over the TRAILING window. After a volatility
    collapse the estimate must stay elevated for a while, which a forecast would not."""
    storm = simulate_garch_returns(0.90, n=500, seed=6)
    calm = simulate_garch_returns(0.10, n=8, seed=7)
    df = returns_frame(np.concatenate([storm, calm]), start="2017-01-02")
    vol, _ = garch_vol(df, 30)
    series = vol.set_index("date")["vol_g"].dropna()
    # Eight calm days cannot undo a 30-day trailing window built on a storm. A forward
    # forecast would have collapsed toward 10% by now; a trailing average has not.
    assert series.iloc[-1] > 0.40
    # ...and the realised estimator, which is unambiguously backward, agrees.
    assert realized_vol(df, 30).set_index("date")["vol_h"].iloc[-1] > 0.40
