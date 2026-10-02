"""Black-Scholes tests.

Three classes of check: known published values, internal identities that must hold whatever
the implementation does (parity, bounds, monotonicity), and the boundary behaviour the
hedging engine depends on (expiry, zero volatility, sign conventions).
"""

from __future__ import annotations

import numpy as np
import pytest

from vrp.pricing import (
    bs_delta,
    bs_greeks,
    bs_price,
    bs_vega,
    crr_american_delta,
    escrowed_spot,
    implied_vol,
)


# ---------------------------------------------------------------------------
# Known values
# ---------------------------------------------------------------------------

def test_matches_a_textbook_value():
    """Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, tau=0.5
    gives a call of 4.76 and a put of 0.81."""
    args = dict(S=42.0, K=40.0, tau=0.5, sigma=0.20, r=0.10)
    assert float(bs_price(**args, cp="C")) == pytest.approx(4.76, abs=0.01)
    assert float(bs_price(**args, cp="P")) == pytest.approx(0.81, abs=0.01)


def test_atm_delta_is_near_half():
    d = float(bs_delta(S=100.0, K=100.0, tau=30 / 365, sigma=0.30, r=0.02, cp="C"))
    assert 0.50 < d < 0.56        # slightly above 0.5 from the drift term


def test_put_delta_is_negative_and_bounded():
    d = float(bs_delta(S=100.0, K=100.0, tau=30 / 365, sigma=0.30, r=0.02, cp="P"))
    assert -1.0 < d < 0.0


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def test_put_call_parity_holds():
    S, K, tau, sigma, r, q = 100.0, 95.0, 0.25, 0.28, 0.03, 0.01
    c = float(bs_price(S, K, tau, sigma, r, q, "C"))
    p = float(bs_price(S, K, tau, sigma, r, q, "P"))
    lhs = c - p
    rhs = S * np.exp(-q * tau) - K * np.exp(-r * tau)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_delta_parity_holds():
    """call_delta - put_delta = e^{-q*tau}. A sign error on either side breaks this."""
    S, K, tau, sigma, r, q = 100.0, 103.0, 0.1, 0.35, 0.02, 0.015
    dc = float(bs_delta(S, K, tau, sigma, r, q, "C"))
    dp = float(bs_delta(S, K, tau, sigma, r, q, "P"))
    assert dc - dp == pytest.approx(np.exp(-q * tau), abs=1e-12)


def test_vega_is_positive_and_identical_for_calls_and_puts():
    """Both are long volatility, which is why a negative premium shows up in both (Q1)."""
    args = dict(S=100.0, K=100.0, tau=0.25, sigma=0.3, r=0.02)
    vc = float(bs_vega(**args, cp="C"))
    vp = float(bs_vega(**args, cp="P"))
    assert vc > 0
    assert vc == pytest.approx(vp, rel=1e-12)


def test_price_respects_arbitrage_bounds():
    S, K, tau, sigma, r = 100.0, 90.0, 0.3, 0.4, 0.02
    c = float(bs_price(S, K, tau, sigma, r, 0.0, "C"))
    assert max(S - K * np.exp(-r * tau), 0.0) <= c <= S
    p = float(bs_price(S, K, tau, sigma, r, 0.0, "P"))
    assert max(K * np.exp(-r * tau) - S, 0.0) <= p <= K


def test_price_is_increasing_in_volatility():
    args = dict(S=100.0, K=100.0, tau=0.25, r=0.02)
    prices = [float(bs_price(**args, sigma=s, cp="C")) for s in (0.1, 0.2, 0.4, 0.8)]
    assert prices == sorted(prices)


def test_call_delta_is_increasing_in_spot():
    deltas = [float(bs_delta(S=s, K=100.0, tau=0.25, sigma=0.3, r=0.02, cp="C"))
              for s in (80.0, 95.0, 100.0, 110.0, 130.0)]
    assert deltas == sorted(deltas)
    assert 0.0 < deltas[0] < 0.2
    assert 0.9 < deltas[-1] <= 1.0


# ---------------------------------------------------------------------------
# Boundaries the hedging engine relies on
# ---------------------------------------------------------------------------

def test_at_expiry_price_is_intrinsic_not_nan():
    """The engine evaluates the last rebalance at tau = 0; NaN there would poison the P&L."""
    assert float(bs_price(110.0, 100.0, 0.0, 0.3, 0.02, 0.0, "C")) == pytest.approx(10.0)
    assert float(bs_price(90.0, 100.0, 0.0, 0.3, 0.02, 0.0, "C")) == pytest.approx(0.0)
    assert float(bs_price(90.0, 100.0, 0.0, 0.3, 0.02, 0.0, "P")) == pytest.approx(10.0)


def test_at_expiry_delta_is_the_step_function():
    assert float(bs_delta(110.0, 100.0, 0.0, 0.3, 0.02, 0.0, "C")) == 1.0
    assert float(bs_delta(90.0, 100.0, 0.0, 0.3, 0.02, 0.0, "C")) == 0.0
    assert float(bs_delta(90.0, 100.0, 0.0, 0.3, 0.02, 0.0, "P")) == -1.0


def test_vega_is_zero_at_expiry():
    assert float(bs_vega(100.0, 100.0, 0.0, 0.3, 0.02)) == 0.0


def test_zero_volatility_degrades_to_intrinsic():
    """A halted name can produce a zero vol estimate; it must not produce NaN."""
    assert float(bs_price(110.0, 100.0, 0.25, 0.0, 0.0, 0.0, "C")) == pytest.approx(10.0)
    assert np.isfinite(float(bs_delta(110.0, 100.0, 0.25, 0.0, 0.0, 0.0, "C")))


def test_vectorises_over_arrays_and_mixed_cp_flags():
    S = np.array([100.0, 100.0, 100.0])
    cp = np.array(["C", "P", "C"])
    d = bs_delta(S, 100.0, 0.25, 0.3, 0.02, 0.0, cp)
    assert d.shape == (3,)
    assert d[0] > 0 and d[1] < 0 and d[2] > 0


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

def test_escrowed_spot_subtracts_dividend_pv():
    assert float(escrowed_spot(100.0, 1.5)) == pytest.approx(98.5)


def test_escrowed_spot_treats_missing_dividends_as_zero():
    assert float(escrowed_spot(100.0, np.nan)) == pytest.approx(100.0)


def test_escrowed_spot_cannot_go_negative():
    """A dividend PV above the share price is a data error, not a negative stock."""
    assert float(escrowed_spot(10.0, 50.0)) == 0.0


def test_escrowed_dividend_lowers_a_call_and_raises_a_put():
    base = dict(K=100.0, tau=0.25, sigma=0.3, r=0.02)
    c0 = float(bs_price(S=100.0, **base, cp="C"))
    c1 = float(bs_price(S=float(escrowed_spot(100.0, 2.0)), **base, cp="C"))
    p0 = float(bs_price(S=100.0, **base, cp="P"))
    p1 = float(bs_price(S=float(escrowed_spot(100.0, 2.0)), **base, cp="P"))
    assert c1 < c0
    assert p1 > p0


# ---------------------------------------------------------------------------
# American delta and implied volatility
# ---------------------------------------------------------------------------

def test_american_call_without_dividends_matches_european():
    """With q = 0 an American call is never exercised early, so the deltas must agree."""
    args = dict(S=100.0, K=100.0, tau=0.25, sigma=0.3, r=0.02, q=0.0)
    a = crr_american_delta(**args, cp="C", steps=400)
    e = float(bs_delta(**args, cp="C"))
    assert a == pytest.approx(e, abs=0.01)


def test_american_put_delta_is_more_negative_than_european():
    """Early exercise is worth something for a put, so its delta is steeper."""
    args = dict(S=95.0, K=100.0, tau=0.5, sigma=0.3, r=0.08, q=0.0)
    a = crr_american_delta(**args, cp="P", steps=400)
    e = float(bs_delta(**args, cp="P"))
    assert a < e
    assert a >= -1.0


def test_implied_vol_round_trips():
    S, K, tau, r, sigma = 100.0, 105.0, 0.25, 0.02, 0.37
    price = float(bs_price(S, K, tau, sigma, r, 0.0, "C"))
    assert implied_vol(price, S, K, tau, r, 0.0, "C") == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_returns_nan_for_an_unattainable_price():
    """A price above the spot cannot be produced by any volatility; the inverter must say so
    rather than returning the bracket edge."""
    assert np.isnan(implied_vol(150.0, 100.0, 100.0, 0.25, 0.02, 0.0, "C"))


def test_bs_greeks_agrees_with_the_individual_functions():
    args = dict(S=101.0, K=100.0, tau=0.2, sigma=0.33, r=0.025, q=0.01, cp="P")
    g = bs_greeks(**args)
    assert float(g.price) == pytest.approx(float(bs_price(**args)))
    assert float(g.delta) == pytest.approx(float(bs_delta(**args)))
    assert float(g.vega) == pytest.approx(float(bs_vega(**args)))
