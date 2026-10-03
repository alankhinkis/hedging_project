"""Black-Scholes pricing and Greeks, with the escrowed-dividend adjustment.

The hedge ratio is a **Black-Scholes European delta evaluated at physical volatility**
(Q2, paper Eq. 30) -- not at implied volatility, and not from OptionMetrics' own binomial
delta. OM's delta is still used for *contract selection*, where an American model with
discrete dividends is the better instrument; it just must not leak into the hedge, since the
whole experiment is about what happens when you hedge at the physical rather than the
risk-neutral volatility.

Single-name options are American. The approximation is deliberate and bounded: at ~0.50 delta
and ~30 days with modest dividends the early-exercise premium is small, and
`crr_american_delta` is provided as the robustness column (Q2) rather than as the default.

**Dividends are escrowed, not continuous** (M9). The PV of dividends with ex-dates inside the
option's life is subtracted from the spot, and the European formula is applied to the
remainder. That is what BK do, and for single names it is also the only honest choice, since
IvyDB provides discrete dividends rather than a yield for equities (Q5).
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
from scipy.stats import norm

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x):
    """Standard normal CDF with a scalar fast path.

    `scipy.stats.norm.cdf` carries large per-call overhead (argument broadcasting and
    reduction machinery) that dwarfs the arithmetic for a single number. The hedging engine
    calls this once per rebalance day per position -- millions of scalar calls across a full
    run -- where it measured as ~36% of total runtime. `math.erf` is exact to double
    precision and roughly two orders of magnitude cheaper per scalar call; arrays still go
    to scipy.
    """
    a = np.asarray(x, dtype="float64")
    if a.ndim == 0:
        v = float(a)
        return np.float64(math.nan if math.isnan(v) else 0.5 * (1.0 + math.erf(v / _SQRT2)))
    return norm.cdf(a)


def _npdf(x):
    """Standard normal PDF, scalar fast path for the same reason as `_ncdf`."""
    a = np.asarray(x, dtype="float64")
    if a.ndim == 0:
        v = float(a)
        return np.float64(math.nan if math.isnan(v) else _INV_SQRT_2PI * math.exp(-0.5 * v * v))
    return norm.pdf(a)

__all__ = [
    "BSResult", "bs_price", "bs_delta", "bs_vega", "bs_greeks",
    "escrowed_spot", "crr_american_delta", "implied_vol",
]


def _d1_d2(S, K, tau, sigma, r, q=0.0):
    """Standard BS d1/d2 with a continuous yield `q`.

    Returns NaN where the inputs are degenerate (tau <= 0, sigma <= 0, S <= 0) rather than
    raising or silently producing inf -- an expired or zero-vol option has no d1, and the
    callers handle the boundary explicitly.
    """
    S, K, tau, sigma = map(lambda x: np.asarray(x, dtype="float64"), (S, K, tau, sigma))
    r = np.asarray(r, dtype="float64")
    q = np.asarray(q, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        vol_t = sigma * np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / vol_t
        d2 = d1 - vol_t
    bad = ~(np.isfinite(d1) & (tau > 0) & (sigma > 0) & (S > 0) & (K > 0))
    d1 = np.where(bad, np.nan, d1)
    d2 = np.where(bad, np.nan, d2)
    return d1, d2


@dataclass(frozen=True)
class BSResult:
    price: np.ndarray
    delta: np.ndarray
    vega: np.ndarray


def bs_price(S, K, tau, sigma, r, q=0.0, cp="C"):
    """Black-Scholes European price. At tau <= 0 returns intrinsic value, not NaN."""
    d1, d2 = _d1_d2(S, K, tau, sigma, r, q)
    S, K, tau = (np.asarray(x, dtype="float64") for x in (S, K, tau))
    r = np.asarray(r, dtype="float64")
    q = np.asarray(q, dtype="float64")
    is_call = _is_call(cp)

    disc_k = K * np.exp(-r * tau)
    disc_s = S * np.exp(-q * tau)
    call = disc_s * _ncdf(d1) - disc_k * _ncdf(d2)
    put = disc_k * _ncdf(-d2) - disc_s * _ncdf(-d1)
    price = np.where(is_call, call, put)

    # Expiry (or a degenerate input): the option is worth its intrinsic value.
    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    return np.where(np.isnan(d1), intrinsic, price)


def bs_delta(S, K, tau, sigma, r, q=0.0, cp="C"):
    """Black-Scholes European delta -- the hedge ratio (Eq. 30).

    Calls return N(d1) * e^{-q*tau} in [0, 1]; puts return a negative number in [-1, 0], so a
    "hedge" of -delta shares is a long stock position for a long put (M2). At expiry the
    delta is the step function, which is what the last rebalance should use.
    """
    d1, _ = _d1_d2(S, K, tau, sigma, r, q)
    S, K, tau = (np.asarray(x, dtype="float64") for x in (S, K, tau))
    q = np.asarray(q, dtype="float64")
    is_call = _is_call(cp)

    disc = np.exp(-q * tau)
    delta = np.where(is_call, disc * _ncdf(d1), disc * (_ncdf(d1) - 1.0))

    itm = np.where(is_call, S > K, S < K)
    terminal = np.where(is_call, np.where(itm, 1.0, 0.0), np.where(itm, -1.0, 0.0))
    return np.where(np.isnan(d1), terminal, delta)


def bs_vega(S, K, tau, sigma, r, q=0.0, cp="C"):
    """Vega per unit of volatility (not per percentage point). Zero at expiry.

    Sign matters for the hypothesis: vega is positive for both calls and puts, so under
    Proposition 1 a negative volatility risk premium produces a negative expected
    delta-hedged gain for both -- which is why calls and puts give independent checks on the
    sign (Q1).
    """
    d1, _ = _d1_d2(S, K, tau, sigma, r, q)
    S, tau = (np.asarray(x, dtype="float64") for x in (S, tau))
    q = np.asarray(q, dtype="float64")
    vega = S * np.exp(-q * tau) * _npdf(d1) * np.sqrt(tau)
    return np.where(np.isnan(d1), 0.0, vega)


def bs_greeks(S, K, tau, sigma, r, q=0.0, cp="C") -> BSResult:
    """Price, delta and vega in one pass -- the per-rebalance call in the hedging engine."""
    return BSResult(
        price=bs_price(S, K, tau, sigma, r, q, cp),
        delta=bs_delta(S, K, tau, sigma, r, q, cp),
        vega=bs_vega(S, K, tau, sigma, r, q, cp),
    )


def _is_call(cp):
    """Accept 'C'/'P' strings, arrays of them, or a boolean mask."""
    arr = np.asarray(cp)
    if arr.dtype == bool:
        return arr
    return np.char.upper(arr.astype("U1")) == "C"


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

def escrowed_spot(S, pv_dividends):
    """Spot less the PV of dividends paid during the option's life (M9).

    The European formula is then applied to this reduced spot, which is the standard
    "escrowed dividend" treatment: the dividend stream is removed from the diffusing part of
    the price. Clipped at zero -- a dividend PV exceeding the share price is a data error,
    not a negative stock.
    """
    S = np.asarray(S, dtype="float64")
    pv = np.nan_to_num(np.asarray(pv_dividends, dtype="float64"), nan=0.0)
    return np.maximum(S - pv, 0.0)


# ---------------------------------------------------------------------------
# American delta (Q2 robustness)
# ---------------------------------------------------------------------------

def crr_american_delta(S, K, tau, sigma, r, q=0.0, cp="C", steps: int = 200) -> float:
    """Cox-Ross-Rubinstein American delta, for the Q2 robustness column.

    Scalar-only and ~50x slower than the closed form, which is fine at this scale: it is run
    on a subsample, not on every rebalance of every position. The delta is read off the first
    step of the tree, which is the standard finite-difference estimate at t=0.
    """
    S, K, tau, sigma, r, q = map(float, (S, K, tau, sigma, r, q))
    is_call = str(cp).upper().startswith("C")
    if tau <= 0 or sigma <= 0 or S <= 0:
        itm = (S > K) if is_call else (S < K)
        return (1.0 if is_call else -1.0) if itm else 0.0

    dt = tau / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:  # tree is not risk-neutral at this step size
        return float(bs_delta(S, K, tau, sigma, r, q, cp))

    j = np.arange(steps + 1)
    prices = S * u**j * d ** (steps - j)
    values = np.maximum(prices - K, 0.0) if is_call else np.maximum(K - prices, 0.0)

    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        prices = S * u**j * d ** (i - j)
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        exercise = (prices - K) if is_call else (K - prices)
        values = np.maximum(values, exercise)   # American: exercise any time
        if i == 1:
            v_up, v_dn = values[1], values[0]
            s_up, s_dn = S * u, S * d
    return float((v_up - v_dn) / (s_up - s_dn))


# ---------------------------------------------------------------------------
# Implied volatility (for the Stage 4c placebo and diagnostics)
# ---------------------------------------------------------------------------

def implied_vol(price, S, K, tau, r, q=0.0, cp="C",
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-8) -> float:
    """Invert Black-Scholes for sigma by bisection.

    Bisection rather than Newton: vega collapses for deep in- or out-of-the-money options and
    a Newton step then diverges, whereas bisection on a bracketed monotone function cannot.
    Speed is irrelevant here -- this is a diagnostic, not an inner loop.
    """
    price, S, K, tau, r, q = map(float, (price, S, K, tau, r, q))
    if tau <= 0 or price <= 0:
        return float("nan")
    f = lambda s: float(bs_price(S, K, tau, s, r, q, cp)) - price  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:        # price outside the attainable range -> no solution
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
