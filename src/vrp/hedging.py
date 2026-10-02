"""Stage 4 -- the delta-hedging engine.

This is the reusable core: one position-level function, and everything else is a loop over
it. The P&L follows BK's Sec. 3 definition exactly (M1, M2):

    pi = max(S_T - K, 0) - C_t - sum_n Delta_n (S_{n+1} - S_n)
                               - sum_n r_n (C_t - Delta_n S_n) (tau/N)

Three things in that formula are easy to get subtly wrong, and each is called out at the
point it happens below:

* **The terminal value is intrinsic, not a market mark** (M1). The position is held to
  expiration; there is no closing trade.
* **The hedge delta uses PHYSICAL volatility** (M5, Eq. 30), never implied. Hedging at
  implied volatility would partially hedge away the very premium being measured -- which is
  exactly why that variant is the Stage 4c placebo rather than the specification.
* **The financing term is on the net position** `(C_t - Delta_n S_n)`: long the option costs
  money, short Delta shares raises it, and only the difference is financed. It is accrued at
  the daily rate r_n, updated every day (M4).

Three design properties earn their keep in Phase 2 at no cost now: signed `quantity` (Phase 2
shorts options), the full daily path rather than a scalar (Phase 2 needs mark-to-market by
date), and an injectable `delta_fn` (which is what makes the placebo and the American-delta
robustness column one-line changes rather than edits to this file).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd

from .pricing import bs_delta, escrowed_spot

log = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class OptionPosition:
    """One delta-hedged option position, held from entry to expiry."""

    secid: int
    permno: int
    optionid: int
    cp_flag: str                  # 'C' | 'P'
    strike: float
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    entry_price: float            # bid-ask midpoint at entry
    quantity: float = 1.0         # signed: +1 long, -1 short (Phase 2 needs shorts)

    @property
    def is_call(self) -> bool:
        return str(self.cp_flag).upper().startswith("C")


@dataclass
class HedgeResult:
    pnl: float                    # scalar pi
    daily: pd.DataFrame           # the full path -- Phase 2 marks to market off this
    diagnostics: dict[str, Any] = field(default_factory=dict)


def delta_hedged_gain(
    position: OptionPosition,
    underlying: pd.DataFrame,          # date, S, pv_divs
    vol_path: pd.Series,               # date -> physical sigma
    rate_path: pd.Series,              # date -> continuously-compounded r
    *,
    delta_fn: Callable = bs_delta,
    terminal: str = "intrinsic",       # "intrinsic" (paper) | "mid" (mark-to-market)
    option_marks: pd.Series | None = None,   # date -> option mid, only for terminal="mid"
) -> HedgeResult:
    """Delta-hedged gain for one position, rebalanced daily at the close (M3).

    `underlying` must already be normalised to the entry-date adjustment factor (Q11), so S
    and the strike are in the same share units for the whole life of the position. Passing a
    split-adjusted series against an as-traded strike is the bug that silently manufactures a
    spectacular result; the engine cannot detect it, so Stage 5 must not build the input
    carelessly.

    A missing stock day inside the hold (halt, suspension) carries the previous delta forward
    rather than rebalancing, and is counted in `diagnostics` -- dropping the day would
    silently shorten the position, and interpolating would invent a price that never traded.
    """
    px = underlying.loc[
        (underlying["date"] >= position.entry_date) & (underlying["date"] <= position.expiry)
    ].sort_values("date").reset_index(drop=True)
    if len(px) < 2:
        raise ValueError(
            f"position {position.optionid} has {len(px)} price rows between "
            f"{position.entry_date.date()} and {position.expiry.date()}; need at least 2"
        )

    dates = pd.DatetimeIndex(px["date"])
    S = px["S"].to_numpy(dtype="float64")
    pv_divs = (px["pv_divs"].to_numpy(dtype="float64")
               if "pv_divs" in px.columns else np.zeros(len(px)))
    sigma = vol_path.reindex(dates).to_numpy(dtype="float64")
    r = rate_path.reindex(dates).to_numpy(dtype="float64")

    # Carry the last good value forward rather than dropping the day (see docstring).
    sigma = pd.Series(sigma).ffill().to_numpy()
    r = pd.Series(r).ffill().to_numpy()
    missing_vol = int(np.isnan(sigma).sum())
    missing_rate = int(np.isnan(r).sum())
    missing_px = int(np.isnan(S).sum())

    # Fractional day-count throughout. Truncating to whole days (datetime64[D]) is fine for
    # a daily grid but silently collapses any sub-daily step to zero, which breaks the
    # rebalancing-frequency leg of Checkpoint 4(a).
    secs_per_year = DAYS_PER_YEAR * 86400.0
    tau = np.maximum(
        (position.expiry - dates).total_seconds().to_numpy(dtype="float64"), 0.0
    ) / secs_per_year
    K = float(position.strike)

    # Escrowed dividends (M9): the European formula prices the ex-dividend diffusion, so the
    # PV of dividends falling inside the remaining life is removed from the spot first.
    S_escrow = escrowed_spot(S, pv_divs)

    n = len(px)
    delta = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(S_escrow[i]) or not np.isfinite(sigma[i]) or not np.isfinite(r[i]):
            continue
        delta[i] = float(np.asarray(
            delta_fn(S_escrow[i], K, tau[i], sigma[i], r[i], 0.0, position.cp_flag)
        ))
    # A day we could not price keeps yesterday's hedge rather than going unhedged.
    delta = pd.Series(delta).ffill().to_numpy()
    carried = int(np.isnan(delta).sum())

    # --- the two sums, accumulated over rebalance intervals -----------------
    # Interval i runs from date i to date i+1, using the delta set at the CLOSE of date i.
    # That ordering is what makes the strategy implementable: nothing uses S_{i+1}.
    dS = np.diff(S)
    d_hedge = -delta[:-1] * dS                       # short delta shares against a long option

    dt = np.diff(dates.to_numpy()).astype("timedelta64[s]").astype("float64") / secs_per_year
    net_investment = float(position.entry_price) - delta[:-1] * S[:-1]
    d_financing = -r[:-1] * net_investment * dt

    # --- terminal value ------------------------------------------------------
    S_T = S[-1]
    if terminal == "intrinsic":
        payoff = max(S_T - K, 0.0) if position.is_call else max(K - S_T, 0.0)
    elif terminal == "mid":
        if option_marks is None:
            raise ValueError("terminal='mid' requires option_marks")
        payoff = float(option_marks.reindex([dates[-1]]).iloc[0])
    else:
        raise ValueError(f"unknown terminal {terminal!r}")

    option_pnl = payoff - float(position.entry_price)
    pnl = float(position.quantity) * (option_pnl + np.nansum(d_hedge) + np.nansum(d_financing))

    cum = np.concatenate([[0.0], np.nancumsum(d_hedge + d_financing)])
    daily = pd.DataFrame({
        "date": dates,
        "S": S,
        "S_escrow": S_escrow,
        "tau": tau,
        "sigma_hat": sigma,
        "r": r,
        "delta": delta,
        "d_hedge_pnl": np.concatenate([d_hedge, [np.nan]]),
        "d_financing": np.concatenate([d_financing, [np.nan]]),
        "cum_hedge_pnl": cum,
    })
    # cum_pnl ties to the scalar on the LAST row (asserted in Checkpoint 4(b)), but the
    # intermediate rows are not a mark-to-market: they carry the option at zero, so each one
    # reads "cash paid for the option plus hedge P&L so far". The jump on the final row is
    # the payoff arriving, not an error. A true daily mark needs the Pass B option quotes,
    # which Phase 2 will join onto this path.
    daily["cum_pnl"] = position.quantity * (cum + np.where(
        daily["date"] == dates[-1], option_pnl, -float(position.entry_price)
    ))

    diagnostics = {
        "optionid": position.optionid,
        "n_rebalances": int(n - 1),
        "n_days": int(n),
        "missing_price_days": missing_px,
        "missing_vol_days": missing_vol,
        "missing_rate_days": missing_rate,
        "delta_carried_days": carried,
        "entry_delta": float(delta[0]) if np.isfinite(delta[0]) else np.nan,
        "terminal_delta": float(delta[-1]) if np.isfinite(delta[-1]) else np.nan,
        "entry_S": float(S[0]),
        "terminal_S": float(S_T),
        "terminal_moneyness": float(S_T / K) if K else np.nan,
        "payoff": float(payoff),
        "entry_price": float(position.entry_price),
        "option_pnl": float(option_pnl),
        "hedge_pnl": float(np.nansum(d_hedge)),
        "financing_pnl": float(np.nansum(d_financing)),
        "finished_itm": bool(payoff > 0),
    }
    return HedgeResult(pnl=pnl, daily=daily, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def run_positions(
    positions: list[OptionPosition],
    underlying_by_permno: dict[int, pd.DataFrame],
    vol_by_permno: dict[int, pd.Series],
    rate_path: pd.Series,
    *,
    delta_fn: Callable = bs_delta,
    keep_paths: bool = False,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Run the engine over many positions. Returns (results, optional paths by optionid).

    Failures are recorded with a reason rather than raised: one unpriceable position must not
    abort a 27,000-position run, and the failure count is itself a reportable number.
    """
    rows, paths = [], {}
    it = positions
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(positions, desc="hedging", unit="pos")
        except ImportError:
            pass

    for pos in it:
        try:
            und = underlying_by_permno[pos.permno]
            vol = vol_by_permno[pos.permno]
            res = delta_hedged_gain(pos, und, vol, rate_path, delta_fn=delta_fn)
        except Exception as exc:  # noqa: BLE001
            rows.append({"optionid": pos.optionid, "permno": pos.permno, "secid": pos.secid,
                         "cp_flag": pos.cp_flag, "entry_date": pos.entry_date,
                         "pnl": np.nan, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row = {
            "optionid": pos.optionid, "permno": pos.permno, "secid": pos.secid,
            "cp_flag": pos.cp_flag, "strike": pos.strike,
            "entry_date": pos.entry_date, "expiry": pos.expiry,
            "pnl": res.pnl, "error": None,
        }
        row.update(res.diagnostics)
        rows.append(row)
        if keep_paths:
            paths[pos.optionid] = res.daily

    out = pd.DataFrame(rows)
    if len(out):
        # The paper's three normalisations (M13): dollar pi, pi/S and pi/C.
        out["pnl_over_S"] = out["pnl"] / out["entry_S"]
        out["pnl_over_C"] = out["pnl"] / out["entry_price"]
    return out, paths


# ---------------------------------------------------------------------------
# Checkpoint 4(a) -- the zero-VRP synthetic test
# ---------------------------------------------------------------------------

def simulate_zero_vrp(
    n_paths: int = 2000,
    n_steps: int = 30,
    S0: float = 100.0,
    sigma: float = 0.30,
    r: float = 0.02,
    tau_days: int = 30,
    cp: str = "C",
    seed: int = 0,
    delta_fn: Callable = bs_delta,
) -> pd.DataFrame:
    """Proposition 1 in simulation: price at the SAME sigma the stock diffuses at, hedge
    daily, and the mean delta-hedged gain must be zero up to O(1/N).

    This is the single most important validation in the project. A sign error, a financing
    error or an off-by-one in the rebalance loop all show up here as a spuriously non-zero
    mean, in a world where the paper proves the answer is zero -- so it isolates engine bugs
    from anything about the data.

    Deliberately self-contained: it builds its own prices, so it tests the P&L arithmetic
    without depending on CRSP, OptionMetrics, or the Stage 5 assembly.
    """
    from .pricing import bs_price

    rng = np.random.default_rng(seed)
    tau = tau_days / DAYS_PER_YEAR
    dt = tau / n_steps
    # The grid must span exactly tau_days of CALENDAR time, otherwise the variance the paths
    # accumulate and the tau the engine reads off the dates disagree -- and the disagreement
    # grows with n_steps, which looks exactly like an engine bug. Business days would span
    # 14/42/84 calendar days for n_steps of 10/30/60 against a fixed 30-day tau.
    start = pd.Timestamp("2020-01-01")
    dates = pd.DatetimeIndex([
        start + pd.Timedelta(seconds=i * tau_days * 86400.0 / n_steps)
        for i in range(n_steps + 1)
    ])
    expiry = dates[-1]

    # Risk-neutral drift: under Proposition 1 the hedged gain is zero when the option is
    # priced at the same volatility the underlying diffuses at and there is no volatility
    # risk premium. The drift is r so the discounted price is a martingale.
    shocks = rng.normal(size=(n_paths, n_steps))
    log_paths = np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), S0 * np.exp(log_paths)])

    K = S0
    entry_price = float(bs_price(S0, K, tau, sigma, r, 0.0, cp))
    rate_path = pd.Series(r, index=dates)
    vol_path = pd.Series(sigma, index=dates)

    rows = []
    for i in range(n_paths):
        und = pd.DataFrame({"date": dates, "S": S[i], "pv_divs": 0.0})
        pos = OptionPosition(secid=0, permno=0, optionid=i, cp_flag=cp, strike=K,
                             entry_date=dates[0], expiry=expiry, entry_price=entry_price)
        res = delta_hedged_gain(pos, und, vol_path, rate_path, delta_fn=delta_fn)
        rows.append({"path": i, "pnl": res.pnl, "pnl_over_S": res.pnl / S0,
                     "terminal_S": S[i, -1]})
    return pd.DataFrame(rows)


def checkpoint_4a(n_paths: int = 2000, seed: int = 0) -> list[tuple[str, bool, str]]:
    """Checkpoint 4(a): the zero-VRP synthetic test, for calls and puts, at two frequencies."""
    results: list[tuple[str, bool, str]] = []

    for cp, label in (("C", "calls"), ("P", "puts")):
        sim = simulate_zero_vrp(n_paths=n_paths, n_steps=30, cp=cp, seed=seed)
        mean = float(sim["pnl"].mean())
        se = float(sim["pnl"].std(ddof=1) / np.sqrt(len(sim)))
        t = mean / se if se > 0 else np.nan
        results.append((
            f"zero-VRP synthetic: mean pi indistinguishable from zero ({label})",
            abs(t) < 2.0,
            f"mean {mean:+.4f} (pi/S {mean / 100:+.4%}), se {se:.4f}, t = {t:+.2f}",
        ))

    # The plan asks to "confirm the mean shrinks as you increase rebalancing frequency".
    # In practice the discretisation BIAS is far smaller than the Monte Carlo noise in the
    # mean at any feasible number of paths -- at 600 paths the means bounce around 0.00-0.04
    # with no visible trend -- so a mean-based comparison tests sampling luck, not the
    # engine.
    #
    # The sharp version of the same property is the DISPERSION. Boyle & Emanuel (1980): the
    # standard deviation of discrete delta-hedging error is O(1/sqrt(N)). That is a large,
    # low-noise signal, and it is just as diagnostic -- an engine that mis-accumulates the
    # hedge or financing legs does not produce textbook 1/sqrt(N) convergence.
    n_lo, n_hi = 10, 40
    lo = simulate_zero_vrp(n_paths=n_paths, n_steps=n_lo, cp="C", seed=seed + 1)
    hi = simulate_zero_vrp(n_paths=n_paths, n_steps=n_hi, cp="C", seed=seed + 1)
    s_lo = float(lo["pnl"].std(ddof=1))
    s_hi = float(hi["pnl"].std(ddof=1))
    ratio = s_lo / s_hi if s_hi > 0 else np.nan
    expected = np.sqrt(n_hi / n_lo)
    results.append((
        "hedging-error dispersion falls as 1/sqrt(N) (Boyle-Emanuel)",
        bool(0.75 * expected <= ratio <= 1.35 * expected),
        f"std(pi) {s_lo:.4f} at N={n_lo} -> {s_hi:.4f} at N={n_hi}; "
        f"ratio {ratio:.2f} vs sqrt({n_hi}/{n_lo}) = {expected:.2f}",
    ))

    # The mean is still reported at the finer grid, as a second look at Proposition 1.
    m_hi = float(hi["pnl"].mean())
    se_hi = float(hi["pnl"].std(ddof=1) / np.sqrt(len(hi)))
    t_hi = m_hi / se_hi if se_hi > 0 else np.nan
    results.append((
        f"mean pi still indistinguishable from zero at N={n_hi}",
        abs(t_hi) < 2.0,
        f"mean {m_hi:+.4f}, t = {t_hi:+.2f}",
    ))
    return results
