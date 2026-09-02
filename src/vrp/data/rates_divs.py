"""Stage 2b supporting pulls -- the zero curve and the projected dividend schedule.

**Rates (Q6, a deliberate deviation).** BK back the interest rate out of put-call parity on
strike-and-maturity-matched pairs (M10). For 150 single names whose ~0.50-delta call and put
sit at *different* strikes that is noisy and slow, so we use `optionm.zerocd`, linearly
interpolated in days to each option's remaining maturity and updated daily (M4). This is
standard practice in the single-name literature and strictly cleaner data. IvyDB's zero curve
is already continuously compounded and quoted in percent, so the only transformation is /100.

**Dividends (M9, Q5).** No continuous yield exists for equities, so the escrowed-dividend
adjustment uses discrete dividends. The point-in-time projection file `distrprojd{YYYY}` --
which would have been ideal -- is listed on this WRDS subscription but **not queryable**: the
views resolve to missing `optionm_all` backing tables. So dividends come from `optionm.distrd`
(announced distributions), which carries `declare_date` on 100% of rows.

That declaration date makes the look-ahead question answerable rather than assumed.
`pv_dividends(..., known_only=True)` counts only dividends already declared as of the
valuation date; the default counts every dividend with an ex-date in the option's life, which
is what BK's escrowed treatment does (M9) and what the market prices for a predictable
quarterly payer. The two differ more than one might guess -- the median declaration lead is
15 days against a ~30-day hold -- so both are computed and the difference is reported rather
than waved away.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..schema import table_for
from ..wrds_conn import cached_query, get_connection, split_table

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zero curve
# ---------------------------------------------------------------------------

def fetch_zero_curve(
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
) -> pd.DataFrame:
    """The IvyDB zero curve over the sample, with `rate` converted to a decimal."""
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    schema, tbl = split_table(table_for("Zero-coupon curve", cfg))
    sql = f"""
        SELECT date, days, rate
        FROM   {schema}.{tbl}
        WHERE  date BETWEEN %(a)s AND %(b)s
    """
    df = cached_query(
        sql,
        {"a": str(cfg.burnin_start), "b": str(cfg.hold_buffer_end)},
        name="zerocd",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("date",),
    )
    df["days"] = df["days"].astype("float64")
    # IvyDB quotes the zero curve in percent, continuously compounded.
    df["r"] = df["rate"].astype("float64") / 100.0
    return df.sort_values(["date", "days"]).reset_index(drop=True)


def interpolate_rate(curve: pd.DataFrame, dates, days) -> np.ndarray:
    """Linearly interpolate the zero curve in `days`, per date (M4: updated daily).

    Outside the quoted tenor range the nearest point is used rather than extrapolating --
    our maturities (15-50 days) sit comfortably inside the curve, so this only guards the
    rare date with a truncated curve.
    """
    dates = pd.to_datetime(pd.Series(dates).values)
    days = np.asarray(days, dtype="float64")
    out = np.full(len(days), np.nan)

    by_date = {d: g for d, g in curve.groupby("date")}
    for i, (d, n) in enumerate(zip(dates, days)):
        g = by_date.get(d)
        if g is None or not len(g) or not np.isfinite(n):
            continue
        out[i] = float(np.interp(n, g["days"].values, g["r"].values))
    return out


def rate_path(curve: pd.DataFrame, dates, expiry) -> pd.Series:
    """Daily continuously-compounded rate matched to the *remaining* maturity each day."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    ttm = (pd.Timestamp(expiry) - dates).days.to_numpy(dtype="float64")
    return pd.Series(interpolate_rate(curve, dates, ttm), index=dates, name="r")


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

def fetch_dividends(
    secids: Sequence[int],
    cfg: Config | None = None,
    *,
    conn=None,
    force: bool = False,
    ordinary_only: bool = True,
) -> pd.DataFrame:
    """Announced cash distributions per secid over the sample.

    Columns: secid, declare_date, ex_date, amount, distr_type, frequency.

    `distr_type = '1'` is an ordinary cash dividend (96.4% of rows in 2017-2023); special
    distributions and stock dividends are excluded by default because the escrowed
    adjustment models the regular cash stream, and a special dividend usually triggers a
    contract adjustment on the option side anyway. Cancelled rows are always dropped.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    secids = tuple(int(s) for s in sorted(set(secids)))
    if not secids:
        return pd.DataFrame(
            columns=["secid", "declare_date", "ex_date", "amount", "distr_type", "frequency"]
        )

    schema, tbl = split_table(table_for("Projected dividends (discrete)", cfg))
    type_clause = "AND distr_type = '1'" if ordinary_only else ""
    sql = f"""
        SELECT secid, declare_date, ex_date, amount, distr_type, frequency, cancel_flag
        FROM   {schema}.{tbl}
        WHERE  secid IN %(secids)s
          AND  ex_date BETWEEN %(a)s AND %(b)s
          AND  (cancel_flag IS NULL OR cancel_flag = '0')
          {type_clause}
    """
    df = cached_query(
        sql,
        {"secids": secids, "a": str(cfg.start_date), "b": str(cfg.hold_buffer_end)},
        name="distrd",
        cfg=cfg,
        conn=conn,
        force=force,
        date_cols=("declare_date", "ex_date"),
    )
    if len(df):
        df["secid"] = df["secid"].astype("int64")
        df["amount"] = df["amount"].astype("float64")
    return df.sort_values(["secid", "ex_date"]).reset_index(drop=True)


def pv_dividends(
    divs: pd.DataFrame,
    secid: int,
    as_of,
    expiry,
    rate: float,
    *,
    known_only: bool = False,
) -> float:
    """PV of dividends with ex-dates inside (as_of, expiry], discounted at `rate` (M9).

    The escrowed-dividend adjustment: subtract this from the spot before pricing, so a
    European formula prices the diffusion of the ex-dividend price process.

    `known_only=True` restricts to dividends already declared on `as_of`, which removes
    look-ahead at the cost of missing dividends the market was already pricing. The default
    (False) matches BK's escrowed treatment and the market's own expectation for a regular
    quarterly payer.
    """
    as_of, expiry = pd.Timestamp(as_of), pd.Timestamp(expiry)
    if not len(divs):
        return 0.0
    sched = divs.loc[
        (divs["secid"] == int(secid))
        & (divs["ex_date"] > as_of)
        & (divs["ex_date"] <= expiry)
    ]
    if known_only:
        sched = sched.loc[sched["declare_date"].notna() & (sched["declare_date"] <= as_of)]
    if not len(sched):
        return 0.0
    tau = (sched["ex_date"] - as_of).dt.days.to_numpy(dtype="float64") / 365.0
    return float(np.sum(sched["amount"].to_numpy(dtype="float64") * np.exp(-rate * tau)))


def continuous_yield(pv_div: float, spot: float, tau_years: float) -> float:
    """The continuous `q` equivalent to an escrowed PV of dividends (Q5).

        q = -ln(1 - PVD/S) / tau

    Exposed so a Black-Scholes routine expecting a yield can be fed the same economics as
    the escrowed adjustment. Returns 0 for degenerate inputs rather than raising.
    """
    if spot <= 0 or tau_years <= 0 or not np.isfinite(pv_div):
        return 0.0
    frac = pv_div / spot
    if frac <= 0:
        return 0.0
    if frac >= 1:
        return float("nan")
    return float(-np.log(1.0 - frac) / tau_years)
