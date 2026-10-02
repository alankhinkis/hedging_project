"""Stage 5 -- volatility-regime splits and the Eq. (33) regressions.

Two regime questions, and only one of them exists in the paper:

* **Time-series** (BK Table 3, M16): are hedged losses worse in high-volatility *markets*?
  Buckets entry months by a market-wide volatility level. This is the direct analogue of the
  paper's test.
* **Cross-sectional**: do high-volatility *stocks* lose more, within a given month? This has
  no analogue in BK because BK have one underlying. It is the genuinely new question the
  extension can ask, and it is flagged as such rather than presented as replication.

The market volatility proxy is the cross-sectional median VOL^h on each entry date, computed
from the panel itself. The plan specifies VIX; VIX is not in the cached data and pulling it
needs WRDS. The proxy is highly correlated with VIX by construction (it is the median
realised volatility of 150 large caps) and the substitution is recorded here so it can be
swapped for the real series later.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime construction
# ---------------------------------------------------------------------------

def market_vol_proxy(vol_panel: pd.DataFrame, col: str = "vol_h") -> pd.Series:
    """Cross-sectional median volatility by date -- the VIX stand-in (see module docstring)."""
    return vol_panel.groupby("date")[col].median().sort_index()


def attach_entry_vol(
    results: pd.DataFrame,
    vol_panel: pd.DataFrame,
    col: str = "vol_h",
) -> pd.DataFrame:
    """Attach each position's own entry-date volatility and the market proxy."""
    df = results.loc[results["error"].isna()].copy()
    own = vol_panel[["permno", "date", col]].rename(
        columns={"date": "entry_date", col: "own_vol"}
    )
    df = df.merge(own, on=["permno", "entry_date"], how="left")
    mkt = market_vol_proxy(vol_panel, col).rename("mkt_vol")
    df = df.merge(mkt, left_on="entry_date", right_index=True, how="left")
    df["entry_month"] = pd.PeriodIndex(df["entry_date"], freq="M")
    return df


def regime_table(
    df: pd.DataFrame,
    vol_col: str,
    value: str = "pnl_over_S",
    *,
    n_buckets: int = 7,
    within_month: bool = False,
) -> pd.DataFrame:
    """Mean AND median gain per volatility bucket (M16).

    BK report both because their Table 3 medians are more negative than their means --
    whether ours behave the same way says how much of any result is tail-driven.

    `within_month=True` ranks each stock against its contemporaries rather than against the
    whole sample, which is what isolates the cross-sectional question from the time-series
    one: without it, "high own-volatility" would largely mean "March 2020".
    """
    d = df.dropna(subset=[vol_col, value]).copy()
    if not len(d):
        return pd.DataFrame()

    if within_month:
        d["bucket"] = d.groupby("entry_month")[vol_col].transform(
            lambda s: pd.qcut(s.rank(method="first"), min(n_buckets, max(s.nunique(), 1)),
                              labels=False, duplicates="drop") + 1
            if s.notna().sum() >= n_buckets else np.nan
        )
    else:
        d["bucket"] = pd.qcut(d[vol_col].rank(method="first"), n_buckets,
                              labels=False, duplicates="drop") + 1

    out = d.dropna(subset=["bucket"]).groupby("bucket").agg(
        N=(value, "size"),
        mean_vol=(vol_col, "mean"),
        mean=(value, "mean"),
        median=(value, "median"),
        frac_negative=("pnl", lambda s: float((s < 0).mean())),
    ).reset_index()
    out["mean_pct"] = out["mean"] * 100
    out["median_pct"] = out["median"] * 100
    return out


# ---------------------------------------------------------------------------
# Eq. (33) regressions (M17)
# ---------------------------------------------------------------------------

def timeseries_regression(
    df: pd.DataFrame,
    vol_col: str = "mkt_vol",
    value: str = "pnl_over_S",
    *,
    nw_lags: int = 12,
) -> pd.DataFrame:
    """BK Eq. (33) on the monthly cross-sectional average, matching their specification.

        GAINS_t = W0 + W1 * VOL_t + W2 * GAINS_{t-1} + e_t

    OLS with Newey-West standard errors at lag 12, and the lagged dependent variable
    included to soak up residual autocorrelation. Null is W1 = 0; the prediction is W1 < 0.
    This is the one number directly comparable to the paper's -0.032.
    """
    import statsmodels.api as sm

    monthly = df.groupby("entry_month").agg(
        gains=(value, "mean"), vol=(vol_col, "mean")
    ).sort_index()
    monthly["gains_lag"] = monthly["gains"].shift(1)
    monthly = monthly.dropna()
    if len(monthly) < 10:
        return pd.DataFrame()

    X = sm.add_constant(monthly[["vol", "gains_lag"]].to_numpy(dtype="float64"))
    y = monthly["gains"].to_numpy(dtype="float64")
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(nw_lags, len(y) - 2)})
    names = ["const (W0)", f"{vol_col} (W1)", "gains_lag (W2)"]
    return pd.DataFrame({
        "term": names,
        "coef": fit.params,
        "se": fit.bse,
        "t": fit.tvalues,
        "p": fit.pvalues,
    }).assign(n_obs=len(y), spec="time series, Newey-West lag 12")


def panel_regression(
    df: pd.DataFrame,
    vol_col: str = "own_vol",
    value: str = "pnl_over_S",
    *,
    stock_fe: bool = True,
    include_return: bool = False,
) -> pd.DataFrame:
    """Panel version of Eq. (33), with stock fixed effects and month-clustered errors.

    `include_return=True` adds the contemporaneous underlying return over the hold, which is
    the mishedging check (M19): a positive coefficient means Black-Scholes **underhedges**,
    so pi is biased *upward* -- i.e. against finding a loss, which would strengthen rather
    than explain away a negative result.
    """
    import statsmodels.api as sm

    d = df.dropna(subset=[vol_col, value]).copy().sort_values(["permno", "entry_date"])
    d["gains_lag"] = d.groupby("permno")[value].shift(1)
    cols = [vol_col, "gains_lag"]

    if include_return:
        if "terminal_S" in d.columns and "entry_S" in d.columns:
            d["underlying_ret"] = d["terminal_S"] / d["entry_S"] - 1.0
            cols.append("underlying_ret")
        else:
            log.warning("cannot build underlying_ret; skipping the M19 term")

    d = d.dropna(subset=cols)
    if len(d) < 50:
        return pd.DataFrame()

    X = d[cols].to_numpy(dtype="float64")
    names = list(cols)
    if stock_fe:
        dummies = pd.get_dummies(d["permno"], drop_first=True, dtype="float64")
        X = np.column_stack([X, dummies.to_numpy()])
    X = sm.add_constant(X)
    y = d[value].to_numpy(dtype="float64")

    groups = pd.factorize(d["entry_month"].astype(str))[0]
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

    keep = 1 + len(names)
    return pd.DataFrame({
        "term": ["const"] + names,
        "coef": fit.params[:keep],
        "se": fit.bse[:keep],
        "t": fit.tvalues[:keep],
        "p": fit.pvalues[:keep],
    }).assign(
        n_obs=len(y), n_months=d["entry_month"].nunique(),
        spec=f"panel{' + stock FE' if stock_fe else ''}, clustered by month",
    )
