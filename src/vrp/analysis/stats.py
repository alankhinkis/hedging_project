"""Stage 5 -- summary statistics and inference.

**The inference is where this extension diverges hardest from the paper, and where it is
most likely to overclaim** (Q4). BK have one underlying; we have ~240 that all load on the
same market volatility factor. Two positions in the same month are not independent draws, so
a naive pooled t-statistic over ~27,000 observations will produce something absurd. BK flag
the weaker version of this themselves (M15) and fix it by standardising each gain by its
Bertsimas-Kogan-Lo theoretical standard deviation.

Three flavours are reported, and the **conservative one leads**:

1. **Month-level portfolio** (primary): average pi/S across names within each entry month,
   giving 84 monthly observations, then a Newey-West t-stat. Effective N is 84, not 27,000.
2. **Two-way clustered** by permno and entry month.
3. **Naive pooled**, reported only for comparison and labelled overstated.

That ordering was fixed in the planning session, before any result existed (Q4). If (1) and
(3) differ by an order of magnitude -- they will -- that gap is itself a finding.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary table (mirrors BK Table 1, M13)
# ---------------------------------------------------------------------------

def summary_table(
    results: pd.DataFrame,
    selected: pd.DataFrame | None = None,
    *,
    by: str | None = None,
) -> pd.DataFrame:
    """BK Table 1's columns, optionally split by a grouping column.

    Reports all three normalisations (M13). **pi/S is the headline**: Lemma 1 / Eq. (19)
    prove pi scales with S, which makes it the theoretically motivated normalisation rather
    than a convenience (M14). pi/C travels alongside it but never alone -- the option price
    sits in the denominator and collapses for OTM options, which is why BK's full-sample
    pi/C (-12.18%) is three times their ATM figure.

    `selected` supplies the bid-ask spread for the M18 economic-significance comparison,
    which the plan insists belongs in the main table rather than an appendix: single-name
    spreads are wide enough that it is likely the binding constraint on whether any result
    here is tradable.
    """
    df = results.loc[results["error"].isna()].copy()
    if selected is not None and "rel_spread" in selected.columns:
        df = df.merge(
            selected[["optionid", "spread", "rel_spread"]].drop_duplicates("optionid"),
            on="optionid", how="left",
        )

    groups = [("all", df)] if by is None else list(df.groupby(by, observed=True))
    rows = []
    for key, g in groups:
        row = {
            "group": key,
            "N": int(len(g)),
            "mean_pnl": float(g["pnl"].mean()),
            "median_pnl": float(g["pnl"].median()),
            "mean_pnl_over_S_pct": float(g["pnl_over_S"].mean() * 100),
            "median_pnl_over_S_pct": float(g["pnl_over_S"].median() * 100),
            "mean_pnl_over_C_pct": float(g["pnl_over_C"].mean() * 100),
            "median_pnl_over_C_pct": float(g["pnl_over_C"].median() * 100),
            "frac_negative": float((g["pnl"] < 0).mean()),
        }
        if "spread" in g.columns:
            mean_spread = float(g["spread"].mean())
            row["mean_spread"] = mean_spread
            row["mean_abs_pnl"] = float(g["pnl"].abs().mean())
            # M18: BK compare a $0.43 mean loss against a $0.375 mean spread. The decision
            # rule is whether the loss clears HALF the spread -- the cost of crossing once.
            row["loss_over_half_spread"] = (
                abs(float(g["pnl"].mean())) / (0.5 * mean_spread) if mean_spread else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _nw_lag(n: int, requested: int | None) -> int:
    """Newey-West lag, capped so the HAC estimator stays defined on short samples."""
    if requested is not None:
        return max(0, min(int(requested), max(n - 2, 0)))
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def monthly_portfolio_tstat(
    results: pd.DataFrame,
    value: str = "pnl_over_S",
    *,
    nw_lags: int = 6,
) -> dict[str, float]:
    """Primary inference: equal-weighted monthly portfolio, Newey-West t-stat.

    Collapsing the cross-section to one observation per entry month is what makes the
    standard error honest. Within a month, every position shares the same market volatility
    shock, so the cross-section carries far less independent information than its row count
    suggests; the monthly series is the unit that is plausibly close to independent.
    """
    import statsmodels.api as sm

    df = results.loc[results["error"].isna()].copy()
    df["entry_month"] = pd.PeriodIndex(df["entry_date"], freq="M")
    monthly = df.groupby("entry_month")[value].mean().sort_index()

    y = monthly.to_numpy(dtype="float64")
    x = np.ones((len(y), 1))
    lags = _nw_lag(len(y), nw_lags)
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "method": "monthly portfolio (Newey-West)",
        "mean": float(fit.params[0]),
        "se": float(fit.bse[0]),
        "t": float(fit.tvalues[0]),
        "p": float(fit.pvalues[0]),
        "n_obs": int(len(y)),
        "nw_lags": lags,
        "frac_months_negative": float((monthly < 0).mean()),
    }


def clustered_tstat(
    results: pd.DataFrame,
    value: str = "pnl_over_S",
) -> dict[str, float]:
    """Secondary: two-way clustered by permno and entry month."""
    import statsmodels.api as sm

    df = results.loc[results["error"].isna()].copy()
    df["entry_month"] = pd.PeriodIndex(df["entry_date"], freq="M").astype(str)
    y = df[value].to_numpy(dtype="float64")
    x = np.ones((len(y), 1))
    # The SPX anchor has a single underlying and no permno column at all, so the "cluster by
    # name" dimension collapses to one group -- which is correct, not an error: BK's own
    # setting is exactly one underlying.
    g_permno = (pd.factorize(df["permno"])[0] if "permno" in df.columns
                else np.zeros(len(df), dtype="int64"))
    g_month = pd.factorize(df["entry_month"])[0]
    n_p, n_m = len(set(g_permno)), len(set(g_month))

    # Two-way clustering needs at least two groups in BOTH dimensions; statsmodels divides
    # by (n_groups - 1) and raises ZeroDivisionError otherwise. That never happens on the
    # full cross-section, but it does on a single-name debug subset, and crashing there
    # would be an obstacle to exactly the small-scale checking the build rule asks for.
    if n_p >= 2 and n_m >= 2:
        groups = np.column_stack([g_permno, g_month])
        method = "two-way clustered (permno, month)"
    elif n_m >= 2:
        groups, method = g_month, "one-way clustered (month) -- only one permno"
    elif n_p >= 2:
        groups, method = g_permno, "one-way clustered (permno) -- only one month"
    else:
        return {"method": "clustering not identified (one permno, one month)",
                "mean": float(np.mean(y)), "se": np.nan, "t": np.nan, "p": np.nan,
                "n_obs": int(len(y)), "n_permno": n_p, "n_month": n_m}

    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return {
        "method": method,
        "mean": float(fit.params[0]),
        "se": float(fit.bse[0]),
        "t": float(fit.tvalues[0]),
        "p": float(fit.pvalues[0]),
        "n_obs": int(len(y)),
        "n_permno": n_p,
        "n_month": n_m,
    }


def naive_tstat(results: pd.DataFrame, value: str = "pnl_over_S") -> dict[str, float]:
    """Naive pooled t-stat. **Overstated by construction** -- reported for comparison only.

    Included because the gap between this and the monthly figure is itself worth a paragraph:
    it quantifies how much of the apparent significance is an artefact of treating correlated
    positions as independent draws.
    """
    df = results.loc[results["error"].isna()]
    x = df[value].to_numpy(dtype="float64")
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return {
        "method": "naive pooled (OVERSTATED -- for comparison only)",
        "mean": mean, "se": se, "t": mean / se if se else np.nan,
        "p": np.nan, "n_obs": int(len(x)),
    }


def inference_table(results: pd.DataFrame, value: str = "pnl_over_S") -> pd.DataFrame:
    """All three flavours, conservative first (Q4)."""
    rows = [
        monthly_portfolio_tstat(results, value),
        clustered_tstat(results, value),
        naive_tstat(results, value),
    ]
    out = pd.DataFrame(rows)
    cols = ["method", "mean", "se", "t", "n_obs"]
    return out[cols + [c for c in out.columns if c not in cols]]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def outlier_robustness(results: pd.DataFrame, value: str = "pnl_over_S",
                       trim: float = 0.01) -> pd.DataFrame:
    """Does the result survive trimming the tails, and is the MEDIAN also negative?

    BK's Table 3 medians are more negative than their means; checking whether ours behave the
    same way is a cheap test of whether a handful of observations drive everything.
    """
    df = results.loc[results["error"].isna()]
    x = df[value]
    lo, hi = x.quantile(trim), x.quantile(1 - trim)
    trimmed = df.loc[x.between(lo, hi)]
    return pd.DataFrame([
        {"sample": "full", "N": len(df), "mean": x.mean(), "median": x.median()},
        {"sample": f"trimmed {trim:.0%}/{1 - trim:.0%}", "N": len(trimmed),
         "mean": trimmed[value].mean(), "median": trimmed[value].median()},
    ])


def by_side(results: pd.DataFrame, value: str = "pnl_over_S") -> pd.DataFrame:
    """Calls and puts separately -- two independent checks on the sign (Q1)."""
    df = results.loc[results["error"].isna()]
    rows = []
    for cp, g in df.groupby("cp_flag"):
        inf = monthly_portfolio_tstat(g, value)
        rows.append({"cp_flag": cp, "N": len(g), "mean": inf["mean"],
                     "t_monthly": inf["t"], "frac_negative": float((g["pnl"] < 0).mean())})
    return pd.DataFrame(rows)


def by_year(results: pd.DataFrame, value: str = "pnl_over_S") -> pd.DataFrame:
    """Per-year means. The plan predicts 2018 and 2020 more negative than 2017 and 2019."""
    df = results.loc[results["error"].isna()].copy()
    df["year"] = df["entry_date"].dt.year
    out = df.groupby("year").agg(
        N=(value, "size"), mean=(value, "mean"), median=(value, "median"),
        frac_negative=("pnl", lambda s: float((s < 0).mean())),
    ).reset_index()
    return out
