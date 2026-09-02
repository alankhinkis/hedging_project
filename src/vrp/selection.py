"""Stage 2b -- screens and contract selection.

One contract per (stock, month, cp_flag): the ~30-day, ~0.50-delta one. Two ideas govern the
implementation.

**Every screen reports what it dropped.** Coverage is a result, not a nuisance (Q9). If the
drop rate is 5%, ignore it; if it is 30% and concentrated in high-volatility months or smaller
names, the selection itself could produce the whole finding. So `apply_screens` returns an
audit trail -- rows in, rows out, and stock-months lost, per screen, in order.

**The implied-vol cap is a parameter, not a constant** (Q8). BK's 100% cap is stated as a
recording-error filter; on single names in March 2020 a 30-day ATM IV above 100% was a real
observation. The primary cap is 300% and the literal 100% is a robustness row, so the screen
takes the cap as an argument and the pipeline is run twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, load_config

log = logging.getLogger(__name__)


@dataclass
class ScreenLog:
    """Audit trail for the screen cascade -- the raw material for the Q9 coverage table."""

    rows: list[dict] = field(default_factory=list)

    def record(self, name: str, before: pd.DataFrame, after: pd.DataFrame, note: str = "") -> None:
        self.rows.append(
            {
                "screen": name,
                "rows_in": int(len(before)),
                "rows_out": int(len(after)),
                "rows_dropped": int(len(before) - len(after)),
                "stock_months_in": _stock_months(before),
                "stock_months_out": _stock_months(after),
                "note": note,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if len(df):
            df["stock_months_lost"] = df["stock_months_in"] - df["stock_months_out"]
            df["pct_rows_dropped"] = df["rows_dropped"] / df["rows_in"].replace(0, np.nan)
        return df


def _stock_months(df: pd.DataFrame) -> int:
    if not len(df) or "secid" not in df.columns:
        return 0
    return int(df.groupby(["secid", "date"]).ngroups)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def apply_screens(
    candidates: pd.DataFrame,
    cfg: Config | None = None,
    *,
    iv_cap: float | None = None,
    spot: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the data-quality and liquidity screens, in order, logging each one.

    `spot` is an optional (secid, date, S) frame enabling the arbitrage-bound screen (M11);
    without it that screen is skipped and the log says so.

    Returns (surviving candidates, screen log).
    """
    cfg = cfg or load_config()
    sel = cfg["selection"]
    screens = cfg["screens"]
    iv_cap = float(iv_cap if iv_cap is not None else screens["iv_cap_primary"])
    iv_floor = float(screens["iv_floor"])

    df = candidates.copy()
    slog = ScreenLog()
    slog.record("raw candidates", df, df, f"DTE {int(df['dte'].min())}-{int(df['dte'].max())}"
                if len(df) else "empty")

    def step(name: str, mask: pd.Series, note: str = "") -> None:
        nonlocal df
        before = df
        df = df.loc[mask.reindex(df.index, fill_value=False)]
        slog.record(name, before, df, note)

    # --- data quality -----------------------------------------------------
    step("standard settlement (ss_flag == '0')",
         df["ss_flag"].astype("string") == str(sel["ss_flag"]),
         "verified VARCHAR(1) in Stage 0, so the literal is quoted")
    step("standard deliverable (contract_size == 100)",
         df["contract_size"].astype("float64") == float(sel["contract_size"]),
         "drops adjusted/non-standard deliverables after corporate actions")

    # --- quotes -----------------------------------------------------------
    step("positive bid", df["best_bid"].astype("float64") > 0)
    step("offer strictly above bid",
         df["best_offer"].astype("float64") > df["best_bid"].astype("float64"),
         "excludes crossed and locked quotes")
    step("open interest > 0",
         df["open_interest"].astype("float64") > float(sel["min_open_interest"]) - 1e-9)

    # --- implied volatility (Q8) ------------------------------------------
    step("implied volatility present", df["impl_volatility"].notna())
    step(f"implied volatility in [{iv_floor:.0%}, {iv_cap:.0%}]",
         df["impl_volatility"].astype("float64").between(iv_floor, iv_cap),
         "BK's screen is a recording-error filter (Sec. 2 p. 538), not an economic one")

    # --- delta ------------------------------------------------------------
    step("delta present", df["delta"].notna())

    # --- arbitrage bounds (M11), only if a spot price is available ---------
    if spot is not None and len(spot):
        merged = df.merge(spot, on=["secid", "date"], how="left")
        merged.index = df.index
        S = merged["S"].astype("float64")
        K = df["strike"].astype("float64")
        mid = (df["best_bid"].astype("float64") + df["best_offer"].astype("float64")) / 2.0
        is_call = df["cp_flag"].astype("string") == "C"
        # Undiscounted bounds: max(S-K, 0) <= C <= S, and max(K-S, 0) <= P <= K. Discounting
        # would tighten them, but the point is to catch recording errors, not to price.
        ok_call = (mid >= np.maximum(S - K, 0.0) - 1e-9) & (mid <= S + 1e-9)
        ok_put = (mid >= np.maximum(K - S, 0.0) - 1e-9) & (mid <= K + 1e-9)
        step("arbitrage bounds on the mid quote",
             (is_call & ok_call) | (~is_call & ok_put) | S.isna(),
             "undiscounted; rows without a spot price are passed through")
    else:
        slog.record("arbitrage bounds on the mid quote", df, df, "SKIPPED -- no spot supplied")

    return df.reset_index(drop=True), slog.to_frame()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_contracts(
    candidates: pd.DataFrame,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """One contract per (secid, entry date, cp_flag), following the plan's rule exactly.

    1. Restrict to the DTE window (the pull is deliberately wider than the rule).
    2. Pick the expiration minimising |dte - 30|; ties go to the SHORTER one.
    3. Within that expiration, restrict to the delta band and pick the contract minimising
       |delta| - 0.50 in absolute value.
    4. Break remaining ties on higher open interest, then lower optionid for determinism.

    Step 2 resolves at the *expiration* level before step 3 looks at strikes, which matters:
    picking the globally-nearest-0.50-delta contract could land on a different expiry than
    the nearest-30-day one.
    """
    cfg = cfg or load_config()
    sel = cfg["selection"]
    lo, hi = float(sel["delta_band"][0]), float(sel["delta_band"][1])
    target = float(sel["delta_target"])

    df = candidates.copy()
    df = df.loc[df["dte"].between(int(sel["dte_min"]), int(sel["dte_max"]))]
    if not len(df):
        return df.assign(abs_delta=[], dte_gap=[])

    df["abs_delta"] = df["delta"].astype("float64").abs()
    df["dte_gap"] = (df["dte"].astype("float64") - float(sel["dte_target"])).abs()

    # Step 2: choose the expiration per (secid, date, cp_flag). Ties -> shorter dte.
    keys = ["secid", "date", "cp_flag"]
    exp_rank = (
        df.groupby(keys + ["exdate"], as_index=False)
        .agg(dte=("dte", "first"), dte_gap=("dte_gap", "first"))
        .sort_values(keys + ["dte_gap", "dte"], ascending=[True] * len(keys) + [True, True])
        .drop_duplicates(subset=keys, keep="first")[keys + ["exdate"]]
    )
    df = df.merge(exp_rank, on=keys + ["exdate"], how="inner")

    # Step 3: within that expiration, the delta band, then nearest to |0.50|.
    df = df.loc[df["abs_delta"].between(lo, hi)]
    if not len(df):
        return df

    df["delta_gap"] = (df["abs_delta"] - target).abs()
    picked = (
        df.sort_values(
            keys + ["delta_gap", "open_interest", "optionid"],
            ascending=[True] * len(keys) + [True, False, True],
        )
        .drop_duplicates(subset=keys, keep="first")
        .reset_index(drop=True)
    )
    picked["mid"] = (
        picked["best_bid"].astype("float64") + picked["best_offer"].astype("float64")
    ) / 2.0
    picked["spread"] = (
        picked["best_offer"].astype("float64") - picked["best_bid"].astype("float64")
    )
    picked["rel_spread"] = picked["spread"] / picked["mid"].replace(0, np.nan)
    return picked.rename(columns={"date": "entry_date"})


def parity_implied_forward(
    candidates: pd.DataFrame,
    spot: pd.DataFrame,
    curve: pd.DataFrame,
) -> pd.DataFrame:
    """Put-call parity on SAME-STRIKE pairs, as a units-and-wiring check.

    The plan proposes checking parity on the selected call and put, but those sit at
    different strikes (a ~0.50-delta call and a ~-0.50-delta put rarely share one), so a
    naive check is impossible. The candidate chain, however, is full of same-strike pairs.
    For each, European parity gives the forward directly:

        F = K + e^{r tau} (C - P)

    Comparing that against the spot carried forward, S e^{r tau}, catches a strike-scaling
    error (the /1000), a delta or cp_flag sign error, a rate-units error (percent vs decimal)
    and a broken secid->spot linkage, all at once and on tens of thousands of observations.

    The residual is not expected to be exactly zero: these are American options on
    dividend-paying stocks, so the early-exercise premium and the dividends both push the
    put-side value up and the implied forward down. The check is on the *magnitude* of the
    discrepancy -- basis points, not percent.
    """
    from .data.rates_divs import interpolate_rate

    df = candidates.copy()
    df["mid"] = (df["best_bid"].astype("float64") + df["best_offer"].astype("float64")) / 2.0
    keys = ["secid", "date", "exdate", "strike"]
    calls = df.loc[df["cp_flag"].astype("string") == "C", keys + ["mid"]].rename(columns={"mid": "C"})
    puts = df.loc[df["cp_flag"].astype("string") == "P", keys + ["mid"]].rename(columns={"mid": "P"})
    pairs = calls.merge(puts, on=keys, how="inner")
    if not len(pairs):
        return pairs

    pairs = pairs.merge(spot, on=["secid", "date"], how="inner")
    pairs["tau"] = (pairs["exdate"] - pairs["date"]).dt.days.astype("float64") / 365.0
    pairs["r"] = interpolate_rate(curve, pairs["date"], (pairs["exdate"] - pairs["date"]).dt.days)
    pairs = pairs.loc[pairs["r"].notna() & pairs["S"].notna() & (pairs["tau"] > 0)]

    pairs["F_parity"] = pairs["strike"] + np.exp(pairs["r"] * pairs["tau"]) * (pairs["C"] - pairs["P"])
    pairs["F_spot"] = pairs["S"] * np.exp(pairs["r"] * pairs["tau"])
    pairs["rel_error"] = pairs["F_parity"] / pairs["F_spot"] - 1.0
    return pairs


def checkpoint_2b(
    spec: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    paths: pd.DataFrame,
    spot: pd.DataFrame,
    curve: pd.DataFrame,
    cfg: Config | None = None,
) -> list[tuple[str, bool, str]]:
    """Checkpoint 2b from PHASE1_PLAN.md, as an explicit pass/fail list."""
    cfg = cfg or load_config()
    sel = cfg["selection"]
    results: list[tuple[str, bool, str]] = []

    # --- 1. one contract per stock-month-side, no duplicates ---------------
    dupes = int(selected.duplicated(subset=["secid", "entry_date", "cp_flag"]).sum())
    n_months = int(selected["entry_date"].nunique())
    results.append((
        "exactly one contract per (secid, entry date, cp_flag)",
        dupes == 0,
        f"{dupes} duplicates across {len(selected):,} positions over {n_months} entry dates",
    ))

    # --- 2. selected contracts obey the selection rule ---------------------
    dte_ok = selected["dte"].between(int(sel["dte_min"]), int(sel["dte_max"])).all()
    lo, hi = float(sel["delta_band"][0]), float(sel["delta_band"][1])
    band_ok = selected["abs_delta"].between(lo, hi).all()
    results.append((
        f"every contract has dte in [{sel['dte_min']}, {sel['dte_max']}] "
        f"and |delta| in [{lo}, {hi}]",
        bool(dte_ok and band_ok),
        f"dte {int(selected['dte'].min())}-{int(selected['dte'].max())}, "
        f"|delta| {selected['abs_delta'].min():.3f}-{selected['abs_delta'].max():.3f}",
    ))

    # --- 3. the selection is actually centred on 0.50 ----------------------
    med = float(selected["abs_delta"].median())
    med_c = float(selected.loc[selected["cp_flag"] == "C", "abs_delta"].median())
    med_p = float(selected.loc[selected["cp_flag"] == "P", "abs_delta"].median())
    results.append((
        "median |delta| within 0.02 of 0.50 (a sign or convention error shows up here)",
        abs(med - 0.50) <= 0.02,
        f"median |delta| {med:.4f} (calls {med_c:.4f}, puts {med_p:.4f})",
    ))

    # --- 4. strike within +-10% of spot ------------------------------------
    with_spot = selected.merge(spot, on=["secid", "date"], how="left") if "date" in selected.columns \
        else selected.merge(spot.rename(columns={"date": "entry_date"}), on=["secid", "entry_date"], how="left")
    moneyness = (with_spot["strike"].astype("float64") / with_spot["S"].astype("float64") - 1.0).abs()
    frac_out = float((moneyness > 0.10).mean())
    results.append((
        "selected strike within +-10% of spot",
        frac_out < 0.01,
        f"{frac_out:.2%} of contracts outside +-10%; median |K/S - 1| = {moneyness.median():.3%}",
    ))

    # --- 5. put-call parity on same-strike pairs ---------------------------
    pairs = parity_implied_forward(candidates, spot, curve)
    if not len(pairs):
        results.append(("put-call parity implies the spot forward", False, "no same-strike pairs"))
    else:
        med_bp = float(pairs["rel_error"].median() * 1e4)
        iqr_bp = float((pairs["rel_error"].quantile(0.75) - pairs["rel_error"].quantile(0.25)) * 1e4)
        results.append((
            "put-call parity implies the spot forward (units/wiring check)",
            abs(med_bp) < 100.0,
            f"{len(pairs):,} same-strike pairs; median error {med_bp:+.1f} bp, IQR {iqr_bp:.1f} bp",
        ))

    # --- 6. Pass B path length ---------------------------------------------
    if not len(paths):
        results.append(("Pass B path length ~21 trading days per position", False, "no paths pulled"))
    else:
        per = paths.groupby("optionid").size()
        med_len = float(per.median())
        expected = 21.0
        results.append((
            "Pass B path length ~21 trading days per position",
            0.7 * expected <= med_len <= 1.4 * expected,
            f"{len(paths):,} rows over {len(per):,} contracts; "
            f"median {med_len:.0f} days (min {per.min()}, max {per.max()})",
        ))

    # --- 7. the December year-boundary union actually worked ---------------
    dec = selected.loc[selected["entry_date"].dt.month == 12]
    if not len(dec) or not len(paths):
        results.append(("December cohorts span the year boundary (the UNION trap)", False,
                        "no December positions or no paths to check"))
    else:
        dec_ids = set(dec["optionid"])
        dec_paths = paths.loc[paths["optionid"].isin(dec_ids)]
        crossed = dec_paths.loc[dec_paths["date"].dt.month == 1]
        n_with_jan = int(crossed["optionid"].nunique())
        n_should = int(dec.loc[dec["exdate"].dt.month == 1, "optionid"].nunique())
        results.append((
            "December cohorts span the year boundary (the UNION trap)",
            n_should > 0 and n_with_jan >= 0.95 * n_should,
            f"{n_with_jan} of {n_should} December contracts expiring in January have "
            f"January quotes",
        ))

    return results


def coverage_report(
    spec: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    by: str = "year",
) -> pd.DataFrame:
    """What fraction of intended stock-months produced a contract (Q9).

    A drop rate that concentrates in high-volatility months or small names is a selection
    bias capable of producing the entire result, so this is a reported table, not a footnote.
    """
    want = spec.loc[spec["has_secid"], ["entry_month", "entry_date", "secid", "rank", "mktcap_k"]].copy()
    want["secid"] = want["secid"].astype("int64")

    got = selected[["secid", "entry_date", "cp_flag"]].drop_duplicates()
    calls = got.loc[got["cp_flag"] == "C"].assign(has_call=True).drop(columns="cp_flag")
    puts = got.loc[got["cp_flag"] == "P"].assign(has_put=True).drop(columns="cp_flag")

    merged = (
        want.merge(calls, on=["secid", "entry_date"], how="left")
        .merge(puts, on=["secid", "entry_date"], how="left")
    )
    merged["has_call"] = merged["has_call"].notna()
    merged["has_put"] = merged["has_put"].notna()
    merged["has_either"] = merged["has_call"] | merged["has_put"]

    if by == "year":
        merged["group"] = merged["entry_date"].dt.year
    elif by == "month":
        merged["group"] = merged["entry_month"].astype(str)
    elif by == "cap_quintile":
        # Quintiles are formed WITHIN each month, so a name's bucket is relative to its
        # contemporaries rather than to the whole sample (mega-caps grew ~5x over 2017-2023;
        # a pooled cut would put most of the early sample in the bottom buckets). Months with
        # fewer than 5 names cannot form 5 bins -- that happens on debug subsets, not on the
        # real universe -- and are left unlabelled rather than raising.
        def _quintile(s: pd.Series) -> pd.Series:
            if s.notna().sum() < 5:
                return pd.Series(np.nan, index=s.index)
            return pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

        merged["group"] = merged.groupby("entry_month")["mktcap_k"].transform(_quintile)
        if merged["group"].isna().all():
            log.warning("cap-quintile coverage skipped: no month has 5+ names")
            return pd.DataFrame(
                columns=["group", "stock_months", "with_call", "with_put", "with_either",
                         "call_coverage", "put_coverage", "coverage"]
            )
    else:
        raise ValueError(f"unknown grouping {by!r}")

    out = merged.groupby("group", observed=True).agg(
        stock_months=("secid", "size"),
        with_call=("has_call", "sum"),
        with_put=("has_put", "sum"),
        with_either=("has_either", "sum"),
    ).reset_index()
    out["call_coverage"] = out["with_call"] / out["stock_months"]
    out["put_coverage"] = out["with_put"] / out["stock_months"]
    out["coverage"] = out["with_either"] / out["stock_months"]
    return out
