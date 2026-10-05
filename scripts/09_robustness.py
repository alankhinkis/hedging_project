"""Stage 5 robustness -- the specification variants the plan asks for, in one table.

    python scripts/09_robustness.py

Each variant re-runs the engine over the same positions with one thing changed, so the
columns are directly comparable:

* **Q8** -- BK's literal 100% implied-vol cap instead of our 300% primary.
* **Q12** -- enter at the ASK rather than the midpoint: the cost of actually being long.
* **Q3**  -- hedge at VOL^h (rolling realised) instead of the GARCH-led `vol_hedge`.
* **Q2**  -- CRR American delta instead of the Black-Scholes European one, on a subsample
  (it is ~50x slower, which is why the plan scopes it to a subsample).
* **4(c)** -- the implied-volatility placebo, run separately by `06_run_hedge.py
  --implied-placebo` and merged in here if present.

Needs no WRDS connection. Writes output/tables/table11_robustness.csv.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import setup_logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vrp.analysis.stats import monthly_portfolio_tstat  # noqa: E402
from vrp.config import load_config  # noqa: E402
from vrp.hedging import delta_hedged_gain  # noqa: E402
from vrp.positions import (  # noqa: E402
    build_dividend_lookup,
    build_positions,
    build_price_lookup,
    build_rate_lookup,
    build_vol_lookup,
    prepare_inputs,
)
from vrp.pricing import bs_delta, crr_american_delta  # noqa: E402


def run_variant(selected, price_lookup, vol_lookup, rate_lookup, div_lookup, *,
                delta_fn=bs_delta, label="", quiet=False):
    """Run the engine over `selected` and return the per-position results."""
    rows = []
    positions = build_positions(selected)
    it = positions
    if not quiet:
        try:
            from tqdm import tqdm
            it = tqdm(positions, desc=label[:22], unit="pos", leave=False)
        except ImportError:
            pass
    for pos in it:
        try:
            und, vol, rate = prepare_inputs(pos, price_lookup, vol_lookup,
                                            rate_lookup, div_lookup)
            res = delta_hedged_gain(pos, und, vol, rate, delta_fn=delta_fn)
        except Exception:  # noqa: BLE001
            continue
        rows.append({"optionid": pos.optionid, "permno": pos.permno, "cp_flag": pos.cp_flag,
                     "entry_date": pos.entry_date, "pnl": res.pnl,
                     "pnl_over_S": res.pnl / res.diagnostics["entry_S"],
                     "pnl_over_C": res.pnl / res.diagnostics["entry_price"],
                     "error": None})
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, label: str, note: str = "") -> dict:
    """One table row. Every t-stat is paired with the mean from the SAME estimator.

    The t-stat is a Newey-West t on the series of monthly average gains, so the mean reported
    beside it (`mean_monthly_pct`) is the mean of those monthly averages. The pooled
    position-level mean is also reported, but under its own name, because it weights months by
    how many positions they hold (167 to 362) and so differs from the monthly mean. An earlier
    version of this table printed the pooled mean next to the monthly t-stat.
    """
    if not len(results):
        return {"variant": label, "N": 0, "note": note}
    inf = monthly_portfolio_tstat(results, "pnl_over_S")
    x = results["pnl_over_S"]
    return {
        "variant": label,
        "N": len(results),
        "mean_monthly_pct": float(inf["mean"] * 100),
        "t_monthly_NW": float(inf["t"]),
        "n_months": int(inf["n_obs"]),
        "mean_pooled_pct": float(x.mean() * 100),
        "median_pct": float(x.median() * 100),
        "frac_negative": float((results["pnl"] < 0).mean()),
        "note": note,
    }


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--american-subsample", type=int, default=2000,
                    help="positions for the CRR American delta column (it is ~50x slower)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    interim, proc = cfg.data_interim, cfg.data_processed
    base_path = proc / "hedge_results.parquet"
    if not base_path.exists():
        print(f"\n{base_path} is missing. Run scripts/06_run_hedge.py first.\n", file=sys.stderr)
        return 2

    selected = pd.read_parquet(interim / "selected_contracts.parquet")
    prices = pd.read_parquet(interim / "prices.parquet")
    vol_panel = pd.read_parquet(interim / "vol_panel.parquet")
    curve = pd.read_parquet(interim / "zero_curve.parquet")
    divs = pd.read_parquet(interim / "dividends.parquet")

    price_lookup = build_price_lookup(prices)
    rate_lookup = build_rate_lookup(curve)
    div_lookup = build_dividend_lookup(divs)
    vol_hedge = build_vol_lookup(vol_panel, "vol_hedge")

    rows = []
    var_dir = proc / "robustness_variants"
    var_dir.mkdir(exist_ok=True)

    def keep(df, name):
        """Persist a variant's per-position results; return it unchanged."""
        df.to_parquet(var_dir / f"{name}.parquet", index=False)
        return df

    base = pd.read_parquet(base_path)
    base = base.loc[base["error"].isna()]
    rows.append(summarise(base, "BASELINE", "mid entry, GARCH-led vol, BS delta, 300% IV cap"))

    # --- Q12: enter at the ask --------------------------------------------
    ask = selected.copy()
    ask["mid"] = ask["best_offer"].astype("float64")
    r = keep(run_variant(ask, price_lookup, vol_hedge, rate_lookup, div_lookup,
                         label="Q12 ask entry"), "q12_ask")
    rows.append(summarise(r, "Q12: enter at the ASK, not the mid",
                          "the cost of actually being long the option"))

    # --- Q8: BK's literal 100% implied-vol cap -----------------------------
    lit_path = interim / "selected_contracts_iv100.parquet"
    if lit_path.exists():
        lit = pd.read_parquet(lit_path)
        r = keep(run_variant(lit, price_lookup, vol_hedge, rate_lookup, div_lookup,
                             label="Q8 IV cap 100%"), "q8_iv100")
        rows.append(summarise(r, "Q8: BK's literal 100% IV cap",
                              f"{len(lit) - len(selected):+,} positions vs baseline"))

    # --- Q3: hedge at VOL^h instead of the GARCH-led series ----------------
    vol_h = build_vol_lookup(vol_panel, "vol_h")
    r = keep(run_variant(selected, price_lookup, vol_h, rate_lookup, div_lookup,
                         label="Q3 VOL^h"), "q3_volh")
    rows.append(summarise(r, "Q3: hedge at VOL^h (rolling realised)",
                          "baseline uses GARCH with VOL^h fallback"))

    # --- Q2: CRR American delta on a subsample -----------------------------
    sub = selected.sample(n=min(args.american_subsample, len(selected)), random_state=0)
    r_eu = keep(run_variant(sub, price_lookup, vol_hedge, rate_lookup, div_lookup,
                            label="Q2 European (sub)"), "q2a_european_sub")
    rows.append(summarise(r_eu, "Q2a: BS European delta, subsample",
                          f"same {len(sub):,} positions as the row below"))
    r_am = keep(run_variant(sub, price_lookup, vol_hedge, rate_lookup, div_lookup,
                            delta_fn=lambda S, K, t, s, rr, q, cp:
                                crr_american_delta(S, K, t, s, rr, q, cp, steps=120),
                            label="Q2 American (sub)"), "q2b_american_sub")
    rows.append(summarise(r_am, "Q2b: CRR American delta, subsample",
                          "single-name options ARE American; this bounds the approximation"))

    # --- 4(c): the implied-vol placebo ------------------------------------
    plc_path = proc / "hedge_results_implied_placebo.parquet"
    if plc_path.exists():
        plc = pd.read_parquet(plc_path)
        plc = plc.loc[plc["error"].isna()]
        rows.append(summarise(plc, "4(c): hedge at IMPLIED vol (placebo)",
                              "should shrink the premium; also a wiring test"))

    table = pd.DataFrame(rows)
    table.to_csv(cfg.output_tables / "table11_robustness.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Table 11: robustness ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nwrote {(cfg.output_tables / 'table11_robustness.csv').relative_to(cfg.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
