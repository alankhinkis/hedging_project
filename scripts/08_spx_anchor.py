"""Stage 4.5 -- the SPX anchor: run the finished engine on the paper's own asset class.

    python scripts/08_spx_anchor.py

Separates "the engine is right and single-name VRP is small" from "something is wrong that
the synthetic test cannot see". Writes data/processed/spx_hedge_results.parquet and prints
Checkpoint 4.5.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import print_checkpoint, setup_logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vrp.analysis.stats import inference_table  # noqa: E402
from vrp.config import load_config  # noqa: E402
from vrp.data.rates_divs import fetch_zero_curve  # noqa: E402
from vrp.hedging import OptionPosition, delta_hedged_gain  # noqa: E402
from vrp.positions import build_rate_lookup, position_rate_path  # noqa: E402
from vrp.spx import (  # noqa: E402
    SPX_SECID,
    checkpoint_4_5,
    fetch_index_options,
    fetch_index_prices,
    fetch_index_yield,
    select_index_contracts,
)
from vrp.vol import garch_vol, log_price_returns, realized_vol  # noqa: E402
from vrp.wrds_conn import WRDSUnavailable, get_connection  # noqa: E402


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secid", type=int, default=SPX_SECID)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all-settlement", action="store_true",
                    help="keep PM-settled SPXW as well as the AM-settled monthly")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    cal_path = cfg.data_interim / "entry_calendar.parquet"
    if not cal_path.exists():
        print(f"\n{cal_path} is missing. Run scripts/02_pull_crsp.py first.\n", file=sys.stderr)
        return 2
    cal = pd.read_parquet(cal_path)
    cal["entry_month"] = pd.PeriodIndex(cal["entry_month"], freq="M")
    lo, hi = pd.Period(cfg.start_date, "M"), pd.Period(cfg.end_date, "M")
    entry_dates = cal.loc[cal["entry_month"].between(lo, hi), "entry_date"]
    print(f"entry dates: {len(entry_dates)} ({entry_dates.min().date()} to "
          f"{entry_dates.max().date()}), same rule as the cross-section")

    try:
        conn = get_connection(cfg)
    except WRDSUnavailable as exc:
        print(f"\nWRDS connection failed.\n\n{exc}\n", file=sys.stderr)
        return 2

    px = fetch_index_prices(args.secid, cfg, conn=conn, force=args.force)
    print(f"index levels: {len(px):,} days, {px['close'].min():,.0f} to {px['close'].max():,.0f}")
    yld = fetch_index_yield(args.secid, cfg, conn=conn, force=args.force)
    print(f"dividend yield: {len(yld):,} days, median {yld['q'].median():.2%}")

    cand = fetch_index_options(entry_dates, args.secid, cfg, conn=conn, force=args.force)
    print(f"candidates: {len(cand):,} rows over {cand['date'].nunique()} entry dates")

    picked, screen_log = select_index_contracts(cand, cfg,
                                                am_settled_only=not args.all_settlement)
    print(f"\nselected: {len(picked):,} positions "
          f"({int((picked['cp_flag'] == 'C').sum())} calls, "
          f"{int((picked['cp_flag'] == 'P').sum())} puts)")
    if len(picked):
        print(f"  dte {int(picked['dte'].min())}-{int(picked['dte'].max())}, "
              f"median |delta| {picked['abs_delta'].median():.3f}, "
              f"median rel spread {picked['rel_spread'].median():.2%}")
    screen_log.to_csv(cfg.output_tables / "spx_screen_cascade.csv", index=False)
    if not len(picked):
        print("\nno SPX contracts selected.", file=sys.stderr)
        return 1

    # --- volatility on the index itself -----------------------------------
    vol_input = px.rename(columns={"close": "s_adj"}).assign(permno=0, has_price=True)
    returns = log_price_returns(vol_input)
    vol_h = realized_vol(returns, int(cfg["vol"]["realized_window_days"]))
    vol_g, fits = garch_vol(returns, int(cfg["vol"]["realized_window_days"]), cfg=cfg)
    vol = vol_h.merge(vol_g, on=["permno", "date"], how="left")
    vol["vol_hedge"] = vol["vol_g"].where(vol["vol_g"].notna(), vol["vol_h"])
    vol_series = vol.set_index("date")["vol_hedge"].sort_index()
    print(f"\nSPX volatility: median {vol_series.median():.1%}, "
          f"max {vol_series.max():.1%}; GARCH usable in "
          f"{int(fits.loc[~fits['burn_in'], 'converged'].sum())} of "
          f"{int((~fits['burn_in']).sum())} years")

    curve = fetch_zero_curve(cfg, conn=conn, force=args.force)
    rate_lookup = build_rate_lookup(curve)
    yield_series = yld.set_index("date")["q"].sort_index()
    und_all = px[["date", "close"]].rename(columns={"close": "S"}).assign(pv_divs=0.0)

    # --- run the engine ----------------------------------------------------
    rows = []
    for row in picked.itertuples(index=False):
        pos = OptionPosition(
            secid=int(row.secid), permno=0, optionid=int(row.optionid),
            cp_flag=str(row.cp_flag), strike=float(row.strike),
            entry_date=pd.Timestamp(row.entry_date), expiry=pd.Timestamp(row.exdate),
            entry_price=float(row.mid),
        )
        base = {"optionid": pos.optionid, "cp_flag": pos.cp_flag, "strike": pos.strike,
                "entry_date": pos.entry_date, "expiry": pos.expiry}
        try:
            und = und_all.loc[und_all["date"].between(pos.entry_date, pos.expiry)].copy()
            dates = pd.DatetimeIndex(und["date"])
            rates = position_rate_path(rate_lookup, dates, pos.expiry)
            res = delta_hedged_gain(pos, und, vol_series.reindex(dates).ffill(), rates,
                                    yield_path=yield_series)
        except Exception as exc:  # noqa: BLE001
            rows.append({**base, "pnl": np.nan, "error": f"{type(exc).__name__}: {exc}"[:120]})
            continue
        rows.append({**base, "pnl": res.pnl, "error": None, **res.diagnostics})

    results = pd.DataFrame(rows)
    good = results["error"].isna()
    results.loc[good, "pnl_over_S"] = results.loc[good, "pnl"] / results.loc[good, "entry_S"]
    results.loc[good, "pnl_over_C"] = results.loc[good, "pnl"] / results.loc[good, "entry_price"]
    out_path = cfg.data_processed / "spx_hedge_results.parquet"
    results.to_parquet(out_path, index=False)
    print(f"wrote {out_path.relative_to(cfg.root)}")

    ok = results.loc[good]
    print("\n--- SPX ANCHOR ---")
    print(f"  mean pi ($)     {ok['pnl'].mean():+.4f}    median {ok['pnl'].median():+.4f}")
    print(f"  mean pi/S (%)   {ok['pnl_over_S'].mean() * 100:+.4f}%   "
          f"median {ok['pnl_over_S'].median() * 100:+.4f}%   [BK: -0.10% to -0.11%]")
    print(f"  mean pi/C (%)   {ok['pnl_over_C'].mean() * 100:+.2f}%   "
          f"median {ok['pnl_over_C'].median() * 100:+.2f}%   [BK ATM: -3.88%]")
    print(f"  frac negative   {float((ok['pnl'] < 0).mean()):.1%}   [BK ATM: 68%]")
    print(f"  N               {len(ok)}")

    inf = inference_table(ok, "pnl_over_S")
    inf.to_csv(cfg.output_tables / "spx_inference.csv", index=False)
    print("\n--- inference (conservative first) ---")
    print(inf[["method", "mean", "se", "t", "n_obs"]].to_string(index=False))

    by_year = ok.assign(year=ok["entry_date"].dt.year).groupby("year").agg(
        N=("pnl_over_S", "size"), mean_pct=("pnl_over_S", lambda s: s.mean() * 100),
        frac_neg=("pnl", lambda s: float((s < 0).mean())))
    by_year.to_csv(cfg.output_tables / "spx_by_year.csv")
    print("\n--- by year ---")
    print(by_year.round(4).to_string())

    ok_45 = print_checkpoint("CHECKPOINT 4.5 -- SPX anchor", checkpoint_4_5(results, cfg))
    return 0 if ok_45 else 1


if __name__ == "__main__":
    raise SystemExit(main())
