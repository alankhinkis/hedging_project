"""Stage 2b, selection + Pass B -- screen the candidates, pick one contract per
stock-month-side, then pull those contracts' daily paths.

    python scripts/04_select_contracts.py --dev     # AAPL 2019, runs Checkpoint 2b
    python scripts/04_select_contracts.py           # full sample

Pass B lives here rather than in 03 because it is keyed on the selected optionids: you
cannot pull the paths until you know which contracts you hold.

The screen cascade is run twice -- at the primary 300% implied-vol cap and at BK's literal
100% (Q8) -- and both drop counts are written out, so the robustness row exists before any
result does.

Writes data/interim/{selected_contracts,option_paths}.parquet and coverage/screen tables to
output/tables/. Exits non-zero if Checkpoint 2b fails.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import print_checkpoint, setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

from vrp.config import load_config  # noqa: E402
from vrp.data.option_chains import fetch_paths  # noqa: E402
from vrp.selection import (  # noqa: E402
    apply_screens,
    checkpoint_2b,
    coverage_report,
    select_contracts,
)
from vrp.wrds_conn import WRDSUnavailable, get_connection  # noqa: E402


def build_spot(prices: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    """(secid, date, S) on entry dates. S is the as-traded close, since a position's entry
    date is by definition its own normalisation reference (Q11)."""
    spot = prices.loc[prices["has_price"], ["permno", "date", "prc_abs"]]
    out = spot.merge(linked[["permno", "date", "secid"]], on=["permno", "date"], how="inner")
    out = out.loc[out["secid"].notna()].copy()
    out["secid"] = out["secid"].astype("int64")
    return out.rename(columns={"prc_abs": "S"})[["secid", "date", "S"]]


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true", help="restrict to the config dev subsample")
    ap.add_argument("--permnos", type=int, nargs="*", default=None)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--skip-paths", action="store_true", help="selection only, no Pass B pull")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    interim = cfg.data_interim
    for needed in ("option_candidates.parquet", "entry_spec.parquet", "prices.parquet",
                   "linked_panel.parquet", "zero_curve.parquet"):
        if not (interim / needed).exists():
            print(f"\n{interim / needed} is missing. Run scripts/03_pull_options.py first.\n",
                  file=sys.stderr)
            return 2

    candidates = pd.read_parquet(interim / "option_candidates.parquet")
    spec = pd.read_parquet(interim / "entry_spec.parquet")
    prices = pd.read_parquet(interim / "prices.parquet")
    linked = pd.read_parquet(interim / "linked_panel.parquet")
    curve = pd.read_parquet(interim / "zero_curve.parquet")
    spec["entry_month"] = pd.PeriodIndex(spec["entry_month"], freq="M")

    permnos = list(cfg["dev"]["permnos"]) if args.dev else args.permnos
    years = list(cfg["dev"]["years"]) if args.dev else args.years
    if permnos:
        spec = spec.loc[spec["permno"].isin(set(permnos))]
        keep = set(spec.loc[spec["has_secid"], "secid"].astype("int64"))
        candidates = candidates.loc[candidates["secid"].isin(keep)]
    if years:
        spec = spec.loc[spec["entry_date"].dt.year.isin(set(years))]
        candidates = candidates.loc[candidates["date"].dt.year.isin(set(years))]

    spot = build_spot(prices, linked)
    print(f"candidates: {len(candidates):,} rows; spot: {len(spot):,} (secid, date) cells")

    # --- screens, at both IV caps (Q8) ------------------------------------
    screened, screen_log = apply_screens(candidates, cfg, spot=spot)
    literal_cap = float(cfg["screens"]["iv_cap_literal"])
    screened_lit, screen_log_lit = apply_screens(candidates, cfg, iv_cap=literal_cap, spot=spot)

    print("\nscreen cascade (primary IV cap "
          f"{cfg['screens']['iv_cap_primary']:.0%}):")
    with pd.option_context("display.width", 200, "display.max_colwidth", 44):
        print(screen_log[["screen", "rows_in", "rows_out", "rows_dropped",
                          "stock_months_out"]].to_string(index=False))

    # --- selection ---------------------------------------------------------
    selected = select_contracts(screened, cfg)
    selected_lit = select_contracts(screened_lit, cfg)
    print(f"\nselected: {len(selected):,} positions "
          f"({int((selected['cp_flag'] == 'C').sum()):,} calls, "
          f"{int((selected['cp_flag'] == 'P').sum()):,} puts)")
    print(f"  at BK's literal {literal_cap:.0%} IV cap: {len(selected_lit):,} positions "
          f"({len(selected) - len(selected_lit):+,} difference)")

    cov_year = coverage_report(spec, selected, by="year")
    print("\ncoverage by year:")
    print(cov_year[["group", "stock_months", "with_call", "with_put", "coverage"]]
          .to_string(index=False))

    # --- Pass B ------------------------------------------------------------
    paths = pd.DataFrame()
    if not args.skip_paths and len(selected):
        try:
            conn = get_connection(cfg)
        except WRDSUnavailable as exc:
            print(f"\nWRDS connection failed.\n\n{exc}\n", file=sys.stderr)
            return 2
        paths = fetch_paths(selected, cfg, conn=conn, force=args.force)
        print(f"\nPass B paths: {len(paths):,} rows over "
              f"{paths['optionid'].nunique():,} contracts")

    # --- persist -----------------------------------------------------------
    selected.to_parquet(interim / "selected_contracts.parquet", index=False)
    selected_lit.to_parquet(interim / "selected_contracts_iv100.parquet", index=False)
    if len(paths):
        paths.to_parquet(interim / "option_paths.parquet", index=False)

    screen_log.to_csv(cfg.output_tables / "screen_cascade.csv", index=False)
    screen_log_lit.to_csv(cfg.output_tables / "screen_cascade_iv100.csv", index=False)
    cov_year.to_csv(cfg.output_tables / "coverage_by_year.csv", index=False)
    coverage_report(spec, selected, by="month").to_csv(
        cfg.output_tables / "coverage_by_month.csv", index=False)
    if spec["entry_month"].nunique() > 1:
        coverage_report(spec, selected, by="cap_quintile").to_csv(
            cfg.output_tables / "coverage_by_cap_quintile.csv", index=False)

    if not len(selected):
        print("\nno contracts selected -- cannot run Checkpoint 2b.", file=sys.stderr)
        return 1

    ok = print_checkpoint(
        "CHECKPOINT 2b -- option chains, screens and contract selection",
        checkpoint_2b(spec, screened, selected, paths, spot, curve, cfg),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
