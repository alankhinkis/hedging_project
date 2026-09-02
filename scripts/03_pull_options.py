"""Stage 2b, Pass A -- pull candidate option chains on entry dates, plus rates and dividends.

    python scripts/03_pull_options.py --permnos 14593 --years 2019   # the dev subsample
    python scripts/03_pull_options.py                                # full sample

Only volume reducers go in the SQL (secid, entry date, a deliberately wide DTE window).
Every economic screen is applied in scripts/04_select_contracts.py so its drop count is
recoverable from cache -- coverage is a result (Q9), and the IV cap has to be run at two
thresholds (Q8) without a re-pull.

Writes data/interim/{option_candidates,zero_curve,dividends,entry_spec}.parquet.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

from vrp.config import load_config  # noqa: E402
from vrp.data.option_chains import build_entry_spec, fetch_candidates  # noqa: E402
from vrp.data.rates_divs import fetch_dividends, fetch_zero_curve  # noqa: E402
from vrp.wrds_conn import WRDSUnavailable, get_connection  # noqa: E402


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permnos", type=int, nargs="*", default=None,
                    help="restrict to these permnos (default: the whole universe)")
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="restrict to these entry years")
    ap.add_argument("--dev", action="store_true",
                    help="use the config dev subsample (AAPL, 2019) -- run this first")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    permnos = args.permnos
    years = args.years
    if args.dev:
        permnos = permnos or list(cfg["dev"]["permnos"])
        years = years or list(cfg["dev"]["years"])

    interim = cfg.data_interim
    for needed in ("universe.parquet", "entry_calendar.parquet", "linked_panel.parquet"):
        if not (interim / needed).exists():
            print(f"\n{interim / needed} is missing. Run stages 01 and 02 first.\n", file=sys.stderr)
            return 2

    universe = pd.read_parquet(interim / "universe.parquet")
    calendar = pd.read_parquet(interim / "entry_calendar.parquet")
    linked = pd.read_parquet(interim / "linked_panel.parquet")

    spec = build_entry_spec(universe, calendar, linked, cfg)
    if permnos:
        spec = spec.loc[spec["permno"].isin(set(permnos))]
    if years:
        spec = spec.loc[spec["entry_date"].dt.year.isin(set(years))]
    if not len(spec):
        print("\nno (permno, month) cells match the requested subset.\n", file=sys.stderr)
        return 2

    n_missing = int((~spec["has_secid"]).sum())
    print(f"entry spec: {len(spec):,} stock-months, "
          f"{spec['permno'].nunique()} permnos, "
          f"{spec['entry_month'].nunique()} months, "
          f"{spec['entry_date'].dt.year.nunique()} years")
    print(f"  without a secid on the entry date: {n_missing}")

    try:
        conn = get_connection(cfg)
    except WRDSUnavailable as exc:
        print(f"\nWRDS connection failed.\n\n{exc}\n", file=sys.stderr)
        return 2

    candidates = fetch_candidates(spec, cfg, conn=conn, force=args.force)
    print(f"\nPass A candidates: {len(candidates):,} rows")
    if len(candidates):
        print(f"  {candidates['secid'].nunique()} secids, "
              f"{candidates['date'].nunique()} entry dates, "
              f"{candidates['optionid'].nunique():,} distinct contracts")
        print(f"  dte range {int(candidates['dte'].min())}-{int(candidates['dte'].max())}, "
              f"calls {int((candidates['cp_flag'] == 'C').sum()):,} / "
              f"puts {int((candidates['cp_flag'] == 'P').sum()):,}")

    curve = fetch_zero_curve(cfg, conn=conn, force=args.force)
    print(f"\nzero curve: {len(curve):,} rows, "
          f"{curve['days'].min():.0f}-{curve['days'].max():.0f} day tenors, "
          f"r from {curve['r'].min():.4f} to {curve['r'].max():.4f}")

    secids = sorted(spec.loc[spec["has_secid"], "secid"].astype("int64").unique())
    divs = fetch_dividends(secids, cfg, conn=conn, force=args.force)
    if len(divs):
        lead = (divs["ex_date"] - divs["declare_date"]).dt.days
        print(f"dividends: {len(divs):,} ordinary cash rows for {divs['secid'].nunique()} secids; "
              f"declaration lead median {lead.median():.0f}d, "
              f"{(lead >= 30).mean():.0%} declared >=30d before ex-date")
    else:
        print("dividends: none")

    out = spec.copy()
    out["entry_month"] = out["entry_month"].astype(str)
    out.to_parquet(interim / "entry_spec.parquet", index=False)
    candidates.to_parquet(interim / "option_candidates.parquet", index=False)
    curve.to_parquet(interim / "zero_curve.parquet", index=False)
    divs.to_parquet(interim / "dividends.parquet", index=False)
    print(f"\nwrote {interim.relative_to(cfg.root)}/"
          "{entry_spec,option_candidates,zero_curve,dividends}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
