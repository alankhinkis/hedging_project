"""Stage 1 -- build the point-in-time top-N S&P 500 universe.

    python scripts/01_build_universe.py              # full sample, runs Checkpoint 1
    python scripts/01_build_universe.py --top-n 50   # sensitivity

Writes data/interim/universe.parquet (rank_month, entry_month, permno, rank, mktcap_k) plus
a turnover summary to output/tables/. Exits non-zero if Checkpoint 1 fails.

Note on `--permnos`/`--years`: unlike later stages, the universe *is* the cross-section, so
subsetting it defeats the point of the stage. The flags exist for debugging a single name's
membership history; the checkpoint is skipped when they are used.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import print_checkpoint, setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

from vrp.config import load_config  # noqa: E402
from vrp.data.universe import (  # noqa: E402
    build_universe,
    checkpoint_1,
    fetch_membership,
    fetch_monthly_mktcap,
    fetch_names,
    member_counts,
    universe_summary,
)
from vrp.wrds_conn import WRDSUnavailable, get_connection  # noqa: E402


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-n", type=int, default=int(cfg["universe"]["top_n"]))
    ap.add_argument("--permnos", type=int, nargs="*", default=None,
                    help="debug only: restrict to these permnos (skips Checkpoint 1)")
    ap.add_argument("--force", action="store_true", help="ignore the parquet cache and re-pull")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    try:
        conn = get_connection(cfg)
    except WRDSUnavailable as exc:
        print(f"\nWRDS connection failed.\n\n{exc}\n", file=sys.stderr)
        return 2

    membership = fetch_membership(cfg, conn=conn, force=args.force)
    print(f"membership spans: {len(membership):,} rows, "
          f"{membership['permno'].nunique():,} distinct permnos")

    permnos = sorted(membership["permno"].unique())
    if args.permnos:
        permnos = [p for p in permnos if p in set(args.permnos)]
        membership = membership.loc[membership["permno"].isin(permnos)]

    mktcap = fetch_monthly_mktcap(permnos, cfg, conn=conn, force=args.force)
    print(f"monthly market caps: {len(mktcap):,} rows")

    counts = member_counts(membership, mktcap)
    universe = build_universe(membership, mktcap, top_n=args.top_n)
    print(f"universe: {len(universe):,} (month, permno) rows, "
          f"{universe['permno'].nunique():,} distinct permnos, "
          f"{universe['entry_month'].nunique()} entry months")

    names = fetch_names(sorted(universe["permno"].unique()), cfg, conn=conn, force=args.force)

    # --- persist ----------------------------------------------------------
    out = universe.copy()
    out["rank_month"] = out["rank_month"].astype(str)
    out["entry_month"] = out["entry_month"].astype(str)
    uni_path = cfg.data_interim / "universe.parquet"
    out.to_parquet(uni_path, index=False)

    counts_out = counts.copy()
    counts_out["rank_month"] = counts_out["rank_month"].astype(str)
    counts_out.to_csv(cfg.output_tables / "universe_member_counts.csv", index=False)

    summary = universe_summary(universe)
    summary.to_csv(cfg.output_tables / "universe_turnover.csv", index=False)

    print(f"\nwrote {uni_path.relative_to(cfg.root)}")
    print("\nturnover (first and last 6 entry months):")
    with pd.option_context("display.max_rows", 20):
        print(pd.concat([summary.head(6), summary.tail(6)]).to_string(index=False))

    if args.permnos:
        print("\n--permnos was used; Checkpoint 1 skipped (it is a whole-cross-section test).")
        return 0

    ok = print_checkpoint(
        "CHECKPOINT 1 -- point-in-time universe",
        checkpoint_1(membership, mktcap, universe, cfg, names=names),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
