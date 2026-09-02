"""Stage 2a -- pull CRSP daily prices and the OptionMetrics <-> CRSP link.

    python scripts/02_pull_crsp.py                    # full universe, runs Checkpoint 2a
    python scripts/02_pull_crsp.py --permnos 14593    # AAPL only, for debugging

Writes data/interim/{prices,link,linked,delistings}.parquet, the entry-date calendar, and
a split-adjustment figure to output/figures/. Exits non-zero if Checkpoint 2a fails.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import print_checkpoint, setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

from vrp.config import load_config  # noqa: E402
from vrp.data.crsp_prices import (  # noqa: E402
    checkpoint_2a,
    clean_prices,
    fetch_daily_prices,
    fetch_delistings,
    first_trading_days,
)
from vrp.data.linking import (  # noqa: E402
    ambiguous_permnos,
    apply_link,
    fetch_link,
    link_coverage,
    secid_activity,
)
from vrp.wrds_conn import WRDSUnavailable, get_connection  # noqa: E402


def plot_splits(prices, cfg):
    """Plot raw vs adjusted price around each known split. The plan says look, don't assert."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = cfg["sanity"].get("splits", [])
    if not specs:
        return None
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 4), squeeze=False)
    for ax, spec in zip(axes[0], specs):
        d = pd.Timestamp(spec["date"])
        panel = prices.loc[
            (prices["permno"] == spec["permno"])
            & (prices["date"].between(d - pd.Timedelta(days=45), d + pd.Timedelta(days=45)))
        ].sort_values("date")
        ax.plot(panel["date"], panel["prc_abs"], label="raw |prc|", lw=1.2)
        ax.plot(panel["date"], panel["s_adj"], label="adjusted prc/cfacpr", lw=1.6)
        ax.axvline(d, color="k", ls=":", lw=1)
        # Log scale so the two series are comparable in *relative* terms: NVDA's adjusted
        # price sits near $20 against an ~$800 raw price (cfacpr carries the later 10-for-1
        # too), and on a linear axis its continuity is invisible against the axis.
        ax.set_yscale("log")
        ax.set_title(f"{spec['ticker']} {spec['ratio']}-for-1, {spec['date']}")
        ax.set_ylabel("price (log scale)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.tight_layout()
    path = cfg.output_figures / "split_adjustment.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permnos", type=int, nargs="*", default=None,
                    help="debug only: restrict to these permnos (skips Checkpoint 2a)")
    ap.add_argument("--force", action="store_true", help="ignore the parquet cache and re-pull")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    uni_path = cfg.data_interim / "universe.parquet"
    if not uni_path.exists():
        print(f"\n{uni_path} is missing. Run scripts/01_build_universe.py first.\n", file=sys.stderr)
        return 2
    universe = pd.read_parquet(uni_path)
    universe["entry_month"] = pd.PeriodIndex(universe["entry_month"], freq="M")
    universe["rank_month"] = pd.PeriodIndex(universe["rank_month"], freq="M")

    permnos = sorted(universe["permno"].unique())
    if args.permnos:
        permnos = [p for p in permnos if p in set(args.permnos)]
        universe = universe.loc[universe["permno"].isin(permnos)]
    print(f"universe: {len(universe):,} (month, permno) cells over {len(permnos)} permnos")

    try:
        conn = get_connection(cfg)
    except WRDSUnavailable as exc:
        print(f"\nWRDS connection failed.\n\n{exc}\n", file=sys.stderr)
        return 2

    raw = fetch_daily_prices(permnos, cfg, conn=conn, force=args.force)
    prices = clean_prices(raw)
    print(f"daily prices: {len(prices):,} rows, "
          f"{prices['date'].min().date()} to {prices['date'].max().date()}")
    print(f"  quote-average days (prc < 0): {int(prices['price_is_quote_avg'].sum()):,}")
    print(f"  missing-price days:           {int((~prices['has_price']).sum()):,}")
    if prices["cfacpr_is_bad"].any():
        print(f"  WARNING cfacpr == 0 rows:     {int(prices['cfacpr_is_bad'].sum()):,}")

    delistings = fetch_delistings(permnos, cfg, conn=conn, force=args.force)
    print(f"delisting events in window: {len(delistings):,}")

    link = fetch_link(permnos, cfg, conn=conn, force=args.force)
    print(f"link rows (score <= 5, permno not null): {len(link):,}, "
          f"{link['secid'].nunique():,} secids for {link['permno'].nunique():,} permnos")

    # 14 permnos carry a second, usually dormant secid. Count option rows for just those
    # candidates so the tie-break picks the chain that actually trades -- score alone does
    # not separate USB's two score-1 secids, and span picks the empty one.
    ambiguous = ambiguous_permnos(link)
    activity = None
    if ambiguous:
        cand = sorted(link.loc[link["permno"].isin(ambiguous), "secid"].unique())
        print(f"ambiguous permnos: {len(ambiguous)} carrying {len(cand)} candidate secids "
              f"-- counting option rows to adjudicate")
        activity = secid_activity(cand, cfg, conn=conn, force=args.force)
        print(f"  {int((activity > 0).sum())} of {len(activity)} candidate secids have option data")

    panel = prices[["permno", "date"]].drop_duplicates()
    linked = apply_link(panel, link, activity=activity)
    cov = link_coverage(universe, linked)
    print(f"link coverage: {cov['coverage']:.2%} of universe cells; "
          f"{cov['n_permnos_missing']} permnos unlinked")

    calendar = first_trading_days(prices)

    # --- persist ----------------------------------------------------------
    prices.to_parquet(cfg.data_interim / "prices.parquet", index=False)
    link.to_parquet(cfg.data_interim / "link.parquet", index=False)
    linked.to_parquet(cfg.data_interim / "linked_panel.parquet", index=False)
    if len(delistings):
        delistings.to_parquet(cfg.data_interim / "delistings.parquet", index=False)
    cal_out = calendar.copy()
    cal_out["entry_month"] = cal_out["entry_month"].astype(str)
    cal_out.to_parquet(cfg.data_interim / "entry_calendar.parquet", index=False)
    print(f"\nentry calendar: {len(calendar)} first-trading-days, "
          f"{calendar['entry_date'].min().date()} to {calendar['entry_date'].max().date()}")

    fig_path = plot_splits(prices, cfg)
    if fig_path:
        print(f"wrote {fig_path.relative_to(cfg.root)}")

    if args.permnos:
        print("\n--permnos was used; Checkpoint 2a skipped (link coverage is a whole-panel test).")
        return 0

    ok = print_checkpoint(
        "CHECKPOINT 2a -- CRSP prices and the OM<->CRSP link",
        checkpoint_2a(prices, universe, link, linked, delistings, cfg, activity=activity),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
