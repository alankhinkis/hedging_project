"""Stage 3 -- build the physical volatility panel (VOL^h and VOL^g).

    python scripts/05_build_vol.py --dev     # AAPL only, fast
    python scripts/05_build_vol.py           # full panel, runs Checkpoint 3

Reads data/interim/prices.parquet (already cached) and needs no WRDS connection.
Writes data/interim/vol_panel.parquet and output/tables/garch_fits.csv.
Exits non-zero if Checkpoint 3 fails.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import print_checkpoint, setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

from vrp.config import load_config  # noqa: E402
from vrp.vol import build_vol_panel, checkpoint_3  # noqa: E402


def plot_vol(panel: pd.DataFrame, cfg) -> object:
    """AAPL's two estimators through the sample. The March 2020 spike is the eyeball test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    permno = int(cfg["sanity"]["recon_permno"])
    g = panel.loc[panel["permno"] == permno].sort_values("date")
    if not len(g):
        return None
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g["date"], g["vol_h"], lw=0.9, label="VOL^h (30d realised)")
    ax.plot(g["date"], g["vol_g"], lw=1.2, label="VOL^g (GARCH, Eq. 28)")
    ax.set_title(f"Physical volatility, permno {permno}")
    ax.set_ylabel("annualised sigma")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = cfg.output_figures / "volatility_estimators.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true", help="restrict to the config dev subsample")
    ap.add_argument("--permnos", type=int, nargs="*", default=None)
    ap.add_argument("--no-garch", action="store_true", help="VOL^h only (fast)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    prices_path = cfg.data_interim / "prices.parquet"
    if not prices_path.exists():
        print(f"\n{prices_path} is missing. Run scripts/02_pull_crsp.py first.\n", file=sys.stderr)
        return 2
    prices = pd.read_parquet(prices_path)

    permnos = list(cfg["dev"]["permnos"]) if args.dev else args.permnos
    if permnos:
        prices = prices.loc[prices["permno"].isin(set(permnos))]
    print(f"prices: {len(prices):,} rows over {prices['permno'].nunique()} permnos, "
          f"{prices['date'].min().date()} to {prices['date'].max().date()}")

    panel, fits = build_vol_panel(prices, cfg, with_garch=not args.no_garch)
    print(f"\nvol panel: {len(panel):,} (permno, date) rows")
    for col in ("vol_h", "vol_h_alt", "vol_g", "vol_hedge"):
        if col in panel.columns:
            s = panel[col].dropna()
            if len(s):
                print(f"  {col:<10} n={len(s):>8,}  median {s.median():6.1%}  "
                      f"p01 {s.quantile(0.01):5.1%}  p99 {s.quantile(0.99):6.1%}")

    if len(fits):
        real = fits.loc[~fits["burn_in"]]
        print(f"\nGARCH fits: {len(fits):,} stock-years "
              f"({int(fits['burn_in'].sum())} burn-in, {len(real):,} fittable), "
              f"{real['converged'].mean():.1%} converged")
        if (~real["converged"]).any():
            print("  failure reasons:",
                  real.loc[~real["converged"], "note"].value_counts().head(3).to_dict())
        fits.to_csv(cfg.output_tables / "garch_fits.csv", index=False)

    panel.to_parquet(cfg.data_interim / "vol_panel.parquet", index=False)
    print(f"\nwrote {(cfg.data_interim / 'vol_panel.parquet').relative_to(cfg.root)}")

    fig = plot_vol(panel, cfg)
    if fig:
        print(f"wrote {fig.relative_to(cfg.root)}")

    if permnos:
        print("\n--permnos/--dev was used; Checkpoint 3 skipped (it is a whole-panel test).")
        return 0

    ok = print_checkpoint("CHECKPOINT 3 -- physical volatility", checkpoint_3(panel, fits, cfg))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
