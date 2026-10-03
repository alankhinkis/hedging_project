"""Stage 4 -- run the delta-hedging engine over the selected positions.

    python scripts/06_run_hedge.py --checkpoint-only   # 4(a) synthetic test, no data
    python scripts/06_run_hedge.py --dev               # AAPL 2019, prints a full path
    python scripts/06_run_hedge.py                     # the whole cross-section

Needs no WRDS connection: every input is already cached. Writes
data/processed/hedge_results.parquet and prints Checkpoints 4(a) and 4(b).
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import print_checkpoint, setup_logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vrp.config import load_config  # noqa: E402
from vrp.hedging import checkpoint_4a, delta_hedged_gain  # noqa: E402
from vrp.positions import (  # noqa: E402
    build_dividend_lookup,
    build_positions,
    build_price_lookup,
    build_rate_lookup,
    build_vol_lookup,
    prepare_inputs,
)


def checkpoint_4b(results: pd.DataFrame, sample_path: pd.DataFrame | None,
                  sample_pnl: float | None) -> list[tuple[str, bool, str]]:
    """Checkpoint 4(b): the engine behaves sanely on real positions."""
    out: list[tuple[str, bool, str]] = []
    ok = results.loc[results["error"].isna()]

    out.append((
        "the engine prices essentially every position",
        len(ok) / max(len(results), 1) > 0.98,
        f"{len(ok):,} of {len(results):,} ran; "
        + (results.loc[results['error'].notna(), 'error']
           .str.slice(0, 40).value_counts().head(2).to_dict().__str__()
           if len(ok) < len(results) else "no failures"),
    ))

    ed = ok["entry_delta"].abs()
    out.append((
        "entry delta is near 0.50 (the selection targeted it, the engine recomputes it)",
        0.42 <= float(ed.median()) <= 0.58,
        f"median |entry delta| {ed.median():.3f} "
        f"(selection used OptionMetrics' delta; this is ours, at physical vol)",
    ))

    # Terminal delta should be the step function ONLY where the option's expiry is itself a
    # trading day. Before February 2015, CBOE equity options expired on the Saturday after the
    # third Friday; OptionMetrics records that Saturday as `exdate`, so the last available
    # price is the Friday and the final delta is evaluated at tau = 1 day, legitimately
    # interior. (Every one of the 62,611 pre-2015 expiries in this sample is a Saturday;
    # every post-2015 one is a Friday or, in holiday weeks, a Thursday.)
    #
    # This does not touch the P&L: the terminal payoff uses the last traded close, which is
    # the correct settlement reference, and the final row's delta never drives a hedge
    # increment because there is no subsequent price. The condition is therefore split --
    # gated where expiry is a trading day, reported otherwise.
    td = ok["terminal_delta"].abs()
    settles_on_trading_day = ok["expiry"].dt.dayofweek < 5
    at_bounds_td = float(((td < 0.02) | (td > 0.98))[settles_on_trading_day].mean())
    out.append((
        "terminal delta resolves to 0 or 1 where expiry IS a trading day",
        at_bounds_td > 0.90 if settles_on_trading_day.any() else True,
        f"{at_bounds_td:.1%} of {int(settles_on_trading_day.sum()):,} such positions",
    ))
    n_sat = int((~settles_on_trading_day).sum())
    if n_sat:
        at_bounds_sat = float(((td < 0.02) | (td > 0.98))[~settles_on_trading_day].mean())
        out.append((
            "pre-2015 Saturday expiries are reported, not gated (CBOE convention change)",
            True,
            f"{n_sat:,} positions expire on a Saturday; {at_bounds_sat:.1%} end at a delta "
            f"bound, the rest at tau = 1 day, which does not affect the P&L",
        ))

    fin = ok["financing_pnl"].abs()
    opt = ok["option_pnl"].abs()
    out.append((
        "the financing leg is small relative to the option leg",
        float((fin / opt.replace(0, np.nan)).median()) < 0.10,
        f"median |financing| / |option P&L| = "
        f"{float((fin / opt.replace(0, np.nan)).median()):.3f}",
    ))

    if sample_path is not None and sample_pnl is not None:
        last = float(sample_path["cum_pnl"].iloc[-1])
        out.append((
            "cum_pnl on the last path row equals the scalar pnl",
            abs(last - sample_pnl) < 1e-8,
            f"{last:.6f} vs {sample_pnl:.6f}",
        ))

    carried = int(ok["delta_carried_days"].sum())
    out.append((
        "missing days are carried, not dropped (reported, not gated)",
        True,
        f"{carried:,} carried-delta days across {len(ok):,} positions",
    ))
    return out


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--permnos", type=int, nargs="*", default=None)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N positions (smoke test)")
    ap.add_argument("--checkpoint-only", action="store_true",
                    help="run only the 4(a) synthetic test, which needs no data")
    ap.add_argument("--vol-col", default="vol_hedge",
                    help="vol_hedge (GARCH + fallback), vol_h, vol_g, or vol_h_alt")
    ap.add_argument("--known-only-dividends", action="store_true")
    ap.add_argument("--implied-placebo", action="store_true",
                    help="Stage 4(c): hedge at the option's own IMPLIED vol instead of the "
                         "physical estimate. Needs Pass B paths. Hedging at implied should "
                         "partially hedge away the premium, so the measured loss shrinks; if "
                         "nothing moves, the vol input is not reaching the delta.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    print("Running Checkpoint 4(a): the zero-VRP synthetic test ...")
    ok_a = print_checkpoint("CHECKPOINT 4(a) -- zero-VRP synthetic validation",
                            checkpoint_4a(n_paths=1500, seed=0))
    if args.checkpoint_only:
        return 0 if ok_a else 1
    if not ok_a:
        print("4(a) failed; not running on real data.", file=sys.stderr)
        return 1

    interim = cfg.data_interim
    need = ["selected_contracts.parquet", "prices.parquet", "vol_panel.parquet",
            "zero_curve.parquet", "dividends.parquet"]
    for f in need:
        if not (interim / f).exists():
            print(f"\n{interim / f} is missing. Run stages 02-05 first.\n", file=sys.stderr)
            return 2

    selected = pd.read_parquet(interim / "selected_contracts.parquet")
    prices = pd.read_parquet(interim / "prices.parquet")
    vol_panel = pd.read_parquet(interim / "vol_panel.parquet")
    curve = pd.read_parquet(interim / "zero_curve.parquet")
    divs = pd.read_parquet(interim / "dividends.parquet")

    permnos = list(cfg["dev"]["permnos"]) if args.dev else args.permnos
    years = list(cfg["dev"]["years"]) if args.dev else args.years
    if permnos:
        selected = selected.loc[selected["permno"].isin(set(permnos))]
    if years:
        selected = selected.loc[selected["entry_date"].dt.year.isin(set(years))]
    if args.limit:
        selected = selected.head(args.limit)
    print(f"\npositions: {len(selected):,} "
          f"({int((selected['cp_flag'] == 'C').sum()):,} calls, "
          f"{int((selected['cp_flag'] == 'P').sum()):,} puts), vol input = {args.vol_col}")

    # Stage 4(c): swap the hedge volatility for the contract's own daily implied vol. This is
    # simultaneously a robustness column and a wiring test -- if the headline barely moves,
    # the volatility argument is not actually reaching delta_fn.
    iv_lookup = {}
    if args.implied_placebo:
        paths_file = interim / "option_paths.parquet"
        if not paths_file.exists():
            print(f"\n--implied-placebo needs {paths_file}; run "
                  f"scripts/04_select_contracts.py without --skip-paths first.\n",
                  file=sys.stderr)
            return 2
        paths = pd.read_parquet(paths_file, columns=["optionid", "date", "impl_volatility"])
        paths = paths.loc[paths["impl_volatility"].notna()]
        iv_lookup = {
            int(o): g.set_index("date")["impl_volatility"].astype("float64").sort_index()
            for o, g in paths.groupby("optionid")
        }
        print(f"implied-vol placebo: daily IV for {len(iv_lookup):,} contracts")

    price_lookup = build_price_lookup(prices)
    vol_lookup = build_vol_lookup(vol_panel, args.vol_col)
    rate_lookup = build_rate_lookup(curve)
    div_lookup = build_dividend_lookup(divs)
    positions = build_positions(selected)

    rows, first_path, first_pnl = [], None, None
    try:
        from tqdm import tqdm
        it = tqdm(positions, desc="hedging", unit="pos")
    except ImportError:
        it = positions

    for pos in it:
        base = {"optionid": pos.optionid, "permno": pos.permno, "secid": pos.secid,
                "cp_flag": pos.cp_flag, "strike": pos.strike,
                "entry_date": pos.entry_date, "expiry": pos.expiry}
        try:
            und, vol, rate = prepare_inputs(
                pos, price_lookup, vol_lookup, rate_lookup, div_lookup,
                known_only_dividends=args.known_only_dividends,
            )
            if args.implied_placebo:
                iv = iv_lookup.get(pos.optionid)
                if iv is None or not len(iv):
                    raise KeyError("no implied-vol path for this contract")
                # Fall back to the physical estimate only where IV is missing, so the two
                # runs stay on the same positions and the comparison is like-for-like.
                vol = iv.reindex(vol.index).ffill().fillna(vol)
            res = delta_hedged_gain(pos, und, vol, rate)
        except Exception as exc:  # noqa: BLE001
            rows.append({**base, "pnl": np.nan, "error": f"{type(exc).__name__}: {exc}"[:120]})
            continue
        rows.append({**base, "pnl": res.pnl, "error": None, **res.diagnostics})
        if first_path is None:
            first_path, first_pnl = res.daily, res.pnl

    results = pd.DataFrame(rows)
    good = results["error"].isna()
    results.loc[good, "pnl_over_S"] = results.loc[good, "pnl"] / results.loc[good, "entry_S"]
    results.loc[good, "pnl_over_C"] = results.loc[good, "pnl"] / results.loc[good, "entry_price"]

    suffix = "_implied_placebo" if args.implied_placebo else ""
    out_path = cfg.data_processed / f"hedge_results{suffix}.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path.relative_to(cfg.root)}")

    ok = results.loc[good]
    if len(ok):
        print("\n--- HEADLINE (unconditional, pooled; Stage 5 does the inference) ---")
        for label, col, fmt in (("mean pi ($)", "pnl", "{:+.4f}"),
                                ("mean pi/S (%)", "pnl_over_S", "{:+.4%}"),
                                ("mean pi/C (%)", "pnl_over_C", "{:+.2%}")):
            print(f"  {label:<16} {fmt.format(ok[col].mean())}   "
                  f"median {fmt.format(ok[col].median())}")
        print(f"  fraction pi < 0  {float((ok['pnl'] < 0).mean()):.1%}   "
              f"(BK Table 1 ATM: 68%)")
        print(f"  N                {len(ok):,}")

    if args.dev and first_path is not None:
        print("\n--- sample path (Checkpoint 4(b) hand-check) ---")
        cols = ["date", "S", "tau", "sigma_hat", "r", "delta",
                "d_hedge_pnl", "d_financing", "cum_pnl"]
        with pd.option_context("display.width", 200, "display.float_format", "{:,.4f}".format):
            print(first_path[cols].to_string(index=False))

    ok_b = print_checkpoint("CHECKPOINT 4(b) -- the engine on real positions",
                            checkpoint_4b(results, first_path, first_pnl))
    return 0 if ok_b else 1


if __name__ == "__main__":
    raise SystemExit(main())
