"""Stage 0 -- verify every WRDS table this project assumes exists.

Resolves each guess in config.yaml against the live server, records the exact
schema.table / column list / row count / date range, runs the Checkpoint 0 probes, and
writes docs/wrds_schema_notes.md plus docs/wrds_schema_resolved.json.

    python scripts/00_verify_schema.py

Exits non-zero if Checkpoint 0 fails, so it can gate the rest of the build.
"""

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import print_checkpoint, setup_logging  # noqa: E402  (path bootstrap)

from vrp.config import load_config  # noqa: E402
from vrp.schema import (  # noqa: E402
    checkpoint_0,
    probe_aapl_options,
    probe_dividends,
    probe_link_scores,
    profile_targets,
    write_notes,
    write_resolved,
)
from vrp.wrds_conn import WRDSUnavailable, get_connection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    cfg = load_config()
    try:
        conn = get_connection(cfg)
    except WRDSUnavailable as exc:
        print(f"\nWRDS connection failed.\n\n{exc}\n", file=sys.stderr)
        return 2

    findings = profile_targets(cfg, conn=conn)
    resolved = {f.need: f.resolved for f in findings}

    probes: dict[str, object] = {}

    link_table = resolved.get("OM <-> CRSP link")
    if link_table:
        try:
            probes["link_score_distribution"] = probe_link_scores(link_table, cfg, conn).to_dict("records")
        except Exception as exc:  # noqa: BLE001
            probes["link_score_distribution_error"] = str(exc).splitlines()[0]

    opprcd = resolved.get("Daily option prices (2019 probe)")
    if opprcd:
        probes["aapl_options_probe"] = probe_aapl_options(opprcd, link_table, cfg, conn)

    probes["dividend_question_q5"] = probe_dividends(
        resolved.get("Projected dividends (discrete)"), cfg, conn
    )

    notes_path = write_notes(findings, probes, cfg)
    json_path = write_resolved(findings, cfg)

    print("\nResolved tables:")
    for f in findings:
        flag = "" if f.resolved == f.guess else "  <-- via fallback" if f.resolved else "  <-- NOT FOUND"
        print(f"  {f.need:<42} {str(f.resolved):<40}{flag}")
        if f.missing_key_columns:
            print(f"      missing key columns: {f.missing_key_columns}")

    probe = probes.get("aapl_options_probe")
    if isinstance(probe, dict) and probe.get("nearest_atm"):
        print("\nNearest-ATM AAPL contract on the probe date:")
        print(json.dumps(probe["nearest_atm"], indent=2))

    print(f"\nWrote {notes_path.relative_to(cfg.root)}")
    print(f"Wrote {json_path.relative_to(cfg.root)}")

    ok = print_checkpoint("CHECKPOINT 0 -- environment and schema", checkpoint_0(findings, probes))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
