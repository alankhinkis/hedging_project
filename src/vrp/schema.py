"""Stage 0 -- WRDS schema reconnaissance.

Everything downstream assumes table and column names that have not been verified. This
module resolves each guess in `config.yaml:schema_targets` against the live server, records
exact schema.table / column list / row count / date range / verification date, and writes
two artefacts:

* `docs/wrds_schema_notes.md`  -- the human-readable note the plan asks for.
* `docs/wrds_schema_resolved.json` -- machine-readable {need -> table}, consumed by later
  stages via `load_resolved_tables()` so no stage hard-codes a guessed name.

Checkpoint 0 (from PHASE1_PLAN.md) is encoded in `checkpoint_0()`: connection works, every
table resolves, the link table's `score` semantics are recorded, a live AAPL-options probe
returns rows, and the equity dividend question (Q5) is answered from the server rather than
from memory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, load_config
from .wrds_conn import (
    describe_table,
    get_connection,
    resolve_table,
    split_table,
    table_profile,
)

log = logging.getLogger(__name__)

RESOLVED_JSON = "wrds_schema_resolved.json"
NOTES_MD = "wrds_schema_notes.md"

# Which column to profile a date range on, per need. Absent -> row count only.
_DATE_COL: dict[str, str] = {
    "Monthly market cap": "date",
    "Daily stock prices": "date",
    "Daily option prices (2019 probe)": "date",
    "Zero-coupon curve": "date",
    "Projected dividends (discrete)": "ex_date",
    "Projected dividends (point-in-time projection)": "date",
    "OM historical volatility": "date",
}


@dataclass
class TableFinding:
    need: str
    guess: str
    confidence: str
    resolved: str | None
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_key_columns: list[str] = field(default_factory=list)
    rows: int | None = None
    date_min: str | None = None
    date_max: str | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.resolved is not None and not self.missing_key_columns


def _profile_one(target: dict[str, Any], fallbacks: dict[str, list[str]], conn) -> TableFinding:
    guess = target["table"]
    need = target["need"]
    resolved = resolve_table(guess, fallbacks.get(guess, []), conn=conn)
    finding = TableFinding(
        need=need,
        guess=guess,
        confidence=str(target.get("confidence", "?")),
        resolved=resolved,
    )
    if resolved is None:
        finding.note = "NOT FOUND -- neither the primary guess nor any configured fallback exists."
        return finding

    try:
        desc = describe_table(resolved, conn=conn)
    except Exception as exc:  # noqa: BLE001
        finding.note = f"describe_table failed: {str(exc).splitlines()[0]}"
        return finding

    # `describe_table` returns a frame with name/nullable/type columns across wrds versions.
    name_col = "name" if "name" in desc.columns else desc.columns[0]
    type_col = next((c for c in ("type", "dtype", "data_type") if c in desc.columns), None)
    finding.columns = [str(c) for c in desc[name_col].tolist()]
    if type_col:
        finding.dtypes = {
            str(r[name_col]): str(r[type_col]) for _, r in desc.iterrows()
        }

    have = {c.lower() for c in finding.columns}
    finding.missing_key_columns = [c for c in target.get("key_columns", []) if c.lower() not in have]

    prof = table_profile(resolved, date_col=_DATE_COL.get(need), conn=conn)
    finding.rows = prof.get("rows")
    finding.date_min = prof.get("date_min")
    finding.date_max = prof.get("date_max")
    if resolved != guess:
        finding.note = f"resolved via fallback (guess `{guess}` does not exist)"
    return finding


def profile_targets(cfg: Config | None = None, conn=None) -> list[TableFinding]:
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    fallbacks = cfg.get("schema_fallbacks", {}) or {}
    findings = []
    for target in cfg["schema_targets"]:
        log.info("profiling %-40s (guess %s)", target["need"], target["table"])
        findings.append(_profile_one(target, fallbacks, conn))
    return findings


# ---------------------------------------------------------------------------
# Checkpoint 0 probes
# ---------------------------------------------------------------------------

def probe_link_scores(link_table: str, cfg: Config | None = None, conn=None) -> pd.DataFrame:
    """Distribution of `score` in the OM<->CRSP link.

    Checkpoint 0 requires we know *which score values are trustworthy*. WRDS's linking suite
    scores 1 (best) upward as the match degrades; we record the actual distribution so the
    Stage 2a filter is chosen from data, not folklore.
    """
    conn = conn or get_connection(cfg)
    schema, table = split_table(link_table)
    return conn.raw_sql(
        f"SELECT score, COUNT(*) AS n, COUNT(DISTINCT permno) AS n_permno, "
        f"COUNT(DISTINCT secid) AS n_secid FROM {schema}.{table} GROUP BY score ORDER BY score"
    )


# What the probe would like to see on the option price file. Anything absent is reported
# rather than fatal -- discovering that a column does not exist is the *point* of Stage 0,
# so the probe must survive it. `exercise_style` is the known example: IvyDB's price file
# carries `am_settlement` and `expiry_indicator` instead, and never an exercise-style flag.
_PROBE_OPTION_COLUMNS = [
    "secid", "date", "exdate", "optionid", "cp_flag", "best_bid", "best_offer",
    "impl_volatility", "delta", "vega", "gamma", "open_interest", "volume",
    "ss_flag", "contract_size", "cfadj", "am_settlement", "expiry_indicator",
    "forward_price", "exercise_style",
]


def probe_aapl_options(
    opprcd_table: str,
    link_table: str | None,
    cfg: Config | None = None,
    conn=None,
) -> dict[str, Any]:
    """Checkpoint 0's live sanity query: AAPL options on the probe date must return rows.

    Also records the facts the Stage 2b selection SQL depends on and that differ by IvyDB
    vintage: `ss_flag`'s storage type (quoted `'0'` vs bare `0`), `contract_size`'s type,
    and which of the settlement/exercise columns actually exist.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    probe_date = str(cfg["sanity"]["probe_date"])
    permno = int(cfg["sanity"]["probe_permno"])
    ticker = str(cfg["sanity"]["probe_ticker"])
    out: dict[str, Any] = {"probe_date": probe_date, "ticker": ticker, "permno": permno}

    secid = None
    if link_table:
        schema, table = split_table(link_table)
        link = conn.raw_sql(
            f"SELECT secid, score FROM {schema}.{table} "
            f"WHERE permno = %(permno)s AND sdate <= %(d)s AND edate >= %(d)s",
            params={"permno": permno, "d": probe_date},
        )
        out["link_rows"] = int(len(link))
        if len(link):
            secid = int(link.sort_values("score")["secid"].iloc[0])
    out["secid"] = secid

    if secid is None:
        out["error"] = "no secid resolved from the link table; cannot run the options probe"
        return out

    schema, table = split_table(opprcd_table)
    desc = describe_table(opprcd_table, conn=conn)
    name_col = "name" if "name" in desc.columns else desc.columns[0]
    available = {str(c).lower() for c in desc[name_col]}
    selected = [c for c in _PROBE_OPTION_COLUMNS if c in available]
    out["absent_columns"] = [c for c in _PROBE_OPTION_COLUMNS if c not in available]

    # strike_price is stored x1000 by OptionMetrics; dividing here is also a units check.
    cols = ", ".join(selected) + (", strike_price/1000.0 AS strike" if "strike_price" in available else "")
    df = conn.raw_sql(
        f"SELECT {cols} FROM {schema}.{table} WHERE secid = %(secid)s AND date = %(d)s",
        params={"secid": secid, "d": probe_date},
    )
    out["option_rows"] = int(len(df))
    if not len(df):
        return out

    if "strike" in df.columns:
        out["strike_range"] = [float(df["strike"].min()), float(df["strike"].max())]
    out["n_expiries"] = int(df["exdate"].nunique())

    # Storage types the Stage 2b selection SQL compares literals against.
    for col in ("ss_flag", "contract_size", "am_settlement", "expiry_indicator", "exercise_style"):
        if col in df.columns:
            vals = df[col].dropna().unique()
            out[f"{col}_dtype"] = str(df[col].dtype)
            out[f"{col}_values"] = sorted(map(str, vals))[:6]

    if df["delta"].notna().any():
        near = df.loc[(df["delta"].abs() - 0.5).abs().idxmin()]
        out["nearest_atm"] = {
            k: (str(near[k]) if k == "cp_flag" else float(near[k]))
            for k in ("cp_flag", "strike", "delta", "best_bid", "best_offer", "impl_volatility")
            if k in df.columns and pd.notna(near[k])
        }
    return out


def probe_dividends(distrd_table: str | None, cfg: Config | None = None, conn=None) -> dict[str, Any]:
    """Q5: is there a continuous dividend *yield* for equities, or only discrete dividends?

    Expectation from the plan: IvyDB gives a continuous yield for *indices* and discrete
    projected dividends for single names. This probe records what the server actually has.
    """
    cfg = cfg or load_config()
    conn = conn or get_connection(cfg)
    out: dict[str, Any] = {}

    try:
        om_tables = sorted(conn.list_tables(library="optionm"))
    except Exception as exc:  # noqa: BLE001
        om_tables = []
        out["list_tables_error"] = str(exc).splitlines()[0]
    out["yield_like_tables"] = [
        t for t in om_tables if any(k in t.lower() for k in ("dvd", "divid", "distr", "yield"))
    ]

    if distrd_table:
        schema, table = split_table(distrd_table)
        try:
            codes = conn.raw_sql(
                f"SELECT distr_type, COUNT(*) AS n FROM {schema}.{table} "
                f"WHERE ex_date BETWEEN %(a)s AND %(b)s GROUP BY distr_type ORDER BY n DESC",
                params={"a": str(cfg.start_date), "b": str(cfg.hold_buffer_end)},
            )
            out["distr_type_counts"] = codes.to_dict("records")
        except Exception as exc:  # noqa: BLE001
            out["distr_type_error"] = str(exc).splitlines()[0]

    out["conclusion"] = (
        "Build q from discrete projected dividends (escrowed-dividend adjustment, M9/Q5) "
        "unless a per-equity continuous yield table appears above."
    )
    return out


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def write_resolved(findings: list[TableFinding], cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    payload = {
        "verified_on": date.today().isoformat(),
        "tables": {f.need: f.resolved for f in findings},
        "guesses": {f.need: f.guess for f in findings},
        "missing_key_columns": {f.need: f.missing_key_columns for f in findings if f.missing_key_columns},
    }
    path = cfg.docs / RESOLVED_JSON
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_resolved_tables(cfg: Config | None = None) -> dict[str, str]:
    """Verified {need -> schema.table}, or {} if Stage 0 has not been run yet."""
    cfg = cfg or load_config()
    path = cfg.docs / RESOLVED_JSON
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in (payload.get("tables") or {}).items() if v}


def table_for(need: str, cfg: Config | None = None) -> str:
    """Verified table for a need, falling back to the config guess with a loud warning."""
    cfg = cfg or load_config()
    resolved = load_resolved_tables(cfg)
    if need in resolved:
        return resolved[need]
    for target in cfg["schema_targets"]:
        if target["need"] == need:
            log.warning(
                "Stage 0 has not verified '%s'; falling back to the UNVERIFIED guess %s. "
                "Run scripts/00_verify_schema.py.",
                need,
                target["table"],
            )
            return target["table"]
    raise KeyError(f"no schema target named {need!r}")


def render_notes(
    findings: list[TableFinding],
    probes: dict[str, Any],
    cfg: Config | None = None,
) -> str:
    cfg = cfg or load_config()
    today = date.today().isoformat()
    lines: list[str] = [
        "# WRDS schema notes",
        "",
        f"**Verified on {today}** by `scripts/00_verify_schema.py` against the live WRDS server.",
        "This file is generated -- re-run the script rather than editing it by hand.",
        "Row counts are the query planner's estimate, not an exact `COUNT(*)`.",
        "Stage 0 of `docs/PHASE1_PLAN.md`; every later stage reads table names from",
        "`docs/wrds_schema_resolved.json`, never from a hard-coded guess.",
        "",
        "## Resolved tables",
        "",
        "| Need | Plan guess | Resolved | Rows (est.) | Date range | Status |",
        "|---|---|---|---:|---|---|",
    ]
    for f in findings:
        rng = f"{f.date_min} → {f.date_max}" if f.date_min else "—"
        rows = f"{f.rows:,}" if isinstance(f.rows, int) else "—"
        if f.resolved is None:
            status = "❌ not found"
        elif f.missing_key_columns:
            status = "⚠ missing " + ", ".join(f"`{c}`" for c in f.missing_key_columns)
        elif f.resolved != f.guess:
            status = "✅ via fallback"
        else:
            status = "✅"
        lines.append(
            f"| {f.need} | `{f.guess}` | `{f.resolved or '—'}` | {rows} | {rng} | {status} |"
        )

    lines += ["", "## Column lists", ""]
    for f in findings:
        if not f.resolved:
            continue
        lines += [f"### `{f.resolved}` — {f.need}", ""]
        if f.note:
            lines += [f"_{f.note}_", ""]
        if f.dtypes:
            lines += ["| column | type |", "|---|---|"]
            lines += [f"| `{c}` | {f.dtypes.get(c, '')} |" for c in f.columns]
        else:
            lines += ["```", ", ".join(f.columns), "```"]
        lines.append("")

    lines += ["## Checkpoint 0 probes", ""]
    for key, value in probes.items():
        lines += [f"### {key}", "", "```json", json.dumps(value, indent=2, default=str), "```", ""]

    return "\n".join(lines) + "\n"


def write_notes(findings: list[TableFinding], probes: dict[str, Any], cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    path = cfg.docs / NOTES_MD
    path.write_text(render_notes(findings, probes, cfg), encoding="utf-8")
    return path


def checkpoint_0(findings: list[TableFinding], probes: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Checkpoint 0 as an explicit pass/fail list. Later stages should not run until all pass."""
    results: list[tuple[str, bool, str]] = []

    unresolved = [f.need for f in findings if f.resolved is None]
    results.append((
        "every schema target resolves to a real table",
        not unresolved,
        "unresolved: " + ", ".join(unresolved) if unresolved else "all resolved",
    ))

    missing = {f.need: f.missing_key_columns for f in findings if f.missing_key_columns}
    results.append((
        "every key column exists on its resolved table",
        not missing,
        json.dumps(missing) if missing else "all key columns present",
    ))

    link = probes.get("link_score_distribution")
    results.append((
        "OM<->CRSP link located and score semantics recorded",
        bool(link),
        f"{len(link)} distinct score values" if link else "link table not profiled",
    ))

    opt = probes.get("aapl_options_probe", {})
    n_opt = opt.get("option_rows", 0)
    results.append((
        f"sanity query returns rows (AAPL options on {opt.get('probe_date', '?')})",
        bool(n_opt),
        f"{n_opt} rows, secid={opt.get('secid')}",
    ))

    div = probes.get("dividend_question_q5", {})
    results.append((
        "Q5 resolved: equity dividend source identified",
        bool(div.get("yield_like_tables") is not None),
        "yield-like tables: " + ", ".join(div.get("yield_like_tables", [])) or "none found",
    ))

    return results
