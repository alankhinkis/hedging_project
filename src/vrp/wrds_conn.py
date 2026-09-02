"""Single WRDS connection factory plus a query -> parquet cache.

Two jobs:

1. One connection per process. `wrds.Connection()` is slow to open and WRDS limits
   concurrent sessions, so everything goes through `get_connection()`.
2. Never pull the same rows twice. `cached_query()` keys a parquet file on a hash of the
   SQL *and* its bound parameters, so re-running a stage is free. Stage 4 will be re-run
   many times; it must never re-hit WRDS to do it.

Credentials: WRDS_USERNAME comes from the environment or .env. The password belongs in
~/.pgpass so it is never typed, logged, or committed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import Config, load_config

log = logging.getLogger(__name__)

_CONN = None


class WRDSUnavailable(RuntimeError):
    """Raised when a WRDS connection cannot be established.

    Carries the underlying cause so scripts can print an actionable message instead of a
    psycopg2 stack trace.
    """


def get_connection(cfg: Config | None = None, *, force_new: bool = False):
    """Return the process-wide `wrds.Connection`, opening it on first use."""
    global _CONN
    if _CONN is not None and not force_new:
        return _CONN

    cfg = cfg or load_config()
    username = cfg.wrds_username
    if not username:
        raise WRDSUnavailable(
            "WRDS_USERNAME is not set. Create a .env at the repo root containing\n"
            "    WRDS_USERNAME=your_wrds_login\n"
            "and store the password in ~/.pgpass (see README.md). Never put the password in .env."
        )

    try:
        import wrds  # imported lazily so the rest of the package works without it
    except ImportError as exc:  # pragma: no cover
        raise WRDSUnavailable("The `wrds` package is not installed: pip install wrds") from exc

    log.info("Opening WRDS connection as %s ...", username)
    t0 = time.time()
    try:
        _CONN = wrds.Connection(wrds_username=username)
    except Exception as exc:  # noqa: BLE001 - surface any auth/network failure uniformly
        raise WRDSUnavailable(
            f"Could not connect to WRDS as '{username}': {exc}\n"
            "Check the username, and that ~/.pgpass has a line of the form\n"
            "    wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>"
        ) from exc
    log.info("WRDS connection established in %.1fs", time.time() - t0)
    return _CONN


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _normalize_sql(sql: str) -> str:
    """Collapse whitespace so cosmetic reformatting does not invalidate the cache."""
    return re.sub(r"\s+", " ", sql).strip()


def query_key(sql: str, params: dict[str, Any] | None = None) -> str:
    payload = json.dumps(
        {"sql": _normalize_sql(sql), "params": _jsonable(params or {})},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _jsonable(obj: Any) -> Any:
    """Make tuples/sets/dates stable and comparable inside the cache key."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return obj


def cached_query(
    sql: str,
    params: dict[str, Any] | None = None,
    *,
    name: str = "query",
    cfg: Config | None = None,
    conn=None,
    force: bool = False,
    date_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Run `sql` on WRDS and cache the result to parquet under data/raw.

    Parameters
    ----------
    name : short human-readable slug; the cache file is `<name>__<hash>.parquet` so the
        directory listing stays readable while the hash guarantees correctness.
    force : re-pull even if the cache file exists (use after changing a filter's meaning
        without changing the SQL text, which cannot happen -- so mostly for debugging).
    date_cols : columns to coerce to datetime64 on the way in and out, since parquet
        round-trips of object-dtype dates are lossy.
    """
    cfg = cfg or load_config()
    use_cache = bool(cfg.raw.get("wrds", {}).get("cache", True))
    key = query_key(sql, params)
    path = cfg.data_raw / f"{name}__{key}.parquet"

    if use_cache and path.exists() and not force:
        log.info("cache hit  %s", path.name)
        df = pd.read_parquet(path)
        return _coerce_dates(df, date_cols)

    conn = conn or get_connection(cfg)
    log.info("cache miss %s -- querying WRDS", path.name)
    t0 = time.time()
    df = conn.raw_sql(sql, params=params or {}, date_cols=list(date_cols) or None)
    log.info("  %s rows in %.1fs", f"{len(df):,}", time.time() - t0)

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        # Sidecar so a stale cache directory can always be explained.
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "name": name,
                    "sql": _normalize_sql(sql),
                    "params": _jsonable(params or {}),
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                    "pulled_at": pd.Timestamp.utcnow().isoformat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return _coerce_dates(df, date_cols)


def _coerce_dates(df: pd.DataFrame, date_cols: Iterable[str]) -> pd.DataFrame:
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------------------
# Schema reconnaissance helpers (Stage 0)
# ---------------------------------------------------------------------------

def split_table(qualified: str) -> tuple[str, str]:
    schema, _, table = qualified.partition(".")
    if not table:
        raise ValueError(f"expected 'schema.table', got {qualified!r}")
    return schema, table


def table_exists(qualified: str, *, conn=None, cfg: Config | None = None) -> bool:
    conn = conn or get_connection(cfg)
    schema, table = split_table(qualified)
    try:
        return table in set(conn.list_tables(library=schema))
    except Exception:  # noqa: BLE001 - a missing library raises, which is a clean "no"
        return False


def resolve_table(
    primary: str,
    fallbacks: Sequence[str] = (),
    *,
    conn=None,
    cfg: Config | None = None,
) -> str | None:
    """Return the first of `primary`, *fallbacks that actually exists on WRDS."""
    conn = conn or get_connection(cfg)
    for candidate in [primary, *fallbacks]:
        if table_exists(candidate, conn=conn):
            return candidate
    return None


def describe_table(qualified: str, *, conn=None, cfg: Config | None = None) -> pd.DataFrame:
    """Column name/dtype listing for a table (thin wrapper over `describe_table`)."""
    conn = conn or get_connection(cfg)
    schema, table = split_table(qualified)
    return conn.describe_table(library=schema, table=table)


def resolve_columns(
    table: str,
    wanted: dict[str, list[str]],
    *,
    conn=None,
    cfg: Config | None = None,
    optional: Sequence[str] = (),
) -> dict[str, str]:
    """Map a logical column name onto whichever candidate actually exists on `table`.

    CRSP and OptionMetrics vintages differ (the CRSP v2 tables rename nearly everything),
    so no pull hard-codes a column name. A logical name listed in `optional` is omitted
    from the result when none of its candidates exist; anything else raises.
    """
    desc = describe_table(table, conn=conn, cfg=cfg)
    name_col = "name" if "name" in desc.columns else desc.columns[0]
    available = {str(c).lower() for c in desc[name_col]}
    out: dict[str, str] = {}
    for logical, candidates in wanted.items():
        hit = next((c for c in candidates if c.lower() in available), None)
        if hit is None:
            if logical in optional:
                continue
            raise KeyError(
                f"{table} has no column for '{logical}'; tried {candidates}. "
                f"Available: {sorted(available)}"
            )
        out[logical] = hit
    return out


def table_profile(
    qualified: str,
    *,
    date_col: str | None = None,
    conn=None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Row count and, if `date_col` is given, the date range -- the two facts Stage 0
    requires to be written down for every table.

    The count is the planner's estimate (`wrds.Connection.get_row_count`, an EXPLAIN), not
    an exact `COUNT(*)`. A literal count on `optionm.opprcd2019` or `crsp.dsf` is a full
    table scan measured in minutes, and Stage 0 only needs the order of magnitude.
    """
    conn = conn or get_connection(cfg)
    schema, table = split_table(qualified)
    out: dict[str, Any] = {"table": qualified, "rows_are_estimated": True}
    try:
        out["rows"] = int(conn.get_row_count(schema, table))
    except Exception as exc:  # noqa: BLE001
        out["rows"] = None
        out["rows_error"] = str(exc).splitlines()[0]
    if date_col:
        try:
            rng = conn.raw_sql(
                f"SELECT MIN({date_col}) AS min_d, MAX({date_col}) AS max_d FROM {schema}.{table}"
            )
            out["date_col"] = date_col
            out["date_min"] = str(rng["min_d"].iloc[0])
            out["date_max"] = str(rng["max_d"].iloc[0])
        except Exception as exc:  # noqa: BLE001
            out["date_error"] = str(exc).splitlines()[0]
    return out


def cache_manifest(cfg: Config | None = None) -> pd.DataFrame:
    """Every cached pull, for the reproducibility appendix."""
    cfg = cfg or load_config()
    rows = []
    for sidecar in sorted(Path(cfg.data_raw).glob("*.json")):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        rows.append(
            {
                "file": sidecar.with_suffix(".parquet").name,
                "name": meta.get("name"),
                "rows": meta.get("rows"),
                "pulled_at": meta.get("pulled_at"),
                "sql": meta.get("sql", "")[:200],
            }
        )
    return pd.DataFrame(rows)
