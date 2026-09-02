"""Configuration loading and path resolution.

Single source of truth for where things live. Every module imports `load_config()` rather
than constructing paths itself, so the repo can be moved or the sample window changed in
one place (config/config.yaml).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # optional; the repo works without a .env if WRDS_USERNAME is exported
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def repo_root() -> Path:
    """Repo root, resolved from this file's location (src/vrp/config.py -> ../..)."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    # ---- dict-ish access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # ---- paths -----------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a named path from config['paths'], creating it if needed."""
        p = self.root / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_raw(self) -> Path:
        return self.path("data_raw")

    @property
    def data_interim(self) -> Path:
        return self.path("data_interim")

    @property
    def data_processed(self) -> Path:
        return self.path("data_processed")

    @property
    def output_tables(self) -> Path:
        return self.path("output_tables")

    @property
    def output_figures(self) -> Path:
        return self.path("output_figures")

    @property
    def docs(self) -> Path:
        return self.path("docs")

    # ---- dates -----------------------------------------------------------
    @staticmethod
    def _d(value: str) -> date:
        return date.fromisoformat(str(value))

    @property
    def start_date(self) -> date:
        return self._d(self.raw["sample"]["start_date"])

    @property
    def end_date(self) -> date:
        return self._d(self.raw["sample"]["end_date"])

    @property
    def burnin_start(self) -> date:
        return self._d(self.raw["sample"]["burnin_start"])

    @property
    def hold_buffer_end(self) -> date:
        return self._d(self.raw["sample"]["hold_buffer_end"])

    # ---- credentials -----------------------------------------------------
    @property
    def wrds_username(self) -> str | None:
        """WRDS_USERNAME from the environment/.env wins over config.yaml.

        The password never lives here: it belongs in ~/.pgpass (see README).
        """
        return os.environ.get("WRDS_USERNAME") or self.raw.get("wrds", {}).get("username")


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    root = repo_root()
    if load_dotenv is not None:
        load_dotenv(root / ".env")
    cfg_path = Path(path) if path is not None else root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root)
