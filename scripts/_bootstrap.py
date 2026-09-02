"""Make `src/` importable when a script is run directly (no install required).

Also sets up logging once, so every stage script has the same output format.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def print_checkpoint(title: str, results: list[tuple[str, bool, str]]) -> bool:
    """Render a checkpoint's pass/fail list; return True iff everything passed."""
    width = max((len(r[0]) for r in results), default=0)
    print()
    print("=" * (width + 30))
    print(title)
    print("=" * (width + 30))
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name.ljust(width)}   {detail}")
    all_ok = all(ok for _, ok, _ in results)
    print("-" * (width + 30))
    print(f"  {'ALL CHECKS PASSED' if all_ok else 'CHECKPOINT FAILED -- do not proceed to the next stage'}")
    print()
    return all_ok
