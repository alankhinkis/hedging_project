"""vrp -- delta-hedged gains and the volatility risk premium in single-name equity options.

An extension of Bakshi & Kapadia (2003, RFS) from S&P 500 index options to the cross-section
of individual equities, 2017-2023. See docs/PHASE1_PLAN.md for the build plan and
docs/methodology.md for the decisions behind it.
"""

__version__ = "0.1.0"

from .config import Config, load_config, repo_root

__all__ = ["Config", "load_config", "repo_root", "__version__"]
