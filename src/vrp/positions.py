"""Stage 4 assembly -- turn the Stage 2/3 panels into engine inputs.

Kept separate from `hedging.py` so the engine stays a pure function of its arguments and can
be tested on synthetic paths with no data layer at all. This module is where the real-data
hazards live, and there are three:

**Split normalisation (Q11).** Every position's price path is expressed in the share units
that prevailed on its own entry date, so S and the as-traded strike agree for the whole life
of the position. The engine cannot detect a violation -- it would just produce a large,
plausible, entirely fake P&L -- so it has to be right here.

**Rate matched to remaining maturity (M4).** The financing rate on day t is the zero rate
interpolated to the days still left on the option, not a fixed entry-date rate.

**Escrowed dividends (M9).** The PV subtracted on day t covers only ex-dates still ahead of
t, so it shrinks as dividends go ex during the hold.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config, load_config
from .hedging import OptionPosition

log = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0


def build_positions(selected: pd.DataFrame, *, quantity: float = 1.0) -> list[OptionPosition]:
    """One OptionPosition per selected contract, entered at the bid-ask midpoint."""
    out = []
    for row in selected.itertuples(index=False):
        out.append(OptionPosition(
            secid=int(row.secid),
            permno=int(row.permno),
            optionid=int(row.optionid),
            cp_flag=str(row.cp_flag),
            strike=float(row.strike),
            entry_date=pd.Timestamp(row.entry_date),
            expiry=pd.Timestamp(row.exdate),
            entry_price=float(row.mid),
            quantity=quantity,
        ))
    return out


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def build_rate_lookup(curve: pd.DataFrame) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    """date -> (days, rate) arrays, for fast per-position interpolation.

    Built once rather than per position: the curve has ~73,000 rows and re-grouping it 27,000
    times is the difference between seconds and an hour.
    """
    return {
        d: (g["days"].to_numpy(dtype="float64"), g["r"].to_numpy(dtype="float64"))
        for d, g in curve.sort_values(["date", "days"]).groupby("date")
    }


def position_rate_path(
    lookup: dict,
    dates: pd.DatetimeIndex,
    expiry: pd.Timestamp,
    *,
    fallback: float = np.nan,
) -> pd.Series:
    """Zero rate on each date, interpolated in days to the REMAINING maturity (M4).

    Flat extrapolation outside the quoted tenors: our maturities sit well inside the curve,
    so this only guards a date whose curve is unusually short, and extrapolating a yield
    curve linearly would be worse than holding the nearest point.
    """
    out = np.full(len(dates), fallback, dtype="float64")
    for i, d in enumerate(dates):
        entry = lookup.get(d)
        if entry is None:
            continue
        days_left = max((expiry - d).days, 1)
        out[i] = float(np.interp(days_left, entry[0], entry[1]))
    return pd.Series(out, index=dates).ffill().bfill()


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

def build_dividend_lookup(divs: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """secid -> its dividend schedule, for per-position PV without re-filtering."""
    if not len(divs):
        return {}
    cols = [c for c in ("ex_date", "amount", "declare_date") if c in divs.columns]
    return {int(s): g[cols].sort_values("ex_date").reset_index(drop=True)
            for s, g in divs.groupby("secid")}


def position_pv_dividends(
    sched: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    expiry: pd.Timestamp,
    rates: np.ndarray,
    *,
    known_only: bool = False,
) -> np.ndarray:
    """PV on each date of dividends with ex-dates in (date, expiry] (M9).

    Shrinks through the hold as dividends go ex. `known_only` restricts to dividends already
    declared as of each date, which removes look-ahead at the cost of missing dividends the
    market was visibly pricing -- the median declaration lead is 26 days against a ~30-day
    hold, so the two differ materially and both are computed in the robustness table.
    """
    pv = np.zeros(len(dates), dtype="float64")
    if sched is None or not len(sched):
        return pv
    ex = sched["ex_date"].to_numpy()
    amt = sched["amount"].to_numpy(dtype="float64")
    dec = sched["declare_date"].to_numpy() if "declare_date" in sched.columns else None

    for i, d in enumerate(dates):
        mask = (ex > np.datetime64(d)) & (ex <= np.datetime64(expiry))
        if known_only and dec is not None:
            mask &= (dec <= np.datetime64(d))
        if not mask.any():
            continue
        tau = (ex[mask] - np.datetime64(d)).astype("timedelta64[D]").astype("float64") / DAYS_PER_YEAR
        r = rates[i] if np.isfinite(rates[i]) else 0.0
        pv[i] = float(np.sum(amt[mask] * np.exp(-r * tau)))
    return pv


# ---------------------------------------------------------------------------
# Underlying paths
# ---------------------------------------------------------------------------

def build_price_lookup(prices: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """permno -> its cleaned daily price rows, indexed for slicing."""
    cols = ["date", "prc_abs", "cfacpr", "has_price"]
    px = prices.loc[:, ["permno", *cols]].sort_values(["permno", "date"])
    return {int(p): g[cols].reset_index(drop=True) for p, g in px.groupby("permno")}


def position_underlying(
    panel: pd.DataFrame,
    entry_date: pd.Timestamp,
    expiry: pd.Timestamp,
) -> pd.DataFrame:
    """The position's price path, normalised to its ENTRY-DATE adjustment factor (Q11).

        S_used(t) = |prc_t| / cfacpr_t * cfacpr_entry

    so entry-date S and the as-traded strike are in the same units, and a split mid-hold does
    not register as a 75% move. Returns (date, S) with non-trading days dropped -- the engine
    carries the delta forward across them.
    """
    sl = panel.loc[(panel["date"] >= entry_date) & (panel["date"] <= expiry)]
    sl = sl.loc[sl["has_price"]]
    if not len(sl):
        return pd.DataFrame(columns=["date", "S"])

    at_entry = sl.loc[sl["date"] == entry_date, "cfacpr"]
    if at_entry.empty:
        return pd.DataFrame(columns=["date", "S"])
    cfacpr_entry = float(at_entry.iloc[0])
    if not np.isfinite(cfacpr_entry) or cfacpr_entry == 0:
        return pd.DataFrame(columns=["date", "S"])

    cf = sl["cfacpr"].replace(0, np.nan)
    return pd.DataFrame({
        "date": sl["date"].to_numpy(),
        "S": (sl["prc_abs"] / cf * cfacpr_entry).to_numpy(dtype="float64"),
    })


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def prepare_inputs(
    position: OptionPosition,
    price_lookup: dict[int, pd.DataFrame],
    vol_lookup: dict[int, pd.Series],
    rate_lookup: dict,
    div_lookup: dict[int, pd.DataFrame],
    *,
    vol_col: str = "vol_hedge",
    known_only_dividends: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Everything one position needs: (underlying with pv_divs, vol path, rate path)."""
    panel = price_lookup.get(position.permno)
    if panel is None:
        raise KeyError(f"no price panel for permno {position.permno}")
    und = position_underlying(panel, position.entry_date, position.expiry)
    if len(und) < 2:
        raise ValueError(f"only {len(und)} usable price days for optionid {position.optionid}")

    dates = pd.DatetimeIndex(und["date"])
    rates = position_rate_path(rate_lookup, dates, position.expiry)
    und["pv_divs"] = position_pv_dividends(
        div_lookup.get(position.secid), dates, position.expiry,
        rates.to_numpy(), known_only=known_only_dividends,
    )

    vol = vol_lookup.get(position.permno)
    if vol is None:
        raise KeyError(f"no volatility path for permno {position.permno}")
    return und, vol.reindex(dates).ffill(), rates


def build_vol_lookup(vol_panel: pd.DataFrame, col: str = "vol_hedge") -> dict[int, pd.Series]:
    """permno -> date-indexed volatility series."""
    return {
        int(p): g.set_index("date")[col].sort_index()
        for p, g in vol_panel.loc[:, ["permno", "date", col]].groupby("permno")
    }
