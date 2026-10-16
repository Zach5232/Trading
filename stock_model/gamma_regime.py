"""
gamma_regime.py

SPY dealer-gamma regime from the Unusual Whales archive (private UW-Data repo), for sizing.

Finding (UW-Data studies/uw_studies.py, Oct 2023 - Oct 2026, 746 days): the more negative SPY's
net dealer gamma, the bigger the next moves - most-negative fifth: 1.07% average next-day move,
5.3% chance of a -2% day; most-positive fifth: 0.43% and 0%. It holds within the same recent-
volatility bucket, so it adds information beyond realized vol. It does NOT predict direction
(returns after negative-gamma days were slightly higher), so use it for size and stop width only.

Thresholds are rolling quantiles of the past 252 days, so a date's label uses only earlier data.

Data: point UW_DATA_DIR at a checkout of the private UW-Data repo (default ../../uw-data).
The archive is licensed and stays out of this public repo.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pandas as pd

UW_DATA_DIR = Path(os.environ.get("UW_DATA_DIR", Path(__file__).resolve().parents[2] / "uw-data"))
GEX_FILE = UW_DATA_DIR / "uw_data" / "data" / "raw" / "greek_exposure" / "SPY.json.gz"

# position-size multiplier by bucket (most-negative fifth -> half size)
SIZE_MULT = {"NEG_STRONG": 0.5, "NEG": 0.75, "NEUTRAL": 1.0, "POS": 1.0, "UNKNOWN": 1.0}


def load_spy_gamma(path: Path = GEX_FILE) -> pd.Series | None:
    """SPY net gamma (call + put) by date, or None if the archive isn't available."""
    if not path.exists():
        return None
    rows = json.load(gzip.open(path, "rt")).get("data", [])
    df = pd.DataFrame(rows)
    df["net_gamma"] = pd.to_numeric(df["call_gamma"]) + pd.to_numeric(df["put_gamma"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["net_gamma"].sort_index()


def gamma_buckets(net_gamma: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Label each day by where its net gamma sits among the previous `window` days."""
    past = net_gamma.shift(1).rolling(window, min_periods=min_periods)
    q20, q40, q80 = past.quantile(0.2), past.quantile(0.4), past.quantile(0.8)
    lab = pd.Series("NEUTRAL", index=net_gamma.index)
    lab[net_gamma <= q40] = "NEG"
    lab[net_gamma <= q20] = "NEG_STRONG"
    lab[net_gamma >= q80] = "POS"
    lab[q20.isna()] = "UNKNOWN"
    return lab


def label_for(dates: pd.Series, buckets: pd.Series) -> list[str]:
    """Bucket as of the last trading day strictly before each date (no lookahead)."""
    left = pd.DataFrame({"d": pd.to_datetime(pd.Series(dates).values), "i": range(len(dates))}).sort_values("d")
    right = pd.DataFrame({"d": buckets.index, "bucket": buckets.values}).sort_values("d")
    m = pd.merge_asof(left, right, on="d", direction="backward", allow_exact_matches=False)
    return m.sort_values("i")["bucket"].fillna("UNKNOWN").tolist()


def size_multiplier(bucket: str) -> float:
    return SIZE_MULT.get(bucket, 1.0)
