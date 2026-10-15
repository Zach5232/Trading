"""
Read the raw Unusual Whales archive (uw_data/data/raw) into pandas -- no API calls.

    from uw_data.loader import load, datasets
    datasets()                         # what's been harvested, with file counts
    ov = load("options_volume")        # one row per (ticker, day), `_key` = ticker
    tide = load("market_tide")         # 5-min rows, `_key` = date
    fa = load("flow_alerts")           # every page of every day

Each row gets `_key` (the file's key: ticker, date, or date_ticker) so you can join back.
"""

import gzip
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw"


def datasets():
    return {p.name: sum(1 for _ in p.rglob("*.json.gz")) for p in sorted(RAW.iterdir()) if p.is_dir()}


def _rows(payload):
    data = payload.get("data", payload)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def iter_raw(name):
    """(key, payload) for every stored response of a dataset."""
    for f in sorted((RAW / name).rglob("*.json.gz")):
        key = f.parent.name if name == "flow_alerts" else f.name[:-len(".json.gz")]
        with gzip.open(f, "rt") as fh:
            yield key, json.load(fh)


def load(name):
    frames = []
    for key, payload in iter_raw(name):
        rows = _rows(payload)
        if rows:
            df = pd.json_normalize(rows)
            df.insert(0, "_key", key)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
