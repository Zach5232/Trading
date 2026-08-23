"""
build_universe.py

Builds stock_model/Data/raw_data/master_universe.json from:
  - S&P 500:            cached Data/raw_data/sp500_tickers.csv (falls back to
                         a live Wikipedia scrape if that cache is missing)
  - Russell 1000/2000/Midcap: local iShares fund holdings exports
                         (IWB / IWM / IWR), found directly under stock_model/

The iShares "XLS" files are actually Excel 2003 SpreadsheetML XML with a
few unescaped '&' characters in unrelated disclaimer links, which breaks
strict XML parsing — so the Holdings worksheet is extracted with a regex
row/cell scan instead of xml.etree.

Run from anywhere:
    python3 stock_model/scripts/build_universe.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STOCK_MODEL_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = STOCK_MODEL_DIR / "Data" / "raw_data"
SP500_CACHE = RAW_DATA_DIR / "sp500_tickers.csv"
OUTPUT_PATH = RAW_DATA_DIR / "master_universe.json"

# Add tickers here when yfinance consistently fails
# to download them. Re-run build_universe.py after
# adding new exclusions.
DELISTED_EXCLUSIONS = {
    'BFA',    # delisted/renamed
    'HOLX',   # no yfinance data
    'HEIA',   # delisted/renamed
    'LENB',   # delisted/renamed
    'UHALB',  # delisted/renamed
}

ISHARES_FILES = {
    "r1000": STOCK_MODEL_DIR / "iShares-Russell-1000-ETF_fund.xls",
    "r2000": STOCK_MODEL_DIR / "iShares-Russell-2000-ETF_fund.xls",
    "midcap": STOCK_MODEL_DIR / "iShares-Russell-Mid-Cap-ETF_fund.xls",
}

# Curated top-AUM ETF list — the same fallback used previously; ETFdb's
# screener API is unauthenticated/undocumented and not worth depending on
# for a one-off universe build.
ETF_TICKERS = [
    'SPY','IVV','VOO','VTI','QQQ','IWM','DIA','MDY',
    'IWF','IWD','IWO','IWN','VTV','VUG','VO','VB',
    'XLK','XLV','XLF','XLY','XLE','XLI','XLB','XLU','XLRE','XLC',
    'VGT','VHT','VFH','VCR','VDE','VIS','VAW','VPU','VNQ',
    'SOXX','SMH','IGV','SKYY','CIBR','HACK','BOTZ','AIQ',
    'ARKK','ARKQ','ARKF','ARKG','ARKX',
    'XBI','IBB','LABU',
    'XOP','OIH','GDX','GDXJ','SLV','GLD','USO','UNG',
    'REMX','LIT','ICLN','TAN','QCLN',
    'KRE','KBE','FINX','IPAY',
    'XRT','JETS','ITB','XHB',
    'EFA','EEM','INDA','EWJ','FXI','EWZ','KWEB',
    'VEA','VWO','EWU','EWG','EWC',
    'TLT','HYG','LQD','EMB',
    'BITO','BLOK',
    'TQQQ','SOXL','UPRO','TNA','FAS','TECL',
    'SQQQ','SPXU','SPXS',
]


def parse_ishares_holdings(path: Path) -> list[dict]:
    """
    Extracts the 'Holdings' worksheet from an iShares SpreadsheetML export
    as a list of {column: value} dicts, via regex row/cell scanning (the
    file isn't valid XML — see module docstring).
    """
    content = path.read_text(encoding="utf-8")
    start = content.find('<ss:Worksheet ss:Name="Holdings">')
    end = content.find('<ss:Worksheet ss:Name="Historical">')
    if start == -1 or end == -1:
        raise ValueError(f"Could not locate Holdings worksheet in {path.name}")
    section = content[start:end]

    row_blocks = re.findall(r"<ss:Row[^>]*>(.*?)</ss:Row>", section, re.DOTALL)
    rows = []
    for block in row_blocks:
        cells = re.findall(r"<ss:Data[^>]*>(.*?)</ss:Data>", block, re.DOTALL)
        rows.append([html.unescape(c.strip()) for c in cells])

    header_idx = next((i for i, r in enumerate(rows) if r and r[0] == "Ticker"), None)
    if header_idx is None:
        raise ValueError(f"Could not find the 'Ticker' header row in {path.name}")
    header = rows[header_idx]

    return [
        dict(zip(header, r))
        for r in rows[header_idx + 1:]
        if len(r) == len(header)
    ]


def load_equity_tickers(path: Path, dash_fix: dict[str, str]) -> list[str]:
    """Equity-only tickers from an iShares holdings file, deduped and dash-normalized."""
    records = parse_ishares_holdings(path)
    tickers = set()
    for r in records:
        if r.get("Asset Class") != "Equity":
            continue
        t = (r.get("Ticker") or "").strip()
        if not t or t == "--":
            continue
        tickers.add(dash_fix.get(t, t))
    return sorted(tickers)


def exclude_delisted(tickers: list[str]) -> list[str]:
    """Filters out DELISTED_EXCLUSIONS from a ticker list."""
    return [t for t in tickers if t not in DELISTED_EXCLUSIONS]


def load_sp500() -> list[str]:
    if SP500_CACHE.exists():
        df = pd.read_csv(SP500_CACHE, header=None)
        tickers = sorted(set(df[0].dropna().astype(str).str.strip()))
        print(f"   S&P 500: {len(tickers)} tickers (from cache: {SP500_CACHE.name})")
        return tickers

    print("   No cached S&P 500 list — scraping Wikipedia...")
    table = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0
    )[0]
    tickers = sorted(set(table["Symbol"].str.replace(".", "-", regex=False)))
    print(f"   S&P 500: {len(tickers)} tickers (from Wikipedia)")
    return tickers


def main():
    print("=" * 60)
    print("BUILDING MASTER TICKER UNIVERSE")
    print("=" * 60)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading S&P 500...")
    sp500_raw = load_sp500()

    # iShares exports concatenate dual-class suffixes with no separator
    # (e.g. "BRKB", "BFB") instead of "BRK-B"/"BF-B". Correct any that
    # collide with a dash-containing S&P 500 ticker once collapsed —
    # names outside the S&P 500 cache are left in their native form.
    dash_fix = {t.replace("-", ""): t for t in sp500_raw if "-" in t}

    print("\n2. Parsing Russell 1000 holdings (IWB)...")
    r1000_raw = load_equity_tickers(ISHARES_FILES["r1000"], dash_fix)
    print(f"   Russell 1000: {len(r1000_raw)} equity tickers")

    print("\n3. Parsing Russell 2000 holdings (IWM)...")
    r2000_raw = load_equity_tickers(ISHARES_FILES["r2000"], dash_fix)
    print(f"   Russell 2000: {len(r2000_raw)} equity tickers")

    print("\n4. Parsing Russell Midcap holdings (IWR)...")
    midcap_raw = load_equity_tickers(ISHARES_FILES["midcap"], dash_fix)
    print(f"   Russell Midcap: {len(midcap_raw)} equity tickers")

    # Drop known-bad tickers (delisted/renamed/no yfinance data) before any
    # of the derived sets below are built, so the exclusion propagates
    # everywhere (r1000_extra, r2000_only, all_equity included).
    raw_combined = set(sp500_raw) | set(r1000_raw) | set(r2000_raw) | set(midcap_raw) | set(ETF_TICKERS)
    excluded_found = DELISTED_EXCLUSIONS & raw_combined
    print(f"\n>>> Excluded {len(excluded_found)} delisted/unavailable tickers")

    sp500 = exclude_delisted(sp500_raw)
    r1000 = exclude_delisted(r1000_raw)
    r2000 = exclude_delisted(r2000_raw)
    midcap = exclude_delisted(midcap_raw)

    print("\n5. Building derived sets...")
    sp500_set = set(sp500)
    r1000_set = set(r1000)
    r2000_set = set(r2000)

    r1000_extra = sorted(r1000_set - sp500_set)
    r2000_only = sorted(r2000_set - sp500_set - r1000_set)
    all_equity = sorted(sp500_set | r1000_set | r2000_set | set(midcap))

    print(f"   R1000 extra (not in S&P 500):        {len(r1000_extra):>5}")
    print(f"   R2000 only (not in S&P 500 or R1000): {len(r2000_only):>5}")
    print(f"   All equity (union):                   {len(all_equity):>5}")
    print(f"   ETFs (curated):                       {len(ETF_TICKERS):>5}")

    universe = {
        "sp500": sp500,
        "r1000": r1000,
        "r1000_extra": r1000_extra,
        "r2000": r2000,
        "r2000_only": r2000_only,
        "midcap": midcap,
        "etfs": sorted(set(exclude_delisted(ETF_TICKERS))),
        "all_equity": all_equity,
    }

    print(f"\n6. Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(universe, f, indent=2)
    print(f"   Saved. Total all_equity: {len(all_equity)}, plus {len(universe['etfs'])} ETFs.")


if __name__ == "__main__":
    main()
