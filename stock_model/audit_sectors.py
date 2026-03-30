import csv
import time
import yfinance as yf

INPUT_CSV  = "Data/raw_data/sector_map.csv"
OUTPUT_CSV = "Data/raw_data/sector_audit.csv"

# Mapping: our label -> set of acceptable yfinance labels
MATCH_MAP = {
    "Technology":  {"Technology", "Information Technology"},
    "Consumer":    {"Consumer Cyclical", "Consumer Defensive"},
    "Telecom":     {"Communication Services"},
    "Healthcare":  {"Healthcare", "Health Care"},
    "Financials":  {"Financial Services", "Financials"},
    "Industrials": {"Industrials"},
    "Energy":      {"Energy"},
    "Materials":   {"Basic Materials", "Materials"},
    "Real Estate": {"Real Estate"},
    "Utilities":   {"Utilities"},
    "Other":       None,   # always True
}

def is_match(our_sector, yf_sector):
    if our_sector == "Other":
        return True
    if yf_sector is None:
        return False
    accepted = MATCH_MAP.get(our_sector, set())
    return yf_sector in accepted

def main():
    # Read input
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["ticker"].strip()]

    results = []
    mismatches = []

    for i, row in enumerate(rows, 1):
        ticker  = row["ticker"].strip()
        company = row["company"].strip()
        our_sec = row["sector"].strip()

        try:
            info    = yf.Ticker(ticker).info
            yf_sec  = info.get("sector") or None
        except Exception as e:
            yf_sec = None

        matched = is_match(our_sec, yf_sec)
        results.append({
            "ticker":         ticker,
            "company":        company,
            "our_sector":     our_sec,
            "yfinance_sector": yf_sec or "",
            "match":          matched,
        })

        if not matched:
            mismatches.append((ticker, company, our_sec, yf_sec or "N/A"))

        if i % 25 == 0:
            print(f"  [{i}/{len(rows)}] checked {ticker} — "
                  f"our={our_sec}, yf={yf_sec}, match={matched}")

        time.sleep(0.5)

    # Write output CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker","company","our_sector","yfinance_sector","match"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total      = len(results)
    mismatch_n = len(mismatches)
    print("\n" + "="*60)
    print(f"SUMMARY: {total} tickers checked, {mismatch_n} mismatches")
    print("="*60)
    if mismatches:
        print(f"{'Ticker':<8}  {'Our sector':<14}  {'yfinance sector':<25}  Company")
        print("-"*80)
        for ticker, company, our_sec, yf_sec in mismatches:
            print(f"{ticker:<8}  {our_sec:<14}  {yf_sec:<25}  {company}")
    else:
        print("All sectors match!")
    print(f"\nFull results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
