# Unusual Whales data archive

Pulls everything the Unusual Whales API plan allows into a local archive, so backtests and new
ideas can run against it long after the trial or subscription ends.

**The data stays local.** `uw_data/data/` is gitignored. This repo is public and the data is
licensed, so never commit it. Back the folder up somewhere private (an external drive or a private
cloud folder).

## Run

```bash
pip install requests pandas yfinance        # yfinance: the stock model's S&P 500 list loader
export UW_API_KEY=your-key                  # never commit the key
cd Trading
python3 uw_data/harvest.py --plan           # how much is left to pull, no data requests
python3 uw_data/harvest.py                  # harvest; stops at the daily limit by itself
```

Run it once a day until the trial ends. It skips anything already saved and picks up where it
stopped, and the daily request count resets at 8 PM ET. `--only p1` (or snap/p2/p3) runs a single
tier, and `--max-requests N` caps a run.

On first run it finds the plan's real lookback by probing market tide (about 10 requests) and
caches it in `data/state.json`.

## What gets pulled

| Tier | Requests | Datasets |
|---|---|---|
| snap | ~40 per day | Prediction-market smart money, unusual markets, whales, insiders; movers; economic, FDA and congress calendars. **These have no history**, so each day you skip is lost. |
| p1 | ~6 per ticker (S&P 500 + ETFs) | Full history in one call each: daily options volume and premium, greek exposure (GEX), IV rank, risk-reversal skew, daily bars, short volume |
| p2 | ~25 per session | Market tide (all and OTM-only), 11 sector tides, SPY/QQQ/IWM/DIA ETF tide, SPY/QQQ/IWM GEX by strike, market OI change, top net impact, unusual activity, daily report, **all flow alerts** (paged) |
| p3 | 3 per focus ticker per session | Dark-pool prints (top 500 by premium), net premium ticks, OI change. Focus list is in `FOCUS` in `harvest.py`, or pass `--focus AAPL,NVDA,...` |

With about 2 years of lookback this is roughly 55–60k requests, which is about 4 days at 15k per
day. Snapshots run first every day; the rest goes newest date first, so if the trial ends early
the most recent data is already saved.

The **full options tape** (every option trade, as one zip per day) may also be in the plan.
`python3 uw_data/harvest.py --probe-full-tape` checks access and the size of one day. The files
are large, so download only the days you need.

## Using it

```python
from uw_data.loader import load, datasets
datasets()                       # dataset -> file count
ov = load("options_volume")      # _key = ticker
tide = load("market_tide")       # _key = date
alerts = load("flow_alerts")     # _key = date
```

Responses are stored exactly as the API returned them (`data/raw/<dataset>/<key>.json.gz`), so a
field nobody needs today is still there for the idea you have next year. Failed requests leave a
`.err` file with the status and message (for example 403 = outside the plan's lookback) and are
not retried. Delete the `.err` files for a dataset to retry it after upgrading the plan.
