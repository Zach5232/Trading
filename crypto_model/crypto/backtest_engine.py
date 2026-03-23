"""
crypto/backtest_engine.py
=========================
Walk-forward weekend backtest implementing the locked Var1+Var2+Var4 system
for BTC-USD and ETH-USD.

Signal logic (all three filters must pass for LONG):
  Filter 1 — close > MA20
  Filter 2 — ATR14 today > ATR14 prior Friday   (Var2: volatility expanding)
  Filter 3 — Friday close > prior Friday close   (Var1: momentum confirming)

Timing:
  - Entry  : Saturday open  ≈  Friday close × (1 + SLIPPAGE_PCT)
  - Stop   : entry − ATR_MULT_STOP × ATR14
  - Target : entry + R_TARGET × (entry − stop)
  - Exit   : first of (Saturday stop/target hit) → (Sunday stop/target hit)
             → Sunday close  (TIME exit)
  - Var4 Monday hold: if TIME exit AND exit_price > entry, hold Mon bar
    with breakeven stop = entry; exit Mon close if no hit.

Position sizing:
  risk_dollars = account_equity × RISK_PCT_PER_TRADE
  units        = risk_dollars / (entry − stop)

Fees: Kraken taker 0.26% per leg (both entry and exit).
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import sys

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df, get_fridays

# ── Parameters ──────────────────────────────────────────────────────────────
SLIPPAGE_PCT       = 0.001    # 0.1% weekend spread
ATR_MULT_STOP      = 1.25     # stop = entry − 1.25 × ATR  (was 1.0)
R_TARGET           = 2.0      # target = entry + 2 × risk
STARTING_CAPITAL   = 500.0
RISK_PCT_PER_TRADE = 0.05     # 5% of equity per trade  (was 0.10)
TAX_RATE           = 0.32
FEE_PCT            = 0.0026   # Kraken taker 0.26% per leg

OUTPUT_DIR = Path(__file__).parent.parent / "Results" / "crypto_backtest"

REGIMES = {
    "2018 Bear":         ("2018-01-01", "2018-12-31"),
    "2019 Recovery":     ("2019-01-01", "2019-12-31"),
    "2020 COVID":        ("2020-01-01", "2020-12-31"),
    "2021 Bull":         ("2021-01-01", "2021-12-31"),
    "2022 Bear":         ("2022-01-01", "2022-12-31"),
    "2023-24 Recovery":  ("2023-01-01", "2024-12-31"),
    "2025-Present":      ("2025-01-01", "2099-12-31"),
}


# ── Exit simulation ──────────────────────────────────────────────────────────

def _simulate_exit(
    bars: pd.DataFrame,
    entry: float,
    stop: float,
    target: float,
    trailing_stop: bool = False,
    atr: float = 0.0,
    saturday_time_stop: bool = False,
):
    """
    Walk bars checking stop/target intrabar. Returns (price, type, note).

    When trailing_stop=True:
      - dynamic_stop starts at original stop
      - When bar high reaches 1R (entry + risk), move dynamic_stop to breakeven
      - When bar high reaches 1.5R, trail at (bar_high - 0.5 × ATR), updating
        each bar as price moves higher
      - Stop is checked against dynamic_stop; exit type is "TRAIL" not "STOP"

    When saturday_time_stop=True:
      - After Saturday stop/target checks, if Saturday HIGH < 1R level,
        exit at Saturday close with type "TIME_SAT" (trade never reached 1R)
      - If Saturday HIGH >= 1R level, continue to Sunday as normal
      - Approximation: daily bars cannot model exact 3pm ET intrabar exit
    """
    risk_per_unit = entry - stop
    level_1r      = entry + 1.0 * risk_per_unit
    level_1_5r    = entry + 1.5 * risk_per_unit
    dynamic_stop  = stop

    for _, bar in bars.iterrows():
        dow      = bar.name.dayofweek
        day_name = {5: "Saturday", 6: "Sunday", 0: "Monday"}.get(dow, str(dow))

        if trailing_stop:
            # Upgrade stop based on how far high travelled this bar
            if bar["high"] >= level_1_5r:
                trail = bar["high"] - 0.5 * atr
                if trail > dynamic_stop:
                    dynamic_stop = trail
            if bar["high"] >= level_1r and dynamic_stop < entry:
                dynamic_stop = entry

            if bar["low"] <= dynamic_stop:
                return dynamic_stop, "TRAIL", f"Trail stop hit {day_name}"
        else:
            if bar["low"] <= stop:
                return stop, "STOP", f"Stop hit {day_name}"

        if bar["high"] >= target:
            return target, "TARGET", f"Target hit {day_name}"

        # Saturday time stop: exit if trade never reached 1R on Saturday
        if saturday_time_stop and dow == 5 and bar["high"] < level_1r:
            return bar["close"], "TIME_SAT", "Time stop Saturday close"

    last = bars.iloc[-1]
    dow  = last.name.dayofweek
    day  = {5: "Saturday", 6: "Sunday", 0: "Monday"}.get(dow, str(dow))
    return last["close"], "TIME", f"Timed out {day} close"


# ── Hourly exit simulation ───────────────────────────────────────────────────

def _simulate_exit_hourly(
    hourly_df: pd.DataFrame,
    entry: float,
    stop: float,
    target: float,
    fri_date: pd.Timestamp,
) -> tuple[float, str, str]:
    """
    Higher-fidelity exit simulation using hourly bars for the weekend of fri_date.

    hourly_df must be indexed by UTC-aware Timestamps (from load_hourly_data).
    fri_date is the Friday signal date (naive or UTC-normalized).

    Time stop: if 1R (entry + risk) not reached by the Saturday 20:00 UTC bar
    (= 15:00 ET), exit at that bar's close with type TIME_SAT.

    Returns (exit_price, exit_type, note_string).
    """
    risk     = entry - stop
    level_1r = entry + risk

    fri_dt   = pd.Timestamp(fri_date)
    if fri_dt.tzinfo is None:
        fri_dt = fri_dt.tz_localize("UTC")
    sat_date = (fri_dt + pd.Timedelta(days=1)).date()
    sun_date = (fri_dt + pd.Timedelta(days=2)).date()

    # Filter to weekend bars using UTC date
    idx_dates = [ts.tz_convert("UTC").date() for ts in hourly_df.index]
    weekend   = hourly_df[
        [(d == sat_date or d == sun_date) for d in idx_dates]
    ].sort_index()

    if weekend.empty:
        return entry, "NO_DATA", "no hourly bars for weekend"

    hit_1r_sat    = False
    sat_stop_done = False

    for ts, bar in weekend.iterrows():
        ts_utc      = ts.tz_convert("UTC")
        is_saturday = ts_utc.date() == sat_date

        if bar["low"] <= stop:
            day = "Saturday" if is_saturday else "Sunday"
            return stop, "STOP", f"Stop hit {day} {ts_utc.strftime('%H:%M')} UTC"

        if bar["high"] >= target:
            day = "Saturday" if is_saturday else "Sunday"
            return target, "TARGET", f"Target hit {day} {ts_utc.strftime('%H:%M')} UTC"

        if is_saturday and bar["high"] >= level_1r:
            hit_1r_sat = True

        # Saturday time stop: after processing the 20:00 UTC bar
        if is_saturday and ts_utc.hour == 20 and not sat_stop_done:
            sat_stop_done = True
            if not hit_1r_sat:
                return bar["close"], "TIME_SAT", "Hourly time stop Sat 20:00 UTC"

    last = weekend.iloc[-1]
    last_utc = last.name.tz_convert("UTC")
    day  = "Sunday" if last_utc.date() == sun_date else "Saturday"
    return last["close"], "TIME", f"Timed out {day} close"


# ── Max drawdown ────────────────────────────────────────────────────────────

def _max_drawdown(equity_series: pd.Series) -> float:
    running_max = equity_series.cummax()
    return float(((equity_series - running_max) / running_max).min())


# ── Metrics ─────────────────────────────────────────────────────────────────

def _calc_metrics(trades: pd.DataFrame, starting_capital: float = STARTING_CAPITAL) -> dict:
    completed = trades[trades["direction"] == "LONG"].dropna(subset=["R_multiple"])
    if completed.empty:
        return {}

    wins   = completed[completed["R_multiple"] > 0]
    losses = completed[completed["R_multiple"] <= 0]

    win_rate      = len(wins) / len(completed)
    avg_R         = completed["R_multiple"].mean()
    avg_win_R     = wins["R_multiple"].mean()    if not wins.empty   else 0.0
    avg_loss_R    = losses["R_multiple"].mean()  if not losses.empty else 0.0
    gross_profit  = wins["R_multiple"].sum()
    gross_loss    = abs(losses["R_multiple"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    expectancy_R  = (win_rate * avg_win_R) + ((1 - win_rate) * avg_loss_R)

    equity = [starting_capital]
    for _, row in completed.iterrows():
        equity.append(max(equity[-1] + row["profit_loss"], 0.01))
    equity_s    = pd.Series(equity)
    max_dd      = _max_drawdown(equity_s)
    total_pl    = equity[-1] - starting_capital
    roi_pre_tax = total_pl / starting_capital

    total_gains  = completed[completed["profit_loss"] > 0]["profit_loss"].sum()
    tax_bill     = total_gains * TAX_RATE
    roi_post_tax = (total_pl - tax_bill) / starting_capital

    exit_counts = completed["exit_type"].value_counts().to_dict()

    # Trades per year
    if len(completed) > 1:
        span_years = (completed["date"].max() - completed["date"].min()).days / 365.25
        trades_yr  = round(len(completed) / max(span_years, 0.01), 1)
    else:
        trades_yr = float(len(completed))

    return {
        "n_trades":               len(completed),
        "n_no_trade":             len(trades[trades["direction"] == "NO_TRADE"]),
        "trades_per_year":        trades_yr,
        "win_rate":               round(win_rate * 100, 1),
        "avg_R":                  round(avg_R, 3),
        "avg_win_R":              round(avg_win_R, 3),
        "avg_loss_R":             round(avg_loss_R, 3),
        "profit_factor":          round(profit_factor, 3),
        "expectancy_R":           round(expectancy_R, 3),
        "max_drawdown":           round(max_dd * 100, 1),
        "roi_pre_tax":            round(roi_pre_tax * 100, 1),
        "roi_post_tax_estimated": round(roi_post_tax * 100, 1),
        "exit_TARGET":            exit_counts.get("TARGET", 0),
        "exit_STOP":              exit_counts.get("STOP", 0),
        "exit_TRAIL":             exit_counts.get("TRAIL", 0),
        "exit_TIME":              exit_counts.get("TIME", 0),
        "exit_TIME_SAT":          exit_counts.get("TIME_SAT", 0),
        "exit_TIME_MON":          exit_counts.get("TIME_MON", 0),
        "exit_STOP_BE":           exit_counts.get("STOP_BE", 0),
        "equity_curve":           equity_s.tolist(),
    }


# ── Core backtest (single ticker) ───────────────────────────────────────────

def run_backtest(
    ticker_df: pd.DataFrame,
    ticker: str,
    starting_capital: float = STARTING_CAPITAL,
    risk_pct: float = RISK_PCT_PER_TRADE,
    ma20_distance_cap: float | None = None,
    trailing_stop: bool = False,
    liquidity_trap_filter: bool = False,
    market_structure_filter: bool = False,
    market_structure_mode: str = "AND",
    saturday_time_stop: bool = False,
    expanding_r_target: float = R_TARGET,
    contracting_r_target: float = R_TARGET,
    relax_filter2: bool = False,
    volume_floor_filter: bool = False,
    volume_ceiling_filter: bool = False,
    compression_size_mult: float = 1.0,
    fear_greed_df: pd.DataFrame | None = None,
    fear_greed_min: int = 0,
    fear_greed_max: int = 100,
    fear_greed_sizing: bool = False,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Run Var1+Var2+Var4 backtest for a single ticker.

    Args:
        ma20_distance_cap: Optional filter — skip LONG when
            (close - ma20) / ma20 > cap. None = disabled (default).
        trailing_stop: Dynamic stop — breakeven at 1R, trail at 1.5R.
        liquidity_trap_filter: Skip LONG when Friday candle range
            > 1.5×ATR14 AND volume > 2× rolling 20-day avg volume.
            Missing volume data defaults to allowing the trade.
        market_structure_filter: Skip LONG when the 10 daily bars
            before Friday do not show higher highs and higher lows.
            Splits into recent 5 (days 1-5) vs prior 5 (days 6-10).
            Fewer than 10 prior bars → NO_TRADE (never default-allow).
        market_structure_mode: "AND" (both HH and HL required, default)
            or "OR" (either HH or HL is sufficient to pass).
        saturday_time_stop: If True, exit at Saturday close (TIME_SAT)
            when Saturday HIGH < 1R level. Trade that never showed 1R
            on Saturday is assumed unlikely to recover Sunday. Uses
            daily bar approximation — cannot model exact 3pm ET intrabar.
        expanding_r_target: R multiplier used when ATR is expanding
            (atr_expanding=True). Default 2.0 matches baseline.
        contracting_r_target: R multiplier used when ATR is contracting
            (atr_expanding=False). Only relevant when relax_filter2=True.
            Default 2.0 matches baseline.
        relax_filter2: If True, allow LONG signals even when ATR is
            contracting (Filter 2 fails). contracting_r_target is used
            as the target multiplier for those trades.
        volume_floor_filter: Skip LONG when Friday volume is ≤ 7-day
            rolling average volume. Confirms trend participation.
            Missing 7-day avg data → NO_TRADE.
        volume_ceiling_filter: Skip LONG when Friday volume is ≥ 2×
            7-day rolling average volume. Filters exhaustion spikes.
            Missing 7-day avg data → NO_TRADE.
        (vol_floor_ok and vol_ceil_ok flags are stored on every row
        regardless of whether the filters are enabled, for analysis.)
        compression_size_mult: Risk multiplier applied to compression
            setups only (vol_compression=True). Default 1.0 = flat sizing.
            1.5 = 1.5× normal risk on compression weeks. Does not affect
            filter logic — only position sizing for LONG trades.
            (vol_compression flag stored on all rows for analysis.)
        fear_greed_df: Optional DataFrame indexed by date with a 'value'
            column (0–100). When provided, fear_greed_value is stored on
            every trade row regardless of filter settings.
        fear_greed_min: Minimum allowed Fear & Greed value (default 0 =
            no floor). Trades where value < fear_greed_min → NO_TRADE.
            Missing F&G data for a date → allow trade.
        fear_greed_max: Maximum allowed Fear & Greed value (default 100 =
            no ceiling). Trades where value > fear_greed_max → NO_TRADE.
        fear_greed_sizing: If True, scale position size by F&G zone:
            Extreme Fear (<25) → 0.5× risk, Neutral (25-75) → 1.0×,
            Extreme Greed (>75) → 0.75× risk. Missing data → 1.0×.
            Stacks multiplicatively with compression_size_mult.

    Returns:
        trade_log       – one row per Friday signal
        overall_metrics – dict of aggregate stats
        regime_metrics  – dict[regime_name → metrics dict]
    """
    fridays        = get_fridays(ticker_df)
    all_fri_idx    = ticker_df.index[ticker_df.index.dayofweek == 4]
    trade_rows     = []
    equity         = starting_capital

    # ── Volume averages (computed once upfront, shifted to avoid lookahead) ─
    vol_avg20 = ticker_df["volume"].rolling(20).mean().shift(1)
    vol_avg7  = ticker_df["volume"].rolling(7).mean().shift(1)

    for fri_date, fri_row in fridays.iterrows():
        fri_pos = ticker_df.index.get_loc(fri_date)

        # ── Prior Friday lookups ────────────────────────────────────────────
        prior_fridays = all_fri_idx[all_fri_idx < fri_date]
        if len(prior_fridays) >= 1:
            prior_fri_row   = ticker_df.loc[prior_fridays[-1]]
            prior_fri_atr   = float(prior_fri_row["atr14"])
            prior_fri_close = float(prior_fri_row["close"])
            atr_expanding    = bool(fri_row["atr14"] > prior_fri_atr)
            momentum_confirm = bool(fri_row["close"] > prior_fri_close)
        else:
            prior_fri_atr    = None
            prior_fri_close  = None
            atr_expanding    = False  # first Friday — no prior data, skip trade
            momentum_confirm = False

        # ── MA20 distance (for Filter 4) ────────────────────────────────────
        ma20_val      = float(fri_row["ma20"])
        close_val     = float(fri_row["close"])
        ma20_distance = (close_val - ma20_val) / ma20_val  # fraction above MA20

        # ── Market structure check (prior 10 bars split into two halves) ──────
        ms_higher_highs = False
        ms_higher_lows  = False
        ms_no_data      = False
        if fri_pos >= 10:
            recent_bars = ticker_df.iloc[fri_pos - 5 : fri_pos]   # days 1-5
            prior_bars  = ticker_df.iloc[fri_pos - 10 : fri_pos - 5]  # days 6-10
            recent_high = float(recent_bars["high"].max())
            recent_low  = float(recent_bars["low"].min())
            prior_high  = float(prior_bars["high"].max())
            prior_low   = float(prior_bars["low"].min())
            ms_higher_highs = recent_high > prior_high
            ms_higher_lows  = recent_low  > prior_low
        else:
            ms_no_data = True   # insufficient history — will force NO_TRADE

        # ── Liquidity trap check (computed regardless, stored for analysis) ───
        candle_range  = float(fri_row["high"]) - float(fri_row["low"])
        atr14_val     = float(fri_row["atr14"])
        vol_today     = float(fri_row["volume"])
        vol_avg       = vol_avg20.get(fri_date, float("nan"))
        wide_candle   = candle_range > 1.5 * atr14_val
        vol_spike     = (not np.isnan(vol_avg)) and vol_avg > 0 and (vol_today > 2.0 * vol_avg)
        is_trap       = wide_candle and vol_spike

        # ── Volume confirmation flags (always computed, stored for analysis) ─
        vol7_avg      = vol_avg7.get(fri_date, float("nan"))
        vol7_data_ok  = (not np.isnan(vol7_avg)) and vol7_avg > 0
        vol7_floor_ok = vol7_data_ok and (vol_today > vol7_avg)
        vol7_ceil_ok  = vol7_data_ok and (vol_today < 2.0 * vol7_avg)

        # ── Volatility compression flag (3 consecutive ATR14 declines) ──────
        vol_compression = False
        if fri_pos >= 3:
            atr_d1 = float(ticker_df.iloc[fri_pos - 1]["atr14"])  # Thursday
            atr_d2 = float(ticker_df.iloc[fri_pos - 2]["atr14"])  # Wednesday
            atr_d3 = float(ticker_df.iloc[fri_pos - 3]["atr14"])  # Tuesday
            if not (np.isnan(atr_d1) or np.isnan(atr_d2) or np.isnan(atr_d3)):
                vol_compression = bool(atr_d1 < atr_d2 < atr_d3)

        # ── Fear & Greed lookup (always computed, stored for analysis) ───────
        fg_val_today: int | None = None
        if fear_greed_df is not None:
            fg_key = pd.Timestamp(fri_date).normalize()
            if fg_key in fear_greed_df.index:
                fg_val_today = int(fear_greed_df.loc[fg_key, "value"])

        # ── Filter gate ──────────────────────────────────────────────────────
        above_ma = bool(fri_row["above_ma20"])

        if not above_ma:
            no_trade_reason = "below MA20"
        elif not atr_expanding and not relax_filter2:
            no_trade_reason = "ATR contracting"
        elif not momentum_confirm:
            no_trade_reason = "momentum decelerating"
        elif ma20_distance_cap is not None and ma20_distance > ma20_distance_cap:
            no_trade_reason = "price overextended above MA20"
        elif liquidity_trap_filter and is_trap:
            no_trade_reason = "liquidity trap — wide candle + volume spike"
        elif volume_floor_filter and not vol7_floor_ok:
            if not vol7_data_ok:
                no_trade_reason = "volume confirmation — missing data"
            else:
                no_trade_reason = "volume confirmation — below average"
        elif volume_ceiling_filter and not vol7_ceil_ok:
            if not vol7_data_ok:
                no_trade_reason = "volume confirmation — missing data"
            else:
                no_trade_reason = "volume confirmation — exhaustion spike"
        elif market_structure_filter:
            if ms_no_data:
                no_trade_reason = "market structure — insufficient history"
            elif market_structure_mode == "AND" and not (ms_higher_highs and ms_higher_lows):
                no_trade_reason = "market structure — no higher highs/lows"
            elif market_structure_mode == "OR" and not (ms_higher_highs or ms_higher_lows):
                no_trade_reason = "market structure — no higher highs/lows"
            else:
                no_trade_reason = None
        else:
            no_trade_reason = None

        # ── Fear & Greed post-filter (runs only if no prior reason set) ──────
        if no_trade_reason is None and fg_val_today is not None:
            if fg_val_today < fear_greed_min or fg_val_today > fear_greed_max:
                no_trade_reason = f"fear/greed — {fg_val_today} out of range"

        if no_trade_reason:
            trade_rows.append({
                "date":             fri_date,
                "instrument":       ticker,
                "direction":        "NO_TRADE",
                "exit_type":        "NO_TRADE",
                "no_trade_reason":  no_trade_reason,
                "ma20_distance":    round(ma20_distance * 100, 2),
                "is_trap":          is_trap,
                "ms_higher_highs":  ms_higher_highs,
                "ms_higher_lows":   ms_higher_lows,
                "vol7_floor_ok":    vol7_floor_ok,
                "vol7_ceil_ok":     vol7_ceil_ok,
                "vol_compression":  vol_compression,
                "fear_greed_value": fg_val_today,
                "R_multiple":       None,
                "profit_loss":      0.0,
                "atr_expanding":    "yes" if atr_expanding    else "no",
                "momentum_confirm": "yes" if momentum_confirm else "no",
                "r_target_used":    None,
                "entry_price":      None,
                "stop_price":       None,
                "target_price":     None,
                "exit_price":       None,
                "fees":             0.0,
                "equity_after":     round(equity, 2),
            })
            continue

        # ── Entry levels ────────────────────────────────────────────────────
        entry         = fri_row["close"] * (1 + SLIPPAGE_PCT)
        atr           = float(fri_row["atr14"])
        stop          = entry - ATR_MULT_STOP * atr
        r_mult        = expanding_r_target if atr_expanding else contracting_r_target
        target        = entry + r_mult * (entry - stop)
        risk_per_unit = entry - stop
        fg_size_mult  = 1.0
        if fear_greed_sizing and fg_val_today is not None:
            if fg_val_today < 25:
                fg_size_mult = 0.5
            elif fg_val_today > 75:
                fg_size_mult = 0.75
        size_mult     = (compression_size_mult if vol_compression else 1.0) * fg_size_mult
        risk_dollars  = equity * risk_pct * size_mult
        units         = risk_dollars / risk_per_unit

        # ── Weekend bars (Sat + Sun) ────────────────────────────────────────
        next_bars = ticker_df.iloc[fri_pos + 1 : fri_pos + 3]
        weekend   = next_bars[next_bars.index.dayofweek.isin([5, 6])].sort_index()

        if weekend.empty:
            trade_rows.append({
                "date":             fri_date,
                "instrument":       ticker,
                "direction":        "LONG",
                "exit_type":        "NO_DATA",
                "no_trade_reason":  "no weekend bar",
                "ma20_distance":    round(ma20_distance * 100, 2),
                "is_trap":          is_trap,
                "ms_higher_highs":  ms_higher_highs,
                "ms_higher_lows":   ms_higher_lows,
                "vol7_floor_ok":    vol7_floor_ok,
                "vol7_ceil_ok":     vol7_ceil_ok,
                "vol_compression":  vol_compression,
                "fear_greed_value": fg_val_today,
                "R_multiple":       None,
                "profit_loss":      0.0,
                "atr_expanding":    "yes" if atr_expanding else "no",
                "momentum_confirm": "yes",
                "entry_price":      round(entry, 2),
                "stop_price":       round(stop, 2),
                "target_price":     round(target, 2),
                "exit_price":       None,
                "fees":             0.0,
                "equity_after":     round(equity, 2),
            })
            continue

        # ── Simulate weekend exit ───────────────────────────────────────────
        exit_price, exit_type, _ = _simulate_exit(
            weekend, entry, stop, target,
            trailing_stop=trailing_stop, atr=atr,
            saturday_time_stop=saturday_time_stop,
        )

        # ── Var4 Monday hold: if TIME and profitable, hold Monday ───────────
        if exit_type == "TIME" and exit_price > entry:
            mon_candidates = ticker_df.iloc[fri_pos + 1 : fri_pos + 5]
            monday_bars    = mon_candidates[mon_candidates.index.dayofweek == 0].sort_index()
            if not monday_bars.empty:
                be_stop = entry   # breakeven stop
                mon_bar = monday_bars.iloc[0]
                if mon_bar["low"] <= be_stop:
                    exit_price = be_stop
                    exit_type  = "STOP_BE"
                elif mon_bar["high"] >= target:
                    exit_price = target
                    exit_type  = "TARGET"
                else:
                    exit_price = mon_bar["close"]
                    exit_type  = "TIME_MON"

        # ── Fees (Kraken taker both legs) ───────────────────────────────────
        fees       = (entry + exit_price) * units * FEE_PCT
        raw_pl     = units * (exit_price - entry)
        profit_loss = raw_pl - fees
        r_multiple  = profit_loss / risk_dollars   # net R including fees

        equity = max(equity + profit_loss, 0.01)

        trade_rows.append({
            "date":             fri_date,
            "instrument":       ticker,
            "direction":        "LONG",
            "exit_type":        exit_type,
            "no_trade_reason":  None,
            "ma20_distance":    round(ma20_distance * 100, 2),
            "is_trap":          is_trap,
            "ms_higher_highs":  ms_higher_highs,
            "ms_higher_lows":   ms_higher_lows,
            "R_multiple":       round(r_multiple, 3),
            "profit_loss":      round(profit_loss, 2),
            "atr_expanding":    "yes" if atr_expanding else "no",
            "momentum_confirm": "yes",
            "r_target_used":    r_mult,
            "vol7_floor_ok":    vol7_floor_ok,
            "vol7_ceil_ok":     vol7_ceil_ok,
            "vol_compression":  vol_compression,
            "fear_greed_value": fg_val_today,
            "entry_price":      round(entry, 2),
            "stop_price":       round(stop, 2),
            "target_price":     round(target, 2),
            "exit_price":       round(exit_price, 2),
            "fees":             round(fees, 2),
            "equity_after":     round(equity, 2),
        })

    trade_log = pd.DataFrame(trade_rows)
    trade_log["date"] = pd.to_datetime(trade_log["date"])

    overall_metrics = _calc_metrics(trade_log, starting_capital)

    regime_metrics: dict[str, dict] = {}
    for name, (r_start, r_end) in REGIMES.items():
        subset = trade_log[
            (trade_log["date"] >= r_start) & (trade_log["date"] <= r_end)
        ]
        regime_metrics[name] = _calc_metrics(subset, starting_capital)

    return trade_log, overall_metrics, regime_metrics


# ── Output helpers ───────────────────────────────────────────────────────────

def _print_ticker_results(ticker: str, m: dict, regime_metrics: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {ticker}  —  Var1+Var2+Var4 Baseline")
    print(f"{'='*60}")
    if not m:
        print("  No completed trades.")
        return
    print(f"  Trades          : {m['n_trades']}  ({m['trades_per_year']}/yr)")
    print(f"  NO_TRADE        : {m['n_no_trade']}")
    print(f"  Win Rate        : {m['win_rate']}%")
    print(f"  Avg R (net/fee) : {m['avg_R']}")
    print(f"  Profit Factor   : {m['profit_factor']}")
    print(f"  Max Drawdown    : {m['max_drawdown']}%")
    print(f"  ROI Pre-Tax     : {m['roi_pre_tax']}%")
    tgt = m["exit_TARGET"]; stp = m["exit_STOP"]; tim = m["exit_TIME"]
    mon = m.get("exit_TIME_MON", 0); be = m.get("exit_STOP_BE", 0)
    n   = m["n_trades"]
    print(f"  Exits — TARGET:{tgt}({tgt/n*100:.0f}%)  STOP:{stp}({stp/n*100:.0f}%)"
          f"  TIME:{tim}({tim/n*100:.0f}%)  TIME_MON:{mon}  STOP_BE:{be}")
    print(f"\n  Regime breakdown:")
    for name, rm in regime_metrics.items():
        if rm:
            print(f"    {name:<22} T:{rm['n_trades']:>3}  WR:{rm['win_rate']:>5}%"
                  f"  AvgR:{rm['avg_R']:>6}  PF:{rm['profit_factor']:>5}"
                  f"  DD:{rm['max_drawdown']:>6}%")
        else:
            print(f"    {name:<22} No trades")


def save_trade_log(trade_log: pd.DataFrame, ticker: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = ticker.lower().replace("-", "_") + "_trade_log.csv"
    path  = out_dir / fname
    trade_log.to_csv(path, index=False)
    print(f"  Trade log → {path}")


# ── CLI entry ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Var1+Var2+Var4 Baseline Backtest — BTC & ETH")
    print(f"ATR_MULT_STOP={ATR_MULT_STOP}  RISK_PCT={RISK_PCT_PER_TRADE}"
          f"  FEE={FEE_PCT*100:.2f}% per leg")
    print("=" * 60)

    print("\nLoading data...")
    data = load_crypto_data(["BTC-USD", "ETH-USD"])

    results = {}
    for ticker in ["BTC-USD", "ETH-USD"]:
        ticker_df = get_ticker_df(data, ticker)
        print(f"\nRunning {ticker}...")
        trade_log, overall, regimes = run_backtest(ticker_df, ticker)
        results[ticker] = (trade_log, overall, regimes)
        _print_ticker_results(ticker, overall, regimes)
        save_trade_log(trade_log, ticker, OUTPUT_DIR)

    # ── Target comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TARGET COMPARISON (from master plan)")
    print("=" * 60)
    targets = {
        "BTC-USD": {"trades": 142, "wr": 33.0, "avg_R": 0.139, "pf": 1.50, "dd": -18.8},
        "ETH-USD": {"trades":  96, "wr": 36.5, "avg_R": 0.200, "pf": 1.91, "dd": -17.4},
    }
    for ticker, tgt in targets.items():
        m = results[ticker][1]
        if not m:
            continue
        print(f"\n  {ticker}")
        print(f"    Trades  : got {m['n_trades']:>4}  target ~{tgt['trades']}")
        print(f"    Win %   : got {m['win_rate']:>5}%  target ~{tgt['wr']}%")
        print(f"    Avg R   : got {m['avg_R']:>6}   target ~{tgt['avg_R']}")
        print(f"    PF      : got {m['profit_factor']:>6}   target ~{tgt['pf']}")
        print(f"    Max DD  : got {m['max_drawdown']:>5}%  target ~{tgt['dd']}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
