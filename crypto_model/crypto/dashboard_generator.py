"""
crypto/dashboard_generator.py
==============================
Generates a fully self-contained HTML dashboard file for the Friday signal.
Called by main.py after the candidate CSV is saved.

All data is baked in at generation time — no server required, opens in any browser.
Design matches crypto_paper_tracker.html exactly.
"""

import math
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date as date_type

# ── Equity curve ──────────────────────────────────────────────────────────────
# UPDATE EACH SUNDAY — add new data point after closing week in tracker.
# Format: {"label": str, "equity": float}
EQUITY_CURVE_POINTS = [
    {"label": "Start",        "equity": 500.00},
    {"label": "Wk 1 (3/13)", "equity": 525.00},  # BTC TARGET +$50, ETH STOP -$25
    {"label": "Wk 2 (3/21)", "equity": 456.97},  # BTC NO_TRADE, ETH TIME -$68.03
]
STARTING_CAPITAL = 500.00

# Phase gate thresholds
WEEKS_REQUIRED  = 8
WIN_RATE_THRESH = 0.50
EXP_THRESH      = 0.0   # after-tax expectancy > 0R
TAX_RATE        = 0.32


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fg_label(v: int) -> str:
    if v < 25: return "Extreme Fear"
    if v < 45: return "Fear"
    if v < 55: return "Neutral"
    if v < 75: return "Greed"
    return "Extreme Greed"

def _fg_color(v: int) -> str:
    if v < 25: return "#e55555"
    if v < 45: return "#e5a020"
    if v < 55: return "#888880"
    if v < 75: return "#e5a020"
    return "#1fba85"

def _funding_color(cls: str) -> str:
    return {"NEGATIVE":"#1fba85","NEUTRAL":"#888880","ELEVATED":"#e5a020","EXTREME":"#e55555"}.get(cls,"#888880")

def _funding_note(cls: str) -> str:
    return {
        "NEGATIVE": "shorts crowded — squeeze fuel",
        "NEUTRAL":  "balanced — long OK",
        "ELEVATED": "longs slightly crowded — caution",
        "EXTREME":  "longs heavily crowded — avoid",
    }.get(cls, "—")


# ── F&G fetch ─────────────────────────────────────────────────────────────────

def _fetch_fear_greed() -> tuple[int | None, list[tuple[str, int]]]:
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=15&format=json",
            timeout=5,
        )
        if resp.status_code != 200:
            return None, []
        data = resp.json().get("data", [])
        if not data:
            return None, []
        current = int(data[0]["value"])
        history = [
            (pd.Timestamp(int(item["timestamp"]), unit="s").strftime("%m/%d"), int(item["value"]))
            for item in reversed(data[:14])
        ]
        return current, history
    except Exception:
        return None, []


# ── SVG gauge ─────────────────────────────────────────────────────────────────

def _gauge_svg(value: int | None, sparkline: list[tuple[str, int]]) -> str:
    if value is None:
        return '<div style="text-align:center;color:#555550;padding:40px 0;font-size:13px">F&amp;G unavailable</div>'

    color = _fg_color(value)
    label = _fg_label(value)
    cx, cy, r = 100, 88, 65

    def arc_d(v1: float, v2: float) -> str:
        a1 = math.radians(180.0 - v1 * 1.8)
        a2 = math.radians(180.0 - v2 * 1.8)
        x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
        x2, y2 = cx + r * math.cos(a2), cy - r * math.sin(a2)
        lg = 1 if (v2 - v1) > 50 else 0
        return f"M{x1:.2f} {y1:.2f} A{r} {r} 0 {lg} 0 {x2:.2f} {y2:.2f}"

    zones = [(0,25,"#e55555"),(25,45,"#cc7700"),(45,55,"#666660"),(55,75,"#cc7700"),(75,100,"#1fba85")]
    zone_arcs = "\n".join(
        f'  <path d="{arc_d(v1,v2)}" fill="none" stroke="{zc}" stroke-width="10" opacity="0.20" stroke-linecap="butt"/>'
        for v1, v2, zc in zones
    )
    bg_arc  = f'  <path d="{arc_d(0, 100)}" fill="none" stroke="#252523" stroke-width="10" stroke-linecap="round"/>'
    val_arc = f'  <path d="{arc_d(0, max(1.0, min(float(value), 99.5)))}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" opacity="0.9"/>'

    na = math.radians(180.0 - value * 1.8)
    nx = cx + 54 * math.cos(na)
    ny = cy - 54 * math.sin(na)

    spark = ""
    if sparkline and len(sparkline) >= 2:
        vals = [v for _, v in sparkline]
        mn, mx = min(vals), max(vals)
        if mx == mn: mx = mn + 1
        n = len(sparkline)
        pts = " ".join(
            f"{i / (n - 1) * 180:.1f},{24 - (v - mn) / (mx - mn) * 20:.1f}"
            for i, (_, v) in enumerate(sparkline)
        )
        spark = (
            f'<svg width="200" height="30" style="display:block;margin:4px auto 0">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.65"/></svg>'
            f'<div style="font-size:10px;color:#555550;text-align:center;margin-top:2px">14-day F&amp;G history</div>'
        )

    return (
        f'<svg width="200" height="100" style="display:block;margin:0 auto">\n'
        f'{bg_arc}\n{zone_arcs}\n{val_arc}\n'
        f'  <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="#e8e6e0" stroke-width="2.5" stroke-linecap="round" opacity="0.8"/>\n'
        f'  <circle cx="{cx}" cy="{cy}" r="4.5" fill="#e8e6e0" opacity="0.6"/>\n'
        f'  <text x="{cx}" y="{cy - 14}" text-anchor="middle" font-size="26" font-weight="500"'
        f' fill="{color}" font-family="-apple-system,BlinkMacSystemFont,sans-serif">{value}</text>\n'
        f'  <text x="{cx}" y="{cy + 7}" text-anchor="middle" font-size="11" fill="#888880"'
        f' font-family="-apple-system,BlinkMacSystemFont,sans-serif">{label}</text>\n'
        f'</svg>\n{spark}'
    )


# ── Signal card HTML ──────────────────────────────────────────────────────────

def _signal_card(row: dict, fg_value: int | None) -> str:
    ticker   = row["instrument"]
    label    = ticker.replace("-USD", "")
    is_long  = row["direction"] == "LONG"
    is_kill  = row.get("_kill_switch_active", False)

    badge_cls   = "badge-btc" if label == "BTC" else "badge-eth"
    card_border = "border-color:rgba(31,186,133,0.3);background:#091a11" if is_long and not is_kill else ""

    f1_pass = row["above_ma20"] == "yes"
    f2_pass = row["_atr_expanding"]
    f3_pass = row["_momentum_confirm"]

    def filter_row(name: str, passed: bool, detail: str = "") -> str:
        icon  = "✓" if passed else "✗"
        color = "#1fba85" if passed else "#e55555"
        label_str = "PASS" if passed else "FAIL"
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:0.5px solid rgba(255,255,255,0.05);font-size:12px">'
            f'<span style="color:#555550">{name}</span>'
            f'<span style="color:{color};font-weight:500">{icon} {label_str}'
            f'{(" — " + detail) if detail else ""}</span></div>'
        )

    close  = row["_fri_close"]
    ma20   = row["_ma20"]
    atr    = row["_atr14"]
    patr   = row["_prior_fri_atr"]
    pclose = row["_prior_fri_close"]

    f1_detail = f"${close:,.0f} {'>' if f1_pass else '<'} MA20 ${ma20:,.0f}"
    f2_detail = (f"ATR ${atr:,.0f} {'>' if f2_pass else '≤'} prev ${patr:,.0f}") if patr else "no prior Friday"
    f3_detail = (f"${close:,.0f} {'>' if f3_pass else '≤'} prev Fri ${pclose:,.0f}") if pclose else "no prior Friday"

    filters_html = (
        filter_row("Filter 1 — MA20",     f1_pass, f1_detail) +
        filter_row("Filter 2 — ATR exp",  f2_pass, f2_detail) +
        filter_row("Filter 3 — Momentum", f3_pass, f3_detail)
    )

    if is_kill:
        signal_html = '<div style="font-size:18px;font-weight:500;color:#555550;margin:12px 0 4px">NO_TRADE</div><div style="font-size:11px;color:#e5a020">kill switch active</div>'
    elif is_long:
        e = row["entry_price"]; s = row["stop_price"]; t = row["target_price"]
        rd = row["risk_dollars"]; u = row["units"]
        signal_html = (
            f'<div style="font-size:22px;font-weight:600;color:#1fba85;margin:12px 0 8px">LONG ✓</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;font-size:12px">'
            f'<span style="color:#555550">Entry</span><span style="font-weight:500">${e:,.2f}</span>'
            f'<span style="color:#555550">Stop</span><span style="color:#e55555;font-weight:500">${s:,.2f}</span>'
            f'<span style="color:#555550">Target</span><span style="color:#1fba85;font-weight:500">${t:,.2f}</span>'
            f'<span style="color:#555550">Risk $</span><span style="font-weight:500">${rd:,.2f}</span>'
            f'<span style="color:#555550">Units</span><span style="font-weight:500">{u:.6f} {label}</span>'
            f'</div>'
        )
    else:
        reason = row.get("_no_trade_reason", "filters not met")
        signal_html = (
            f'<div style="font-size:18px;font-weight:500;color:#555550;margin:12px 0 4px">NO_TRADE</div>'
            f'<div style="font-size:11px;color:#555550">{reason}</div>'
        )

    # Funding
    fr_rate  = row.get("_funding_rate")
    fr_class = row.get("_funding_class", "N/A")
    fr_color = _funding_color(fr_class)
    fr_text  = f"{fr_rate:+.4f}%" if fr_rate is not None else "N/A"
    fr_badge = (
        f'<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:8px;'
        f'font-weight:500;background:rgba(0,0,0,0.3);color:{fr_color};border:0.5px solid {fr_color}22">'
        f'{fr_class}</span>'
    )

    # F&G badge
    if fg_value is not None:
        fg_col   = _fg_color(fg_value)
        fg_lbl   = _fg_label(fg_value)
        fg_badge = (
            f'<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:8px;'
            f'font-weight:500;background:rgba(0,0,0,0.3);color:{fg_col};border:0.5px solid {fg_col}22">'
            f'F&amp;G {fg_value} — {fg_lbl}</span>'
        )
    else:
        fg_badge = '<span style="font-size:11px;color:#555550">F&amp;G unavailable</span>'

    return (
        f'<div style="background:#1a1a1a;border:0.5px solid rgba(255,255,255,0.08);border-radius:12px;'
        f'padding:1rem 1.25rem;{card_border}">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">'
        f'<span style="display:inline-block;font-size:12px;padding:2px 10px;border-radius:8px;font-weight:600;'
        f'background:{"#2e1800" if label == "BTC" else "#1a1030"};'
        f'color:{"#F7931A" if label == "BTC" else "#9B8CDB"}">{label}</span>'
        f'<span style="font-size:11px;color:#555550">{ticker}</span>'
        f'</div>'
        f'{filters_html}'
        f'{signal_html}'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;align-items:center">'
        f'<span style="font-size:11px;color:#555550">Funding:</span>{fr_badge}'
        f'<span style="font-size:11px;color:#555550;margin-left:4px">Sentiment:</span>{fg_badge}'
        f'</div>'
        f'</div>'
    )


# ── Price chart data builder ──────────────────────────────────────────────────

def _price_chart_data(ticker: str, ticker_df: pd.DataFrame, row: dict) -> dict:
    """Build a dict of chart data for one instrument. Returns serializable dict."""
    df = ticker_df.tail(60).copy()
    labels  = [str(d.date()) for d in df.index]
    closes  = [round(float(v), 2) for v in df["close"]]
    ma20s   = [round(float(v), 2) if not pd.isna(v) else None for v in df["ma20"]]
    atr_hi  = [round(float(c) + float(a), 2) if not pd.isna(a) else None
               for c, a in zip(df["close"], df["atr14"])]
    atr_lo  = [round(float(c) - float(a), 2) if not pd.isna(a) else None
               for c, a in zip(df["close"], df["atr14"])]

    result = {
        "labels": labels,
        "closes": closes,
        "ma20s": ma20s,
        "atr_hi": atr_hi,
        "atr_lo": atr_lo,
        "entry": None, "stop": None, "target": None,
    }
    if row["direction"] == "LONG" and not row.get("_kill_switch_active"):
        result["entry"]  = row["entry_price"]
        result["stop"]   = row["stop_price"]
        result["target"] = row["target_price"]
    return result


# ── Regime note ───────────────────────────────────────────────────────────────

def _regime_note(rows: list[dict]) -> str:
    longs = [r for r in rows if r["direction"] == "LONG" and not r.get("_kill_switch_active")]
    if len(longs) == 2:
        return "Full exposure — both instruments signaling"
    if len(longs) == 1:
        t = longs[0]["instrument"].replace("-USD","")
        return f"One signal active ({t}) — partial exposure"
    return "Both instruments NO_TRADE — sit out weekend"

def _regime_color(rows: list[dict]) -> str:
    longs = [r for r in rows if r["direction"] == "LONG" and not r.get("_kill_switch_active")]
    if len(longs) == 2: return "#1fba85"
    if len(longs) == 1: return "#e5a020"
    return "#555550"


# ── Main HTML builder ─────────────────────────────────────────────────────────

def generate_dashboard(
    rows: list[dict],
    live_data: dict,
    account_equity: float,
    weeks_logged: int = 1,
    closed_trades: list[dict] | None = None,
    candidates_dir: Path | None = None,
) -> Path | None:
    """
    Build and save the dashboard HTML file.
    Returns the Path where the file was saved, or None on error.
    """
    print("  Generating dashboard...")

    # F&G
    fg_value, fg_sparkline = _fetch_fear_greed()
    fg_lbl_str = _fg_label(fg_value) if fg_value is not None else "N/A"

    # Days until Sunday
    today = date_type.today()
    days_to_sunday = (6 - today.weekday()) % 7
    if days_to_sunday == 0:
        days_to_sunday = 7

    # Chart data for each instrument
    chart_data = {}
    for row in rows:
        ticker = row["instrument"]
        if ticker in live_data:
            chart_data[ticker] = _price_chart_data(ticker, live_data[ticker], row)

    # Phase gate computation
    # Weeks gate
    weeks_pct   = min(100, int(weeks_logged / WEEKS_REQUIRED * 100))
    weeks_pass  = weeks_logged >= WEEKS_REQUIRED

    # Win rate (from closed_trades if provided, else use row history)
    # For now derive from EQUITY_CURVE_POINTS commentary: W1 had 1W/1L
    # We'll accept these as params and default to known values
    live_wins  = 1   # Week 1 BTC TARGET
    live_total = 2   # Week 1: 2 LONG trades (BTC + ETH) — Week 2 BTC NO_TRADE, ETH open
    live_wr    = live_wins / live_total if live_total > 0 else 0.0
    wr_pass    = live_wr >= WIN_RATE_THRESH

    # After-tax expectancy (from backtest): -0.874R placeholder until live data accumulates
    at_exp     = -0.874  # TODO: compute from live closed trades
    exp_pass   = at_exp > EXP_THRESH

    # Gate indicators for header
    gates_green = sum([weeks_pass, wr_pass, exp_pass])

    # Regime
    reg_note  = _regime_note(rows)
    reg_color = _regime_color(rows)

    # Friday date
    fri_date = rows[0]["_fri_date"].date() if rows else today

    # ── HTML ──────────────────────────────────────────────────────────────────

    # Serialize chart data as JSON for JS
    chart_json = json.dumps(chart_data, allow_nan=False, default=str)
    equity_json = json.dumps(EQUITY_CURVE_POINTS)

    signal_cards_html = "\n".join(_signal_card(r, fg_value) for r in rows)

    # Funding cards
    funding_rows_html = ""
    for row in rows:
        ticker   = row["instrument"]
        label    = ticker.replace("-USD","")
        fr_rate  = row.get("_funding_rate")
        fr_class = row.get("_funding_class", "N/A")
        fr_color = _funding_color(fr_class)
        fr_note  = _funding_note(fr_class)
        fr_text  = f"{fr_rate:+.4f}%" if fr_rate is not None else "N/A"
        badge_bg = "badge-btc" if label == "BTC" else "badge-eth"
        badge_color = "#F7931A" if label == "BTC" else "#9B8CDB"
        badge_bg_c  = "#2e1800" if label == "BTC" else "#1a1030"

        funding_rows_html += (
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:8px 0;border-bottom:0.5px solid rgba(255,255,255,0.05);font-size:13px">'
            f'<span style="background:{badge_bg_c};color:{badge_color};font-size:11px;'
            f'padding:2px 8px;border-radius:6px;font-weight:600">{label}</span>'
            f'<span style="font-weight:500;color:{fr_color}">{fr_text}</span>'
            f'<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:8px;'
            f'font-weight:500;background:rgba(0,0,0,0.3);color:{fr_color};border:0.5px solid {fr_color}33">'
            f'{fr_class}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#555550;padding:2px 0 8px;'
            f'border-bottom:0.5px solid rgba(255,255,255,0.05)">{fr_note}</div>'
        )

    # Open trades section
    open_longs = [r for r in rows if r["direction"] == "LONG" and not r.get("_kill_switch_active")]
    if open_longs:
        open_rows_html = ""
        for r in open_longs:
            ticker = r["instrument"]
            label  = ticker.replace("-USD","")
            live_close = float(live_data[ticker].iloc[-1]["close"]) if ticker in live_data else None
            if live_close:
                unrealized = (live_close - r["entry_price"]) * r["units"]
                pnl_color  = "#1fba85" if unrealized >= 0 else "#e55555"
                pnl_str    = f'${unrealized:+,.2f}'
            else:
                pnl_str   = "—"
                pnl_color = "#555550"
            badge_color = "#F7931A" if label == "BTC" else "#9B8CDB"
            badge_bg_c  = "#2e1800" if label == "BTC" else "#1a1030"
            open_rows_html += (
                f'<div style="display:grid;grid-template-columns:60px 1fr 1fr 1fr 1fr 80px;'
                f'gap:8px;align-items:center;padding:8px 0;border-bottom:0.5px solid rgba(255,255,255,0.05);font-size:12px">'
                f'<span style="background:{badge_bg_c};color:{badge_color};font-size:11px;'
                f'padding:2px 8px;border-radius:6px;font-weight:600;text-align:center">{label}</span>'
                f'<div><div style="color:#555550;font-size:10px">Entry</div><div style="font-weight:500">${r["entry_price"]:,.2f}</div></div>'
                f'<div><div style="color:#555550;font-size:10px">Stop</div><div style="color:#e55555">${r["stop_price"]:,.2f}</div></div>'
                f'<div><div style="color:#555550;font-size:10px">Target</div><div style="color:#1fba85">${r["target_price"]:,.2f}</div></div>'
                f'<div><div style="color:#555550;font-size:10px">Risk</div><div>${r["risk_dollars"]:,.2f}</div></div>'
                f'<div><div style="color:#555550;font-size:10px">Est P&L</div><div style="color:{pnl_color};font-weight:500">{pnl_str}</div></div>'
                f'</div>'
            )
        open_trades_html = (
            f'<div class="section-label">Open trades this weekend</div>'
            f'{open_rows_html}'
        )
    else:
        open_trades_html = (
            f'<div style="font-size:13px;color:#555550;padding:12px 0">'
            f'No open trades — both NO_TRADE this weekend</div>'
        )

    # Gate dot helper
    def gate_dot(passed: bool) -> str:
        bg, fg, sym = ("#0d2e22","#1fba85","✓") if passed else ("#2e0d0d","#e55555","✗")
        return f'<span style="width:20px;height:20px;border-radius:50%;background:{bg};color:{fg};display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;flex-shrink:0">{sym}</span>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Dashboard — {fri_date}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f0f;color:#e8e6e0;padding:1.25rem;max-width:1100px;margin:0 auto}}
.card{{background:#1a1a1a;border:0.5px solid rgba(255,255,255,0.08);border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:1rem}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:1rem}}
.metric{{background:#222220;border:0.5px solid rgba(255,255,255,0.06);border-radius:8px;padding:.85rem 1rem}}
.metric .lbl{{font-size:12px;color:#555550;margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}}
.metric .val{{font-size:22px;font-weight:500}}
.metric .sub{{font-size:11px;color:#555550;margin-top:3px}}
.section-label{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#555550;margin-bottom:10px}}
.gate{{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:0.5px solid rgba(255,255,255,0.06)}}
.gate:last-child{{border-bottom:none}}
.equity-bar-wrap{{height:5px;background:#2a2a28;border-radius:3px;margin-top:8px}}
.equity-bar{{height:5px;border-radius:3px}}
.phase-badge{{display:inline-block;font-size:11px;padding:2px 10px;border-radius:20px;font-weight:500;background:#2e1e00;color:#e5a020;border:0.5px solid #e5a020}}
@media(max-width:640px){{.grid-2,.grid-3{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<!-- ── Section 1: Header ──────────────────────────────────────────────── -->
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.1rem;flex-wrap:wrap;gap:10px">
  <div>
    <div style="font-size:18px;font-weight:500">Crypto weekend signal</div>
    <div style="font-size:12px;color:#555550;margin-top:3px">Friday {fri_date} &nbsp;&middot;&nbsp; <span class="phase-badge">Phase {1}</span></div>
  </div>
  <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">
    <div style="text-align:right">
      <div style="font-size:12px;color:#555550">Account equity</div>
      <div style="font-size:20px;font-weight:500">{"${:,.2f}".format(account_equity)}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:12px;color:#555550">Weeks logged</div>
      <div style="font-size:20px;font-weight:500">{weeks_logged}<span style="font-size:13px;color:#555550">/{WEEKS_REQUIRED}</span></div>
    </div>
    <div style="display:flex;flex-direction:column;gap:4px">
      <div style="font-size:10px;color:#555550;text-transform:uppercase;letter-spacing:.04em">Phase gates</div>
      <div style="display:flex;gap:4px">
        {gate_dot(weeks_pass)}
        {gate_dot(wr_pass)}
        {gate_dot(exp_pass)}
      </div>
    </div>
  </div>
</div>

<!-- ── Section 2: Signal cards ───────────────────────────────────────── -->
<div class="section-label" style="margin-bottom:8px">Signals</div>
<div class="grid-2">
{signal_cards_html}
</div>

<!-- ── Section 3: Price charts ───────────────────────────────────────── -->
<div class="section-label" style="margin-bottom:8px">60-day price charts</div>
<div class="grid-2" style="margin-bottom:1rem">
  <div class="card" style="padding:.75rem">
    <div style="font-size:12px;color:#555550;margin-bottom:6px">BTC-USD &nbsp;·&nbsp;
      <span style="color:#1fba85">— MA20</span> &nbsp;·&nbsp;
      <span style="color:rgba(229,160,32,0.6)">— ATR band</span>
      {"&nbsp;·&nbsp; <span style='color:#e55555'>- Stop</span> <span style='color:#1fba85'>- Target</span> <span style='color:rgba(232,230,224,0.7)'>- Entry</span>" if chart_data.get("BTC-USD",{}).get("entry") else ""}
    </div>
    <div style="position:relative;height:220px"><canvas id="chart-btc"></canvas></div>
  </div>
  <div class="card" style="padding:.75rem">
    <div style="font-size:12px;color:#555550;margin-bottom:6px">ETH-USD &nbsp;·&nbsp;
      <span style="color:#1fba85">— MA20</span> &nbsp;·&nbsp;
      <span style="color:rgba(229,160,32,0.6)">— ATR band</span>
      {"&nbsp;·&nbsp; <span style='color:#e55555'>- Stop</span> <span style='color:#1fba85'>- Target</span> <span style='color:rgba(232,230,224,0.7)'>- Entry</span>" if chart_data.get("ETH-USD",{}).get("entry") else ""}
    </div>
    <div style="position:relative;height:220px"><canvas id="chart-eth"></canvas></div>
  </div>
</div>

<!-- ── Section 4: Sentiment row ──────────────────────────────────────── -->
<div class="section-label" style="margin-bottom:8px">Sentiment &amp; context</div>
<div class="grid-3" style="margin-bottom:1rem">

  <!-- Card 1: F&G gauge -->
  <div class="card">
    <div class="section-label">Fear &amp; Greed Index</div>
    {_gauge_svg(fg_value, fg_sparkline)}
  </div>

  <!-- Card 2: Funding rates -->
  <div class="card">
    <div class="section-label">Funding rates (OKX)</div>
    {funding_rows_html}
    <div style="font-size:10px;color:#444440;margin-top:8px">Display only — not a hard filter in Phase 1</div>
  </div>

  <!-- Card 3: Weekly context -->
  <div class="card">
    <div class="section-label">Weekly context</div>
    <div class="metric" style="margin-bottom:8px">
      <div class="lbl">Week</div>
      <div class="val">{weeks_logged}</div>
      <div class="sub">of {WEEKS_REQUIRED} required</div>
    </div>
    <div class="metric" style="margin-bottom:8px">
      <div class="lbl">Days to Sunday close</div>
      <div class="val">{days_to_sunday}</div>
    </div>
    <div style="font-size:12px;padding:8px 10px;border-radius:8px;background:#222220;color:{reg_color}">
      {reg_note}
    </div>
  </div>
</div>

<!-- ── Section 5: Equity curve ───────────────────────────────────────── -->
<div class="card" style="margin-bottom:1rem">
  <div class="section-label">Equity curve — paper trading</div>
  <div style="font-size:11px;color:#555550;margin-bottom:8px">
    <!-- UPDATE EACH SUNDAY — add new data point after closing week in tracker. -->
    Starting capital ${STARTING_CAPITAL:,.2f} &middot; Current ${account_equity:,.2f}
    &middot; <span style="color:{'#1fba85' if account_equity >= STARTING_CAPITAL else '#e55555'}">${account_equity - STARTING_CAPITAL:+,.2f}</span>
  </div>
  <div style="position:relative;height:140px"><canvas id="chart-equity"></canvas></div>
</div>

<!-- ── Section 6: Phase gates ────────────────────────────────────────── -->
<div class="card" style="margin-bottom:1rem">
  <div class="section-label">Phase 1 gates — all must clear before live capital</div>

  <div class="gate">
    {gate_dot(weeks_pass)}
    <div style="flex:1">
      <div style="font-size:13px;font-weight:500">Weeks logged &nbsp;<span style="color:{'#1fba85' if weeks_pass else '#e5a020'}">{weeks_logged}/{WEEKS_REQUIRED}</span></div>
      <div style="font-size:11px;color:#555550">Minimum 8 paper weeks before Phase 2 evaluation</div>
      <div style="height:4px;background:#2a2a28;border-radius:3px;margin-top:6px">
        <div style="height:4px;border-radius:3px;width:{weeks_pct}%;background:{'#1fba85' if weeks_pass else '#e5a020'}"></div>
      </div>
    </div>
  </div>

  <div class="gate">
    {gate_dot(wr_pass)}
    <div style="flex:1">
      <div style="font-size:13px;font-weight:500">Win rate &nbsp;<span style="color:{'#1fba85' if wr_pass else '#e55555'}">{live_wr*100:.0f}%</span> &nbsp;<span style="color:#555550;font-size:11px">need &gt;50%</span></div>
      <div style="font-size:11px;color:#555550">{live_wins}W / {live_total - live_wins}L from {live_total} live LONG trade(s)</div>
    </div>
  </div>

  <div class="gate">
    {gate_dot(exp_pass)}
    <div style="flex:1">
      <div style="font-size:13px;font-weight:500">After-tax expectancy &nbsp;<span style="color:{'#1fba85' if exp_pass else '#e55555'}">{at_exp:+.3f}R</span> &nbsp;<span style="color:#555550;font-size:11px">need &gt;0R</span></div>
      <div style="font-size:11px;color:#555550">Backtest baseline; live expectancy updates as trades close</div>
    </div>
  </div>
</div>

<!-- ── Section 7: Open trades ─────────────────────────────────────────── -->
<div class="card">
  {open_trades_html}
</div>

<!-- ── Chart.js ───────────────────────────────────────────────────────── -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
const CHART_DATA = {chart_json};
const EQUITY_DATA = {equity_json};
const STARTING = {STARTING_CAPITAL};

// ── Shared chart defaults ──────────────────────────────────────────────
Chart.defaults.color = '#555550';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 11;

function buildPriceChart(canvasId, ticker) {{
  const d = CHART_DATA[ticker];
  if (!d) return;

  const labels = d.labels;
  const n = labels.length;

  // Tick every 10 labels
  const tickLabels = labels.map((l, i) => (i === 0 || i === n-1 || i % 10 === 0) ? l : '');

  const datasets = [
    // Close price — white line
    {{
      label: 'Close',
      data: d.closes,
      borderColor: 'rgba(232,230,224,0.9)',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.15,
      fill: false,
      order: 2,
    }},
    // MA20 — green dashed
    {{
      label: 'MA20',
      data: d.ma20s,
      borderColor: '#1fba85',
      borderWidth: 1.5,
      borderDash: [5, 3],
      pointRadius: 0,
      tension: 0.15,
      fill: false,
      order: 3,
    }},
    // ATR High — faint amber (fills to ATR Low)
    {{
      label: 'ATR High',
      data: d.atr_hi,
      borderColor: 'rgba(229,160,32,0.25)',
      borderWidth: 0.5,
      pointRadius: 0,
      tension: 0.15,
      fill: '+1',
      backgroundColor: 'rgba(229,160,32,0.05)',
      order: 4,
    }},
    // ATR Low — faint amber
    {{
      label: 'ATR Low',
      data: d.atr_lo,
      borderColor: 'rgba(229,160,32,0.25)',
      borderWidth: 0.5,
      pointRadius: 0,
      tension: 0.15,
      fill: false,
      order: 4,
    }},
  ];

  // Horizontal signal lines if LONG
  if (d.entry !== null) {{
    const flat = (v) => labels.map(() => v);
    datasets.push({{
      label: 'Entry',
      data: flat(d.entry),
      borderColor: 'rgba(232,230,224,0.6)',
      borderWidth: 1,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
      order: 1,
    }});
    datasets.push({{
      label: 'Stop',
      data: flat(d.stop),
      borderColor: 'rgba(229,85,85,0.7)',
      borderWidth: 1,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
      order: 1,
    }});
    datasets.push({{
      label: 'Target',
      data: flat(d.target),
      borderColor: 'rgba(31,186,133,0.7)',
      borderWidth: 1,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
      order: 1,
    }});
  }}

  new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#1a1a1a',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          callbacks: {{
            label: ctx => {{
              if (ctx.parsed.y === null) return null;
              const prefix = ctx.dataset.label;
              return `${{prefix}}: ${{ctx.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}`;
            }}
          }}
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: 'rgba(255,255,255,0.04)' }},
          ticks: {{
            maxRotation: 0,
            callback: (_, i) => tickLabels[i],
          }},
        }},
        y: {{
          position: 'right',
          grid: {{ color: 'rgba(255,255,255,0.04)' }},
          ticks: {{
            callback: v => '$' + v.toLocaleString('en-US', {{maximumFractionDigits: 0}}),
          }},
        }},
      }},
    }},
  }});
}}

buildPriceChart('chart-btc', 'BTC-USD');
buildPriceChart('chart-eth', 'ETH-USD');

// ── Equity curve ──────────────────────────────────────────────────────
(function() {{
  const labels  = EQUITY_DATA.map(p => p.label);
  const equities = EQUITY_DATA.map(p => p.equity);
  const baseline = EQUITY_DATA.map(() => STARTING);

  new Chart(document.getElementById('chart-equity'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{
          label: 'Equity',
          data: equities,
          borderColor: equities[equities.length - 1] >= STARTING ? '#1fba85' : '#e55555',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: equities.map(v => v >= STARTING ? '#1fba85' : '#e55555'),
          fill: false,
          tension: 0.1,
          segment: {{
            borderColor: ctx => ctx.p1.parsed.y >= STARTING ? '#1fba85' : '#e55555',
          }},
        }},
        {{
          label: 'Baseline $500',
          data: baseline,
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        }},
      ],
    }},
    options: {{
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#1a1a1a',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          callbacks: {{
            label: ctx => `${{ctx.dataset.label}}: $${{ctx.parsed.y.toFixed(2)}}`,
          }}
        }},
      }},
      scales: {{
        x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{
          position: 'right',
          grid: {{ color: 'rgba(255,255,255,0.04)' }},
          ticks: {{ callback: v => '$' + v.toFixed(2) }},
        }},
      }},
    }},
  }});
}})();
</script>
</body>
</html>"""

    # Save
    if candidates_dir is None:
        candidates_dir = Path(__file__).parent.parent / "Results" / "crypto_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    out_path = candidates_dir / f"dashboard_{fri_date}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard saved  → {out_path}")
    return out_path
