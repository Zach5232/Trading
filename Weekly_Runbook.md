# Weekly Runbook — Crypto Weekend System

Phase 1 paper trading. Run every Friday night and Sunday night.

---

## Friday Night Checklist (after 4 PM ET market close)

**1. Run the signal pipeline**
```bash
cd Trading/crypto_model/crypto
python3 main.py
```

Expected output:
- Terminal summary: LONG or NO_TRADE for BTC and ETH, with filter pass/fail details
- `Results/crypto_candidates/candidate_YYYY-MM-DD.csv` saved
- `Results/crypto_candidates/dashboard_YYYY-MM-DD.html` saved
- Firebase signal written to Firestore (check dashboard Crypto tab)

**2. Review the output**
- Open the generated `dashboard_YYYY-MM-DD.html` in Chrome
- Check Fear & Greed index (context only — not a filter)
- Check funding rate classification (NEUTRAL / NEGATIVE / ELEVATED)
- Note both signals: LONG or NO_TRADE

**3. If LONG signal fires**
- Record entry plan in the dashboard (open `index.html`, Crypto tab → Log Trade)
  - Entry: Saturday open ≈ Friday close × 1.001
  - Stop: entry − 1.25 × ATR14
  - Target: entry + 2.0 × (entry − stop)
  - Units: ($500 × 5%) ÷ (entry − stop)

**4. Check kill switch**
- If sum of last 5 R-multiples < −2.0 → pause trading, investigate before next signal

---

## Sunday Night Checklist (before Monday open)

**1. Record exit**
- Open `index.html` → Crypto tab → Close Trade
- Exit type: TARGET / STOP / TIME
- For TIME exit: check if exit price > entry (Var4 Monday hold condition)

**2. If Var4 Monday hold applies**
- TIME exit AND exit price > entry → hold through Monday close
- Set breakeven stop = entry
- Record final exit Monday after close

**3. Update weekly record**
- Click "Close Week" in the Crypto tab after all trades are exited
- Verify equity updates correctly

**4. Phase 1 gates — check weekly**
After each closed week, verify in the Crypto tab:

| Gate | Threshold | Status |
|------|-----------|--------|
| Weeks logged | ≥ 8 | — |
| Win rate | > 50% | — |
| Expectancy (after 32% tax) | > 0 | — |

All three must clear before moving real capital into Phase 2.

---

## Notes

- **Risk per trade:** 5% of account equity ($25 at $500 equity)
- **Max concurrent risk:** 10% (both BTC + ETH trigger simultaneously)
- **Instruments:** BTC-USD and ETH-USD only (Coinbase Advanced API, no auth required)
- **Filters:** close > MA20 AND ATR expanding AND Friday-over-Friday momentum
- **Post-Phase-1 research backlog:** `crypto_model/Results/crypto_backtest/tier2_findings_2026-03.txt`
