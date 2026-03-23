# Trading

Two systematic trading models — stock swing and crypto weekend — sharing a single Firebase dashboard.

## Structure

```
Trading/
├── index.html              # Live dashboard (GitHub Pages root)
├── stock_model/            # S&P 500 swing trade model
├── crypto_model/           # BTC/ETH weekend breakout model
├── Trading.code-workspace  # Open both models in VS Code
└── README.md
```

## Stock Model

Identifies top S&P 500 swing trade candidates each week using momentum, RSI, ATR, and liquidity filters.

**Run:**
```bash
cd Trading/stock_model
python3 main.py
```

Output: `Results/candidates/candidates_YYYY-MM-DD.csv`

**Dependencies:** `pip install pandas yfinance requests`

## Crypto Model

Generates BTC-USD and ETH-USD weekend long signals every Friday after market close. Paper trading Phase 1.

**Run:**
```bash
cd Trading/crypto_model/crypto
python3 main.py
```

Output:
- `Results/crypto_candidates/candidate_YYYY-MM-DD.csv`
- `Results/crypto_candidates/dashboard_YYYY-MM-DD.html`
- Writes signal to Firebase Firestore (requires `firebase_config.json`)

**Dependencies:** `pip install pandas requests`

**Firebase setup:** Copy `crypto/firebase_config.json.template` to `crypto/firebase_config.json` and fill in your project credentials. Never commit `firebase_config.json`.

## Dashboard

Open `index.html` in Chrome to view the live dashboard. Log in with Google (Firebase Auth). Use the **Stock** and **Crypto** tabs to switch between models.

For GitHub Pages: push to main branch and enable Pages from the repo root. The dashboard URL will be `https://<username>.github.io/<repo>/`.
