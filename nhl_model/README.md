# NHL Model — backtest (old sheet model vs improved model)

Seasons tested: 2022-23, 2023-24, 2024-25 (regular season).

## Run

```bash
python3 nhl_model/fetch_data.py                      # NHL API results + MoneyPuck team/goalie game logs
python3 nhl_model/prepare_sheet.py path/to/NHL.xlsx  # sheet ratings, lines, bets -> data/games.csv
python3 nhl_model/backtest.py [--search]             # report -> data/backtest_report.json
```

`data/` is gitignored (downloads + private betting data).

## Improved model (`ratings.py`)

- One weight formula for everything: `(current-season sum + k * prior) / (GP + k)`, prior = last season regressed 40% to league mean
- Offense = ½ goals + ½ xG per game; defense = xG against (team shot suppression only)
- Goalie factor = (GA + 150) / (xGA + 150) over decayed career history (GSAx-based, regressed); rookies 1.02
- Empty-net goals stripped (goals/xG measured through goalie logs) → ratings centre on .500
- B2B from the schedule, not typed by hand
- `logit P(home) = h + c·ln(λhome/λaway) + b·B2B` → home is a constant odds multiplier, `c` replaces the OT/shrink block
- All constants fit by logistic regression and always scored out-of-sample (leave-one-season-out; 2024-25 also strictly walk-forward)

## Results (lower log loss = better)

2,131 sheet games that have an opening line:

| | log loss |
|---|---|
| Market (no-vig opening line) | **0.6610** |
| Old model alone | 0.6659 |
| Old model + Unabated 50/50 blend (what was bet) | 0.6634 |
| New model alone | 0.6642 |
| New model + market, fitted blend | 0.6613 |

Fitted constants (2021-24): home odds ×1.11, B2B 0.34 log-odds (about 8 pts of win prob), blend weight on model 0.06–0.37 depending on season.

Betting every side with an edge at the opening line (flat 1u):

| | bets | ROI | 95% CI |
|---|---|---|---|
| Old (as bet) | 1,550 | −0.3% | −5.6% to +5.0% |
| New model alone | 1,819 | +2.8% | −1.9% to +7.6% |
| New blend | 991 | +0.2% | −6.6% to +7.0% |

Neither model beats the opening line on accuracy, and bigger edges did not win more.

## Bet selection matters more than the model

The sides that were actually bet returned +6.8% (708 bets). The sheet's flagged edges that were not bet returned −6.1% (857 sides).
That includes 192 skipped sides with a ≥5% edge at the opening line, which lost about 14%. If bets were only placed when the edge
still held at the price available at bet time, then the line moving against the model was a strong signal that the model was wrong (goalie or injury news).
Line snapshots (open → bet time → close) are needed to confirm this, so the automated pipeline should capture them.
