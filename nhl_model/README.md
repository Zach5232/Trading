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

## Bet selection (sheet ROI column BB > 3% rule)

| BB > 3% games | games | ROI |
|---|---|---|
| All qualifying games (rule followed mechanically) | 892 | +1.0% |
| Bet | 610 | +6.7% |
| Skipped | 282 | −12% |
| Sheet goalie ≠ actual starter | 123 | −18% |
| Sheet goalie correct (rule + confirmed goalie) | 769 | +4.1% |

Wrong projected goalies explain part of the gap (28% of skipped games vs 7% of bets). The remaining 204 skipped games with the
correct goalie still lost 8%, while the bets with the correct goalie won 8.5%. The pipeline should only price games after goalie confirmation
and log every flagged game, bet or skipped, with a reason, plus line snapshots (open → bet time → close).

Bets placed at a worse price than the sheet line (the market moved toward the pick) returned +17% (130 bets).

## Experiments (`experiments.py`)

All results are out-of-sample. Each variant's blended probability is compared against the opening line:

| Variant | Log loss, all games | Blend vs market (0.6601) | 3%-rule ROI |
|---|---|---|---|
| Baseline (tuned) | 0.6582 | 0.6605 | −6.0% (247) |
| + days of rest | 0.6582 | 0.6606 | −6.3% (273) |
| + 3-in-4 nights | 0.6581 | 0.6605 | −3.9% (281) |
| Recency half-life 20 games | 0.6585 | 0.6608 | +0.8% (288) |
| Recency half-life 40 games | 0.6581 | 0.6607 | −2.9% (286) |
| Score/venue-adjusted xG | 0.6584 | 0.6607 | −8.2% (151) |
| xG only | 0.6596 | 0.6601 | −3.6% (344) |
| Goalie history, no decay | 0.6583 | 0.6605 | −1.0% (232) |

None of the stat tweaks improve on the market. Public team and goalie stats are already in the line, so the remaining gains are
in information timing (confirmed goalies, injuries), line shopping, and measuring closing-line value.

## Live pipeline (`board.py`)

GitHub Actions (`.github/workflows/board.yml`) runs `board.py` at about 11am ET, hourly around midday on weekends,
and every 30 minutes from 5pm to 10:30pm ET. It commits `state/` and `site/data/`, and Netlify serves `site/`.

Each run:
1. Rebuilds ratings once a day (MoneyPuck goalie logs for the last 6 seasons plus NHL API results), using the backtested model and the constants in `model_config.json` (`fit_live.py`)
2. Gets starting goalies from DailyFaceoff; teams without a listed goalie use their most frequent starter from the last 10 games
3. Snapshots moneylines from Pinnacle plus 9 books you bet at (1 Odds API credit per run), saving every line change
4. Blends the model with Pinnacle's no-vig line (weights 0.28 model / 0.72 market, fit on 2022-25) and takes the best price across your books
5. Marks a game **BET** only when the edge is at least 3% and both goalies are confirmed. Stake is 1/8 Kelly on a $100 unit
6. Logs every flagged pick, then grades it after the game: result, closing price at the same book, and CLV against Pinnacle's no-vig close

Secrets: `ODDS_API_KEY`. The schedule uses about 13 Odds API credits a day (about 400 a month).
