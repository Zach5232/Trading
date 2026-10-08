"""Fit the constants the live pipeline uses, on all backtest seasons, and write model_config.json.

  logit P(home) = home + slope * ln(lam_home / lam_away) + b2b * (away_b2b - home_b2b)
  blended      = sigmoid(w_model * logit(P) + w_market * logit(no-vig sharp line))
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import ratings as R  # noqa: E402
from backtest import DATA, american_to_prob, fit_logistic, logit, model_X, predict  # noqa: E402


def main():
    df, gg = R.load_games()
    r = R.compute(df, gg)
    r["hw"] = (r.hg > r.ag).astype(int)
    fit = r[r.season.between(2021, 2024)]
    b = fit_logistic(model_X(fit), fit.hw.values)

    sh = pd.read_csv(os.path.join(DATA, "games.csv"))
    sh = sh[sh.gameId.notna() & (sh.pa.abs() >= 100) & (sh.ph.abs() >= 100)].copy()
    ih, ia = american_to_prob(sh.ph), american_to_prob(sh.pa)
    sh = sh[(ih + ia - 1 >= 0) & (ih + ia - 1 <= 0.08)]
    sh["mkt"] = american_to_prob(sh.ph) / (american_to_prob(sh.ph) + american_to_prob(sh.pa))
    sh["gameId"] = sh.gameId.astype(int)
    m = sh[["gameId", "mkt"]].merge(r[["gameId", "season", "hw"]].assign(p=predict(b, model_X(r))), on="gameId")
    w = fit_logistic(np.column_stack([logit(m.p.values), logit(m.mkt.values)]), m.hw.values, intercept=False)

    cfg = dict(
        params=R.DEFAULTS,
        home=float(b[0]), slope=float(b[1]), b2b=float(b[2]),
        w_model=float(w[0]), w_market=float(w[1]),
        edge_threshold=0.03, kelly_fraction=0.125, unit=100,
        fit_note=f"fit on {len(fit)} games 2021-22..2024-25; blend on {len(m)} sheet games with lines",
    )
    path = os.path.join(os.path.dirname(__file__), "model_config.json")
    json.dump(cfg, open(path, "w"), indent=1)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
