"""Try candidate model improvements; every number is out-of-sample (leave-one-season-out).

Bar to clear: the model has to make the market better when blended with it
(sheet games, log loss of blend < market alone), not just predict games well.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import ratings as R  # noqa: E402
from backtest import (DATA, TEST, american_to_dec, american_to_prob, logit, logloss,  # noqa: E402
                      loso_predict)


def feats(cols):
    def X(d):
        parts = [d.x.values, d.a_b2b.astype(float).values - d.h_b2b.astype(float).values]
        if "rest" in cols:
            parts.append(d.h_rest.values - d.a_rest.values)
        if "3in4" in cols:
            parts.append(d.a_3in4.astype(float).values - d.h_3in4.astype(float).values)
        return np.column_stack(parts)
    return X


def sheet_games():
    sh = pd.read_csv(os.path.join(DATA, "games.csv"))
    sh = sh[sh.gameId.notna() & sh.pa.notna() & sh.ph.notna()].copy()
    sh = sh[(sh.pa.abs() >= 100) & (sh.ph.abs() >= 100)]
    ih, ia = american_to_prob(sh.ph), american_to_prob(sh.pa)
    sh = sh[(ih + ia - 1 >= 0) & (ih + ia - 1 <= 0.08)]
    sh["mkt"] = american_to_prob(sh.ph) / (american_to_prob(sh.ph) + american_to_prob(sh.pa))
    sh["gameId"] = sh.gameId.astype(int)
    return sh[["gameId", "pa", "ph", "mkt"]]


def evaluate(r, X, sh):
    r = r.copy()
    r["hw"] = (r.hg > r.ag).astype(int)
    r["p"], _ = loso_predict(r, X)
    allg = r[r.season.isin(TEST)]
    m = sh.merge(r[["gameId", "season", "hw", "p"]], on="gameId")
    m["blend"], coef = loso_predict(
        m, lambda d: np.column_stack([logit(d.p.values), logit(d.mkt.values)]), train_pool=TEST, intercept=False)
    # 3% edge rule at the sheet line with the blended probability
    dh, da = american_to_dec(m.ph), american_to_dec(m.pa)
    evh, eva = m.blend * dh - 1, (1 - m.blend) * da - 1
    home = evh >= eva
    ev = np.where(home, evh, eva)
    won = np.where(home, m.hw == 1, m.hw == 0)
    pl = np.where(won, np.where(home, dh, da) - 1, -1)
    bet = ev > 0.03
    return dict(ll_all=logloss(allg.p.values, allg.hw.values),
                ll_market=logloss(m.mkt.values, m.hw.values),
                ll_blend=logloss(m.blend.values, m.hw.values),
                model_weight=np.mean([c[0] for c in coef.values()]),
                bets_3pct=int(bet.sum()), roi_3pct=float(pl[bet].mean()))


def main():
    df, gg = R.load_games()
    sh = sheet_games()
    base = dict(R.DEFAULTS, kg=150)
    variants = [
        ("baseline (tuned)", {}, ()),
        ("+ days of rest", {}, ("rest",)),
        ("+ 3-in-4 nights", {}, ("3in4",)),
        ("recency half-life 20 games", {"hl": 20}, ()),
        ("recency half-life 40 games", {"hl": 40}, ()),
        ("score/venue-adjusted xG", {"sva": True}, ()),
        ("xG only (no actual goals)", {"a": 0.0}, ()),
        ("goalie: no history decay", {"decay": 1.0}, ()),
    ]
    cache = {}
    rows = []
    for name, p, cols in variants:
        key = tuple(sorted(p.items()))
        if key not in cache:
            cache[key] = R.compute(df, gg, {**base, **p})
        rows.append(dict(variant=name, **evaluate(cache[key], feats(cols), sh)))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "experiments.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
