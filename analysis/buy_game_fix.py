"""Fixing the Power-vs-mid-major gap (buy games). Candidates, each on top of today's model:
  1. Newcomers valued by program (Duke's freshmen != SWAC freshmen), learned from past newcomers
  2a. Preseason: the team model also uses the conference's average last-season rating
  2b. In-season: conference pooling (teams in a conference share an extra term, so a
      conference that's under-performing its priors moves together)
Scored on: average miss (tune 2022-23..2024-25, hold out 2025-26) AND the Power gap
(Power team's result minus our line in Nov-Dec Power vs non-Power games; 0 = fixed).
Run: python analysis/buy_game_fix.py [mbb|wbb]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import ratings as R     # noqa: E402
import talent as T      # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
TUNE, HOLD = [2023, 2024, 2025], 2026
PG = {"mbb": 6, "wbb": 4}
POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Big East", "Pac-12"}


def evaluate(lg, g, tp, conf_games, confs):
    out = {}
    for s in TUNE + [HOLD]:
        base = R.final_prior(g, s - 1, R.Dials())
        p = tp.prior(s, base)
        p.attrs = dict(base.attrs)
        p.attrs["team_hca"] = R.team_hca(g, s)
        p.attrs["conf"] = confs[s]
        bt = R.backtest(g, s, p, R.Dials(prior_games=PG[lg], conf_games=conf_games), lg,
                        refit_every=1 if s == HOLD else 2)
        gg = g.set_index("game_id")
        hp = gg.loc[bt.game_id, "home_id"].map(confs[s]).isin(POWER).values
        ap = gg.loc[bt.game_id, "away_id"].map(confs[s]).isin(POWER).values
        miss = bt.margin - bt.spread_home
        nd = bt.date.dt.month.isin([11, 12]).values
        cross = nd & (hp != ap)
        out[s] = {"full": (bt.spread_home - bt.margin).abs().mean(),
                  "novdec": (bt.spread_home - bt.margin)[nd].abs().mean(),
                  "gap": np.where(hp, miss, -miss)[cross].mean()}
    t = pd.DataFrame(out).T
    return {"tune full": t.loc[TUNE, "full"].mean(), "tune Nov-Dec": t.loc[TUNE, "novdec"].mean(),
            "tune Power gap": t.loc[TUNE, "gap"].mean(),
            "25-26 Nov-Dec": t.loc[HOLD, "novdec"], "25-26 full": t.loc[HOLD, "full"],
            "25-26 Power gap": t.loc[HOLD, "gap"]}


def main(lg):
    g = D.load_games(lg)
    confs = {s: D.conferences(lg, s) for s in SEASONS}
    tps = {(nb, cf): T.TalentPrior(lg, g, SEASONS, train=SEASONS[:-1], new_value=-4.0,
                                   new_by_program=nb, conf_feature=cf)
           for nb in (False, True) for cf in (False, True)}
    nc = tps[(True, False)].new_coef
    print(f"\n######## {lg.upper()} ########")
    print(f"Newcomer value learned: {nc[0]:+.2f} + {nc[1]:.2f} x team's last-season net "
          f"(a -15 team's newcomers: {nc[0] - 15 * nc[1]:+.1f}, a +20 team's: {nc[0] + 20 * nc[1]:+.1f}; today all -4)")
    print("Team model with conference: net = {:.2f} + {:.2f} roster + {:.2f} last + {:.2f} conference avg".format(
        *tps[(False, True)].coef))
    rows = []
    for name, key, cg in [("today", (False, False), 0),
                          ("1. newcomers by program", (True, False), 0),
                          ("2a. conference in preseason", (False, True), 0),
                          ("1 + 2a", (True, True), 0),
                          ("2b. conference pooling (10)", (False, False), 10),
                          ("2b. conference pooling (30)", (False, False), 30),
                          ("2b. conference pooling (100)", (False, False), 100),
                          ("2a + 2b (10)", (False, True), 10),
                          ("2a + 2b (20)", (False, True), 20),
                          ("2a + 2b (30)", (False, True), 30),
                          ("2a + 2b (50)", (False, True), 50),
                          ("1 + 2a + 2b (10)", (True, True), 10),
                          ("1 + 2a + 2b (30)", (True, True), 30),
                          ("1 + 2a + 2b (100)", (True, True), 100)]:
        rows.append({"version": name, **evaluate(lg, g, tps[key], cg, confs)})
        print(f"  done: {name}", flush=True)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    for lg in [a for a in sys.argv[1:] if a in D.LEAGUES] or D.LEAGUES:
        main(lg)
