"""Quick check: does the current model under- or over-rate Power-conference teams in games vs. non-Power teams?
Run: python analysis/power_bias_check.py"""
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd
import data as D, ratings as R, talent as T
from pace_control import schedule, conf_names, POWER
SEASONS=[2021,2022,2023,2024,2025,2026]
for lg, pg in {"mbb": 6, "wbb": 4}.items():
    g = D.load_games(lg); rows = []
    tp = T.TalentPrior(lg, g, SEASONS, train=SEASONS[:-1], new_value=-4.0)
    for s in [2023, 2024, 2025, 2026]:
        base = R.final_prior(g, s-1, R.Dials()); p = tp.prior(s, base); p.attrs = dict(base.attrs); p.attrs["team_hca"] = R.team_hca(g, s)
        bt = R.backtest(g, s, p, R.Dials(prior_games=pg), lg)
        sc = schedule(lg, s); cn = conf_names(sc); ncaa = set(sc.loc[sc.tournament_id == 22, "game_id"])
        gg = g.set_index("game_id")
        bt["hp"] = gg.loc[bt.game_id, "home_id"].map(cn).isin(POWER).values
        bt["ap"] = gg.loc[bt.game_id, "away_id"].map(cn).isin(POWER).values
        x = bt[bt.hp != bt.ap].copy()
        x["res_p"] = np.where(x.hp, 1, -1) * (x.margin - x.spread_home)   # + = Power team beat the line
        x["when"] = np.where(x.game_id.isin(ncaa), "NCAA tourney", np.where(x.date.dt.month.isin([11, 12]), "Nov-Dec", "Jan-Mar other"))
        x["season"] = s; rows.append(x)
    X = pd.concat(rows)
    t = X.groupby("when").res_p.agg(["size", "mean", "sem"]).round(2)
    print(f"\n{lg.upper()} Power vs non-Power games, Power team's result minus the Mech line (current model, 2022-23..2025-26):\n{t.to_string()}")
    print(X.groupby(["season"]).res_p.mean().round(2).to_dict(), "(Nov-Dec by season:)", X[X.when=="Nov-Dec"].groupby("season").res_p.mean().round(2).to_dict())
