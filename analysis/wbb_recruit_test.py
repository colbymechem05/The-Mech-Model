"""Women's freshmen by recruiting rank: ESPN vs. On3/Rivals vs. 247 vs. their average.
Only one past class has rankings (class of 2025 -> 2025-26), so this is IN-SAMPLE on
2025-26 (the rank->value curve and its weight are learned from the same season): an
optimistic upper bound, not a true test. Run: python analysis/wbb_recruit_test.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import ratings as R     # noqa: E402
import talent as T      # noqa: E402

LG, S, SEASON = "wbb", [2021, 2022, 2023, 2024, 2025, 2026], 2026

if __name__ == "__main__":
    g = D.load_games(LG)
    conf = D.conferences(LG, SEASON)
    rows = []
    for name, mode, src in [("no recruits (today)", None, "avg"), ("ESPN", "feature", "espn_rank"),
                            ("On3 / Rivals", "feature", "on3_rank"), ("247", "feature", "247_rank"),
                            ("average of all", "feature", "avg")]:
        tp = T.TalentPrior(LG, g, S, train=S, new_value=-4.0, conf_feature=True, recruit_mode=mode, recruit_source=src)
        base = R.final_prior(g, SEASON - 1, R.Dials())
        p = tp.prior(SEASON, base)
        p.attrs = dict(base.attrs)
        p.attrs["team_hca"] = R.team_hca(g, SEASON)
        p.attrs["conf"] = conf
        bt = R.backtest(g, SEASON, p, R.Dials(prior_games=4, conf_games=30), LG)
        nd = bt.date.dt.month.isin([11, 12])
        truth = tp.truth[tp.truth.season == SEASON].set_index("team_id").net
        guess = (p.O - p.D).reindex(truth.index).dropna()
        rows.append({"version": name, "Nov-Dec miss": (bt.spread_home - bt.margin)[nd].abs().mean(),
                     "full-season miss": (bt.spread_home - bt.margin).abs().mean(),
                     "preseason miss": (guess - truth.reindex(guess.index)).abs().mean(),
                     "recruit weight": tp.coef[-1] if mode else None})
        print("done", name, flush=True)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
