"""Freshmen valued by recruiting rank (RSCI top 100, men's) vs. today's flat default.
Walk-forward backtests on the current model; tune 2022-23..2024-25, hold out 2025-26.
Run: python analysis/recruit_test.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import ratings as R     # noqa: E402
import talent as T      # noqa: E402

LG = "mbb"
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
TUNE, HOLD = [2023, 2024, 2025], 2026


def run(g, tp, confs):
    out, pre_err = {}, []
    for s in TUNE + [HOLD]:
        base = R.final_prior(g, s - 1, R.Dials())
        p = tp.prior(s, base)
        p.attrs = dict(base.attrs)
        p.attrs["team_hca"] = R.team_hca(g, s)
        p.attrs["conf"] = confs[s]
        bt = R.backtest(g, s, p, R.Dials(prior_games=6, conf_games=30), LG, refit_every=1 if s == HOLD else 2)
        nd = bt.date.dt.month.isin([11, 12])
        out[s] = {"full": (bt.spread_home - bt.margin).abs().mean(),
                  "novdec": (bt.spread_home - bt.margin)[nd].abs().mean()}
        truth = tp.truth[tp.truth.season == s].set_index("team_id").net
        guess = (p.O - p.D).reindex(truth.index).dropna()
        pre_err.append({"season": s, "pre": (guess - truth.reindex(guess.index)).abs().mean()})
    t, pe = pd.DataFrame(out).T, pd.DataFrame(pre_err).set_index("season").pre
    return {"tune full": t.loc[TUNE, "full"].mean(), "tune Nov-Dec": t.loc[TUNE, "novdec"].mean(),
            "25-26 Nov-Dec": t.loc[HOLD, "novdec"], "25-26 full": t.loc[HOLD, "full"],
            "preseason miss (tune)": pe.loc[TUNE].mean(), "preseason miss 25-26": pe.loc[HOLD]}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    g = D.load_games(LG)
    confs = {s: D.conferences(LG, s) for s in SEASONS}
    rows, tps = [], {}
    for mode in [None, "roster", "feature"]:
        tps[mode] = T.TalentPrior(LG, g, SEASONS, train=SEASONS[:-1], new_value=-4.0, conf_feature=True,
                                  recruit_mode=mode)
        rows.append({"version": mode or "today (flat freshmen)", **run(g, tps[mode], confs)})
        print("done", mode, flush=True)
    tp = tps["feature"]
    print("\nWhat a freshman adds by RSCI tier (learned from 2020-21..2024-25 freshmen):")
    print(tp.tier_table.round(2).to_string())
    print(f"typical unranked newcomer value: {tp.v_typical:.2f}")
    print("Team model (feature): net = " + " + ".join(f"{c:.2f}" for c in tp.coef)
          + "   [intercept, roster, last season, conference, recruits]")
    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
