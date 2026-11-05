"""Is the Power-conference gap real, or something else in disguise (big spreads, home court,
bad mid-majors)? Walk-forward backtests with the current model, 2022-23..2025-26.
Run: python analysis/power_bias_proof.py"""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd
import data as D, ratings as R, talent as T
from pace_control import schedule, conf_names, POWER

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PG = {"mbb": 6, "wbb": 4}


def backtests(lg):
    g = D.load_games(lg); out = []
    tp = T.TalentPrior(lg, g, SEASONS, train=SEASONS[:-1], new_value=-4.0)
    for s in [2023, 2024, 2025, 2026]:
        base = R.final_prior(g, s - 1, R.Dials()); p = tp.prior(s, base)
        p.attrs = dict(base.attrs); p.attrs["team_hca"] = R.team_hca(g, s)
        bt = R.backtest(g, s, p, R.Dials(prior_games=PG[lg]), lg)
        sc = schedule(lg, s); cn = conf_names(sc)
        gg = g.set_index("game_id")
        bt["home_id"] = gg.loc[bt.game_id, "home_id"].values; bt["away_id"] = gg.loc[bt.game_id, "away_id"].values
        bt["hp"] = bt.home_id.map(cn).isin(POWER); bt["ap"] = bt.away_id.map(cn).isin(POWER)
        bt["hconf"] = bt.home_id.map(cn); bt["aconf"] = bt.away_id.map(cn)
        final = R.final_prior(g, s, R.Dials())            # end-of-season strength, for grouping only
        bt["h_final"] = (final.O - final.D).reindex(bt.home_id).values
        bt["a_final"] = (final.O - final.D).reindex(bt.away_id).values
        bt["season"] = s; bt["ncaa"] = bt.game_id.isin(set(sc.loc[sc.tournament_id == 22, "game_id"]))
        out.append(bt)
    b = pd.concat(out, ignore_index=True)
    b["miss"] = b.margin - b.spread_home          # + = home did better than our line
    return b


def tbl(x, by, val="res"):
    t = x.groupby(by, observed=True)[val].agg(["size", "mean", "sem"]).round(2)
    t.columns = ["games", "beat line by", "± 1 SE"]
    return t.to_string()


for lg in D.LEAGUES:
    b = backtests(lg)
    nd = b[b.date.dt.month.isin([11, 12])].copy()
    cross = nd[nd.hp != nd.ap].copy()
    # from the favorite's view, for the "is it just big favorites?" check
    for x in (nd,):
        x["fav_home"] = x.spread_home >= 0
        x["res_fav"] = np.where(x.fav_home, x.miss, -x.miss)
        x["spread_abs"] = x.spread_home.abs()
        x["bucket"] = pd.cut(x.spread_abs, [0, 5, 10, 15, 20, 30, 80], right=False)
        x["kind"] = np.where(x.hp & x.ap, "Power vs Power", np.where(~x.hp & ~x.ap, "non-P vs non-P",
                    np.where(x.fav_home == x.hp, "Power favored", "non-P favored")))
    cross["res"] = np.where(cross.hp, cross.miss, -cross.miss)      # Power team's view
    cross["venue"] = np.where(cross.neutral, "neutral", np.where(cross.hp, "Power at home", "Power on road"))
    cross["mid_good"] = np.where(cross.hp, cross.a_final, cross.h_final)
    cross["mid_tier"] = pd.cut(cross.mid_good, [-99, -10, 0, 10, 99], labels=["bad (< -10)", "below avg", "above avg", "good (> +10)"])
    print(f"\n################ {lg.upper()}: Nov-Dec games, 2022-23..2025-26 ################")
    print("\n1) Is it just big favorites beating big spreads? Favorite's result minus our line, by spread size and matchup type:")
    nd2 = nd.rename(columns={"res_fav": "res"})
    piv = nd2.pivot_table(index="bucket", columns="kind", values="res", aggfunc="mean", observed=True).round(1)
    cnt = nd2.pivot_table(index="bucket", columns="kind", values="res", aggfunc="size", observed=True)
    print(piv.to_string()); print("(games)\n" + cnt.to_string())
    print("\n2) Power vs non-P by venue (Power team's view):\n" + tbl(cross, "venue"))
    print("\n3) By how good the mid-major turned out (end-of-season rating):\n" + tbl(cross, "mid_tier"))
    print("\n4) By season:\n" + tbl(cross, "season"))
    tops = cross.assign(mid_conf=np.where(cross.hp, cross.aconf, cross.hconf))
    t = tops.groupby("mid_conf").res.agg(["size", "mean"]).query("size >= 60").sort_values("mean").round(2)
    print("\n5) Which non-Power conferences get over-rated most (Power team's result minus line):\n" + t.to_string())
