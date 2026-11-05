"""Dictators vs. chameleons, 2021-22..2025-26: who's been which, and does it matter?
Control is measured on games BEFORE Jan 15 and then checked against games AFTER Jan 15
(and the NCAA tournament) using the current model's walk-forward lines, so nothing is
graded with information it was built from.
Run: python analysis/pace_history.py"""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd
import data as D, ratings as R, talent as T
from pace_control import schedule, conf_names, team_games, profile, POWER

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PG = {"mbb": 6, "wbb": 4}
TIERS = ["Chameleon", "Middle", "Dictator"]


def run(lg):
    g = D.load_games(lg)
    tp = T.TalentPrior(lg, g, SEASONS, train=SEASONS[:-1], new_value=-4.0)
    profs, rows = [], []
    for s in SEASONS[1:]:
        gs = D.d1_games(g[g.season == s]); cut = pd.Timestamp(f"{s}-01-15")
        r, lgi = R.fit(gs[gs.date < cut], cut, None, R.Dials(hca_games=1.0))
        first = gs[(gs.date < cut) & gs.home_id.isin(r.index) & gs.away_id.isin(r.index)]
        pr = profile(team_games(first, r, lgi))[["control"]]
        pr["pct"] = pr.control.rank(pct=True) * 100
        pr["tier"] = pd.cut(pr.pct, [0, 100 / 3, 200 / 3, 100], labels=TIERS, include_lowest=True)
        sc = schedule(lg, s); cn = conf_names(sc)
        pr["conf"] = cn.reindex(pr.index); pr["season"] = s
        final = R.final_prior(g, s, R.Dials()); pr["net_final"] = (final.O - final.D).reindex(pr.index)
        profs.append(pr)
        base = R.final_prior(g, s - 1, R.Dials()); p = tp.prior(s, base)
        p.attrs = dict(base.attrs); p.attrs["team_hca"] = R.team_hca(g, s)
        bt = R.backtest(g, s, p, R.Dials(prior_games=PG[lg]), lg)
        bt = bt[bt.date >= cut]
        gg = g.set_index("game_id")
        bt["home_id"] = gg.loc[bt.game_id, "home_id"].values; bt["away_id"] = gg.loc[bt.game_id, "away_id"].values
        bt["ncaa"] = bt.game_id.isin(set(sc.loc[sc.tournament_id == 22, "game_id"]))
        bt["miss"] = bt.margin - bt.spread_home
        for side, opp, sign in (("home_id", "away_id", 1), ("away_id", "home_id", -1)):
            rows.append(pd.DataFrame({"season": s, "team": bt[side].values, "opp": bt[opp].values,
                                      "res": sign * bt.miss.values, "won": sign * bt.margin.values > 0,
                                      "ncaa": bt.ncaa.values}))
    P = pd.concat(profs).reset_index().rename(columns={"index": "team"})
    G = pd.concat(rows, ignore_index=True)
    key = P.set_index(["team", "season"])
    G["tier"] = key.tier.reindex(pd.MultiIndex.from_frame(G[["team", "season"]])).values
    G["opp_tier"] = key.tier.reindex(pd.MultiIndex.from_frame(G[["opp", "season"]])).values
    G = G.dropna(subset=["tier", "opp_tier"])
    return P, G


def tbl(x, by):
    t = x.groupby(by, observed=True).res.agg(["size", "mean", "sem"]).round(2)
    t.columns = ["team-games", "beat our line by", "± 1 SE"]
    return t.to_string()


pd.set_option("display.width", 200)
for lg in D.LEAGUES:
    P, G = run(lg)
    names = D.team_names(D.load_games(lg))
    print(f"\n################ {lg.upper()} ################")
    print("\n1) Are dictators better teams? (end-of-season net rating by tier)")
    print(P.groupby("tier", observed=True).net_final.agg(["size", "mean"]).round(1).to_string())
    print("\n2) After Jan 15: does each tier beat our line? (all games)\n" + tbl(G, "tier"))
    m = G[(G.tier == "Dictator") & (G.opp_tier == "Chameleon")]
    print(f"\n3) Dictator vs. chameleon head-to-head after Jan 15, dictator's view: {len(m)} games, "
          f"beat line by {m.res.mean():+.2f} (± {m.res.sem():.2f}), won {100 * m.won.mean():.0f}%")
    t = G[G.ncaa]
    print("\n4) NCAA tournament by tier\n" + tbl(t, "tier"))
    tm = t[(t.tier == "Dictator") & (t.opp_tier == "Chameleon")]
    print(f"   dictator vs chameleon in the tournament: {len(tm)} games, beat line by {tm.res.mean():+.2f} "
          f"(± {tm.res.sem():.2f}), won {100 * tm.won.mean():.0f}%")
    G["power"] = G.team.map(P.drop_duplicates("team").set_index("team").conf).isin(POWER)
    print("\n5) After Jan 15 by conference type x tier\n" + tbl(G.assign(power=G.power.map({True: "Power", False: "non-P"})), ["power", "tier"]))
    h = P.groupby("team").agg(seasons=("pct", "size"), avg_pct=("pct", "mean"),
                              dict_seasons=("tier", lambda x: (x == "Dictator").sum()),
                              cham_seasons=("tier", lambda x: (x == "Chameleon").sum()),
                              net=("net_final", "mean"), conf=("conf", "last"))
    h = h[h.seasons == 5]
    h["team"] = names.reindex(h.index).values
    cols = ["team", "conf", "avg_pct", "dict_seasons", "cham_seasons", "net"]
    print("\n6) Most consistent DICTATORS, 5 seasons (avg control percentile; seasons as a dictator):")
    print(h.nlargest(10, "avg_pct")[cols].round(1).to_string(index=False))
    print("\n   Most consistent CHAMELEONS:")
    print(h.nsmallest(10, "avg_pct")[cols].round(1).to_string(index=False))
    good = h[h.net > 10]
    print("\n   Among good programs (avg net > +10), dictators vs chameleons:")
    print(pd.concat([good.nlargest(6, "avg_pct"), good.nsmallest(6, "avg_pct")])[cols].round(1).to_string(index=False))
    P.assign(team_name=names.reindex(P.team).values).round(2).to_csv(D.ROOT / "output" / f"pace_history_{lg}.csv", index=False)
