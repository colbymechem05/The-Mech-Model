"""Buy-game adjustment: Power team at home vs. a non-Power visitor, November-December.
On top of the current model (talent prior + conference + team home court + pooling).
Candidate boosts (added to the home team's spread, using only pre-game info):
  A. flat:        B = b0
  B. mismatch:    B = b0 + b1 x (our spread)            bigger mismatch -> bigger boost
  C. + road trip: B = b0 + b1 x spread + b2 x (visitor's road/neutral games in the last 7 days)
Fit on 2021-22..2024-25 and checked on 2025-26, plus leave-one-season-out over all 5.
Run: python analysis/buy_game_term.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import ratings as R     # noqa: E402
import talent as T      # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
TEST = [2022, 2023, 2024, 2025, 2026]
PG = {"mbb": 6, "wbb": 4}
POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Big East", "Pac-12"}


def road_trip(games):
    """Visitor's road/neutral games in the 7 days before each game (all opponents, D1 or not)."""
    rows = pd.concat([games[["game_id", "date", "away_id"]].rename(columns={"away_id": "team"}),
                      games.loc[games.neutral, ["game_id", "date", "home_id"]].rename(columns={"home_id": "team"})])
    rows = rows.sort_values("date")
    out = {}
    for team, t in rows.groupby("team"):
        d = t.date.values
        start = np.searchsorted(d, d - np.timedelta64(7, "D"), side="left")
        cnt = np.arange(len(d)) - start           # games strictly before, within 7 days
        out.update(dict(zip(zip(t.game_id, [team] * len(t)), cnt)))
    return out


def backtests(lg):
    g = D.load_games(lg)
    tp = T.TalentPrior(lg, g, SEASONS, train=SEASONS[:-1], new_value=-4.0, conf_feature=True)
    trips = road_trip(g)
    out = []
    for s in TEST:
        conf = D.conferences(lg, s)
        base = R.final_prior(g, s - 1, R.Dials())
        p = tp.prior(s, base)
        p.attrs = dict(base.attrs)
        p.attrs["team_hca"] = R.team_hca(g, s)
        p.attrs["conf"] = conf
        bt = R.backtest(g, s, p, R.Dials(prior_games=PG[lg], conf_games=30), lg)
        gg = g.set_index("game_id")
        bt["home_id"] = gg.loc[bt.game_id, "home_id"].values
        bt["away_id"] = gg.loc[bt.game_id, "away_id"].values
        hp, ap = bt.home_id.map(conf).isin(POWER), bt.away_id.map(conf).isin(POWER)
        bt["buy"] = hp & ~ap & ~bt.neutral & bt.date.dt.month.isin([11, 12])
        bt["trip"] = [trips.get((gid, a), 0) for gid, a in zip(bt.game_id, bt.away_id)]
        bt["season"] = s
        out.append(bt)
    b = pd.concat(out, ignore_index=True)
    b["resid"] = b.margin - b.spread_home
    return b


FORMS = {"A. flat": lambda x: np.ones((len(x), 1)),
         "B. mismatch": lambda x: np.column_stack([np.ones(len(x)), x.spread_home]),
         "C. mismatch + road trip": lambda x: np.column_stack([np.ones(len(x)), x.spread_home, x.trip.clip(upper=3)])}


def fit(x, form):
    return np.linalg.lstsq(FORMS[form](x), x.resid.values, rcond=None)[0]


def boost(x, form, coef):
    return np.where(x.buy, FORMS[form](x) @ coef, 0.0)


def summary(b, adj):
    s = b.spread_home + adj
    nd = b.date.dt.month.isin([11, 12])
    return {"all games": (s - b.margin).abs().mean(), "Nov-Dec": (s - b.margin)[nd].abs().mean(),
            "buy games": (s - b.margin)[b.buy].abs().mean(),
            "buy-game bias": (b.margin - s)[b.buy].mean()}


pd.set_option("display.width", 200)
for lg in D.LEAGUES:
    b = backtests(lg)
    buy = b[b.buy]
    print(f"\n################ {lg.upper()} ################")
    print(f"Buy games per season: {buy.groupby('season').size().to_dict()}  "
          f"(share of all Nov-Dec games: {100 * len(buy) / b.date.dt.month.isin([11, 12]).sum():.0f}%)")
    print("Leftover in buy games with today's model (home result minus our line), by season: "
          + str(buy.groupby("season").resid.mean().round(2).to_dict()))
    print("By our spread:\n" + buy.groupby(pd.cut(buy.spread_home, [-50, 5, 10, 15, 20, 25, 60])).resid
          .agg(["size", "mean"]).round(2).to_string())
    print("By visitor's road games in the last 7 days:\n" + buy.groupby(buy.trip.clip(upper=3)).resid
          .agg(["size", "mean"]).round(2).to_string())

    tune, hold = b[b.season < 2026], b[b.season == 2026]
    rows = [{"version": "today (no boost)", "coef": "", **summary(hold, 0.0)}]
    loso = {"today (no boost)": np.mean([summary(b[b.season == s], 0.0)["all games"] for s in TEST])}
    for form in FORMS:
        c = fit(tune[tune.buy], form)
        rows.append({"version": form, "coef": " ".join(f"{v:+.2f}" for v in c), **summary(hold, boost(hold, form, c))})
        cv = []
        for s in TEST:  # leave one season out
            tr, te = b[(b.season != s) & b.buy], b[b.season == s]
            cv.append(summary(te, boost(te, form, fit(tr, form)))["all games"])
        loso[form] = np.mean(cv)
    print("\n2025-26 holdout (boost fitted on 2021-22..2024-25):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("Leave-one-season-out, avg miss all games: " + ", ".join(f"{k} {v:.3f}" for k, v in loso.items()))

    cB = fit(b[b.buy], "B. mismatch")
    adj = boost(buy, "B. mismatch", cB)
    print(f"\nAll 5 seasons, mismatch form: boost = {cB[0]:+.2f} + {cB[1]:.3f} x spread "
          f"(10-pt favorite: {cB[0] + 10 * cB[1]:+.1f}, 20-pt: {cB[0] + 20 * cB[1]:+.1f}, 30-pt: {cB[0] + 30 * cB[1]:+.1f})")
    print(f"How much it moves buy-game spreads: median {np.median(adj):+.1f}, "
          f"middle 80%: {np.percentile(adj, 10):+.1f} to {np.percentile(adj, 90):+.1f}")
    fl = fit(b[b.buy], "A. flat")[0]
    print(f"All 5 seasons, flat form: boost = {fl:+.2f}")
