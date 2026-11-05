"""Home court by season (2020-21..2025-26) and whether capping blowouts helps.
Run: python analysis/hca_blowouts.py   (after python src/data.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D      # noqa: E402
import ratings as R   # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
LABEL = {2021: "2020-21 (COVID)", 2022: "2021-22", 2023: "2022-23", 2024: "2023-24",
         2025: "2024-25", 2026: "2025-26"}


def home_court(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        g = games[games.season == s]
        _, lg = R.fit(g, g.date.max() + pd.Timedelta(days=1), None, R.Dials(hca_games=1.0))
        home = D.d1_games(g)[lambda x: ~x.neutral]
        rows.append({"season": LABEL[s], "home_court_pts": round(R.hca_points(lg), 2),
                     "raw_home_margin": round(home.margin.mean(), 2),
                     "home_win_pct": round(100 * (home.margin > 0).mean(), 1)})
    return pd.DataFrame(rows)


def blowouts(games: pd.DataFrame) -> pd.DataFrame:
    d1 = D.d1_games(games[games.season.isin(SEASONS)])
    m = d1.margin.abs()
    return pd.DataFrame([{"season": LABEL[s], **{f"{t}+ %": round(100 * (x >= t).mean(), 1)
                          for t in (30, 40, 50)}, "biggest": int(x.max())}
                         for s, x in m.groupby(d1.season)])


def cap_test(games: pd.DataFrame, league: str, caps=(0, 30, 40, 50)):
    """Backtest 2021-22..2025-26 with blowout caps (per 100 possessions; 0 = none)."""
    rows, uncapped = [], []
    for s in SEASONS[1:]:
        for cap in caps:
            d = R.Dials(margin_cap=cap)
            bt = R.backtest(games, s, R.final_prior(games, s - 1, d), d, league, refit_every=2)
            rows.append({"season": LABEL[s], "cap": cap or "none",
                         "mae": R.score(bt, f"{s - 1}-11-01", f"{s}-04-30")["spread_mae"]})
            if cap == 0:
                uncapped.append(bt)
    return pd.DataFrame(rows).pivot(index="season", columns="cap", values="mae"), pd.concat(uncapped)


def big_spreads(bt: pd.DataFrame) -> pd.DataFrame:
    """When the model predicts a big spread, how big is the real margin?"""
    fav = bt.spread_home.abs()
    fav_margin = np.where(bt.spread_home >= 0, bt.margin, -bt.margin)
    t = (pd.DataFrame({"predicted": fav, "actual": fav_margin})
           .groupby(pd.cut(fav, [0, 5, 10, 20, 30, 40, 100]), observed=True)
           .agg(games=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean")))
    return t.assign(miss=t.actual - t.predicted).round(1)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for lg in D.LEAGUES:
        g = D.load_games(lg)
        print(f"\n######## {lg.upper()} ########\n\nHOME COURT\n{home_court(g).to_string(index=False)}")
        print(f"\nBLOWOUTS (D1 vs D1)\n{blowouts(g).to_string(index=False)}")
        caps, bt = cap_test(g, lg)
        print(f"\nBLOWOUT CAP TEST (avg miss, pts)\n{caps.to_string()}")
        print(f"\nBIG PREDICTED SPREADS (no cap)\n{big_spreads(bt).to_string()}")
