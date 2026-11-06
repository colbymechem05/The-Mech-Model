"""Audit check (Nov 6): does the backtest's full-season roster leak information?

Backtests build each season's preseason roster from every player listed in that season's box
scores, all season long. The live model only knows players listed in games played so far.
This rebuilds the 2025-26 holdout with rosters known only through the first N days of the
season and compares accuracy. Run: python analysis/roster_asof_test.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import players as P     # noqa: E402
import ratings as R     # noqa: E402
import talent as T      # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
HOLD = 2026
SETTINGS = {"mbb": 6, "wbb": 4}


def asof_roster(league: str, season: int, days: int) -> pd.DataFrame:
    """Everyone listed in a box score in the first `days` days of the season (incl. DNPs)."""
    prefix, _ = D.LEAGUES[league]
    f = D.fetch(f"{prefix}_player_boxscores/player_box_{season}.csv", D.RAW / f"{league}_player_box_{season}.csv")
    p = pd.read_csv(f, usecols=["athlete_id", "athlete_display_name", "team_id", "game_date", "minutes"],
                    low_memory=False)
    p["game_date"] = pd.to_datetime(p.game_date)
    p["minutes"] = pd.to_numeric(p.minutes, errors="coerce").fillna(0)
    p = p[p.game_date < p.game_date.min() + pd.Timedelta(days=days)]
    return (p.groupby(["athlete_id", "team_id"], as_index=False)
             .agg(minutes=("minutes", "sum"), name=("athlete_display_name", "last")))


def evaluate(league, g, tp, roster):
    base = R.final_prior(g, HOLD - 1, R.Dials())
    p = tp.prior(HOLD, base, roster)
    p.attrs = dict(base.attrs)
    p.attrs["team_hca"] = R.team_hca(g, HOLD)
    p.attrs["conf"] = D.conferences(league, HOLD)
    bt = R.backtest(g, HOLD, p, R.Dials(prior_games=SETTINGS[league], conf_games=30), league)
    nd = bt.date.dt.month.isin([11, 12])
    err = (bt.spread_home - bt.margin).abs()
    truth = tp.truth[tp.truth.season == HOLD].set_index("team_id").net
    guess = (p.O - p.D).reindex(truth.index).dropna()
    return {"games": len(bt), "Nov-Dec miss": err[nd].mean(), "full-season miss": err.mean(),
            "winners": ((bt.p_home > .5) == (bt.margin > 0)).mean() * 100,
            "preseason miss": (guess - truth.reindex(guess.index)).abs().mean(),
            "players on rosters": len(roster)}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for lg in ["mbb", "wbb"]:
        g = D.load_games(lg)
        tp = T.TalentPrior(lg, g, SEASONS, train=SEASONS[:-1], new_value=-4.0, conf_feature=True,
                           recruit_mode="feature")
        rows = [{"roster": "full season (backtests so far)", **evaluate(lg, g, tp, P.season_stats(lg, HOLD))}]
        for days in (3, 14):
            rows.append({"roster": f"first {days} days only", **evaluate(lg, g, tp, asof_roster(lg, HOLD, days))})
            print(lg, days, "done", flush=True)
        print(f"\n{lg} 2025-26 holdout\n" + pd.DataFrame(rows).round(3).to_string(index=False) + "\n", flush=True)
