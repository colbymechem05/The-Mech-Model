"""Elo baseline: the simple benchmark the efficiency model has to beat.

Every D1 team carries one number (average 1500). The rating gap gives a win
probability; after each game both teams move by K x margin-multiplier x
(actual - expected), so upsets move ratings more than expected results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script
from data import d1_games, load_games, team_names  # noqa: E402

K = 30                 # base learning rate
EARLY_MULT = 3         # K is 3x for a team's first games, fading to 1x by game 15
EARLY_GAMES = 15
HOME_ELO = 70          # home court in Elo points
CARRYOVER = 0.6        # keep 60% of last season's distance from 1500
ELO_PER_POINT = {"mbb": 24.9, "wbb": 17.3}  # Elo gap -> points of spread (fit on Jan 2026)


def expected(diff: float) -> float:
    """Win probability for a team that is `diff` Elo points better."""
    return 1 / (1 + 10 ** (-diff / 400))


def mov_multiplier(margin: float, winner_diff: float) -> float:
    """Bigger wins move ratings more, with diminishing returns; the denominator
    stops favorites from snowballing by beating up on weak teams."""
    return (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_diff)


def k_for(n_played: int) -> float:
    return K * (1 + (EARLY_MULT - 1) * max(0.0, 1 - n_played / EARLY_GAMES))


def run(games: pd.DataFrame, league: str) -> tuple[pd.DataFrame, dict]:
    """Walk through D1 games in date order. Returns pre-game predictions and
    final ratings."""
    ratings, played, season_seen, rows = {}, {}, None, []
    for r in d1_games(games).sort_values(["date", "game_id"]).itertuples(index=False):
        if r.season != season_seen:  # new season: pull toward 1500
            ratings = {t: 1500 + CARRYOVER * (v - 1500) for t, v in ratings.items()}
            played, season_seen = {}, r.season
        rh, ra = ratings.get(r.home_id, 1500.0), ratings.get(r.away_id, 1500.0)
        diff = rh - ra + (0 if r.neutral else HOME_ELO)
        p_home = expected(diff)
        rows.append((r.game_id, r.date, r.season, r.home, r.away, diff, p_home,
                     diff / ELO_PER_POINT[league], r.margin))
        home_won = float(r.margin > 0)
        step = mov_multiplier(r.margin, diff if home_won else -diff) * (home_won - p_home)
        ratings[r.home_id] = rh + k_for(played.get(r.home_id, 0)) * step
        ratings[r.away_id] = ra - k_for(played.get(r.away_id, 0)) * step
        played[r.home_id] = played.get(r.home_id, 0) + 1
        played[r.away_id] = played.get(r.away_id, 0) + 1
    preds = pd.DataFrame(rows, columns=["game_id", "date", "season", "home", "away",
                                        "elo_diff", "p_home", "spread_home", "margin"])
    return preds, ratings


def evaluate(preds: pd.DataFrame, start: str, end: str) -> dict:
    p = preds[(preds.date >= start) & (preds.date <= end)]
    return {"games": len(p),
            "spread_mae": round((p.spread_home - p.margin).abs().mean(), 2),
            "pick_acc": round(((p.p_home > 0.5) == (p.margin > 0)).mean(), 3),
            "brier": round(((p.p_home - (p.margin > 0)) ** 2).mean(), 4)}


if __name__ == "__main__":
    for lg in ["mbb", "wbb"]:
        games = load_games(lg)
        last = games[games.season == 2026]
        preds, final = run(last, lg)
        jan = preds[(preds.date >= "2026-01-01") & (preds.date < "2026-02-01")]
        slope = np.polyfit(jan.elo_diff, jan.margin, 1)[0]
        print(f"\n{lg.upper()}: fitted 1 pt = {1 / slope:.1f} Elo | Feb-Apr 2026:",
              evaluate(preds, "2026-02-01", "2026-04-30"))
        top = pd.Series(final).sort_values(ascending=False).head(10)
        print(pd.DataFrame({"team": team_names(last).reindex(top.index).values,
                            "elo": top.round().astype(int).values}).to_string())
