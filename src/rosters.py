"""Returning minutes: share of a team's minutes from last season played by
players who are on the roster again this season (1.0 = everyone back).
Teams that haven't played yet this season get NaN (unknown), not 0.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script
from players import season_stats  # noqa: E402


def returning(league: str, season: int, now: pd.DataFrame | None = None) -> pd.Series:
    """Returning-minutes share by team_id for `season`."""
    last = season_stats(league, season - 1)
    now = season_stats(league, season) if now is None else now
    last = last[last.minutes > 0]
    back = pd.MultiIndex.from_frame(last[["athlete_id", "team_id"]]).isin(
        pd.MultiIndex.from_frame(now[["athlete_id", "team_id"]]))
    share = (last.minutes * back).groupby(last.team_id).sum() / last.groupby("team_id").minutes.sum()
    share[~share.index.isin(now.team_id)] = float("nan")
    return share.rename("ret")


if __name__ == "__main__":
    from data import LEAGUES, SEASONS, d1_teams
    for lg in LEAGUES:
        d1 = d1_teams(lg)
        med = {s: round(returning(lg, s).reindex(list(d1)).median(), 2) for s in SEASONS[1:]}
        print(f"{lg} median returning minutes (D1):", med)
