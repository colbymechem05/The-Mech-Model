"""Does the roster-talent prior beat last season's ratings as the preseason
starting point? Tune on 2021-22..2024-25 (Nov-Jan), hold out 2025-26.
Run: python analysis/talent_prior.py   (after python src/data.py)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import ratings as R     # noqa: E402
import talent as T      # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
TUNE, HOLD = [2022, 2023, 2024, 2025], 2026
WINDOWS = {"Nov-Dec": ("2025-11-01", "2025-12-31"), "Jan": ("2026-01-01", "2026-01-31"),
           "Feb-Apr": ("2026-02-01", "2026-04-30"), "Full": ("2025-11-01", "2026-04-30")}


def tune_mae(games, priors, league, dials) -> float:
    """Average miss over the tuning seasons, Nov-Jan (where the prior matters)."""
    maes = [R.score(R.backtest(games, s, priors[s], dials, league, refit_every=2, end=f"{s}-01-31"),
                    f"{s - 1}-11-01", f"{s}-01-31")["spread_mae"] for s in TUNE]
    return round(sum(maes) / len(maes), 3)


def holdout(games, prior, league, dials) -> dict:
    bt = R.backtest(games, HOLD, prior, dials, league)
    return {w: R.score(bt, a, b)["spread_mae"] for w, (a, b) in WINDOWS.items()}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for lg in D.LEAGUES:
        games = D.load_games(lg)
        base = {s: R.final_prior(games, s - 1, R.Dials()) for s in TUNE + [HOLD]}
        rows = [{"setting": "last season's ratings, worth 3 games",
                 "avg": tune_mae(games, base, lg, R.Dials())}]
        best = None
        for newv in [-6.0, -4.0, -2.0]:
            tp = T.TalentPrior(lg, games, SEASONS, train=SEASONS[:-1], new_value=newv)
            pri = {s: tp.prior(s, base[s]) for s in base}
            for pg in [3, 5, 8]:
                a = tune_mae(games, pri, lg, R.Dials(prior_games=pg))
                rows.append({"setting": f"talent, newcomer {newv:+.0f}, worth {pg} games", "avg": a})
                if best is None or a < best[0]:
                    best = (a, newv, pg, tp, pri)
        print(f"\n######## {lg.upper()} | tuning seasons, avg miss Nov-Jan ########")
        print(pd.DataFrame(rows).sort_values("avg").to_string(index=False))

        _, newv, pg, tp, pri = best
        res = {"last season": holdout(games, base[HOLD], lg, R.Dials()),
               f"talent ({newv:+.0f}, {pg} games)": holdout(games, pri[HOLD], lg, R.Dials(prior_games=pg))}
        print(f"\nHOLDOUT 2025-26\n{pd.DataFrame(res).T.to_string()}")
        truth = tp.truth[tp.truth.season == HOLD].set_index("team_id").net
        for name, p in [("last season", base[HOLD]), ("talent", pri[HOLD])]:
            guess = (p.O - p.D).reindex(truth.index).dropna()
            print(f"  preseason guess vs final 2025-26 rating, {name}: avg miss "
                  f"{(guess - truth.reindex(guess.index)).abs().mean():.2f} net pts")
