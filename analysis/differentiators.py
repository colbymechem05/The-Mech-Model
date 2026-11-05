"""Three tests on top of the current model (talent prior):
  A. How many games should the preseason prior be worth? (wide sweep)
  B. Continuity: teams returning more minutes keep their preseason rating longer.
  C. Team-specific home court, pooled over past seasons and shrunk toward the league.
Same rules as every other test: walk forward (no peeking), tune on 2021-22..2024-25,
2025-26 held out. Run: python analysis/differentiators.py [mbb|wbb] [A|B|C ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import ratings as R     # noqa: E402
import rosters as RS    # noqa: E402
import talent as T      # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
TUNE, HOLD = [2022, 2023, 2024, 2025], 2026
CURRENT_PG = {"mbb": 8, "wbb": 3}


def mae(bt, a, b):
    return R.score(bt, a, b)["spread_mae"]


def early(games, priors, lg, dials, seasons=TUNE):
    """Avg miss Nov-Jan over tuning seasons (where the prior matters)."""
    return round(np.mean([mae(R.backtest(games, s, priors[s], dials, lg, refit_every=2, end=f"{s}-01-31"),
                              f"{s - 1}-11-01", f"{s}-01-31") for s in seasons]), 3)


def full(games, priors, lg, dials, seasons):
    return round(np.mean([mae(R.backtest(games, s, priors[s], dials, lg, refit_every=2),
                              f"{s - 1}-11-01", f"{s}-04-30") for s in seasons]), 3)


def hold(games, prior, lg, dials):
    bt = R.backtest(games, HOLD, prior, dials, lg)
    return mae(bt, "2025-11-01", "2025-12-31"), mae(bt, "2025-11-01", "2026-04-30")


def team_hca(games, season, k):
    return R.team_hca(games, season, k)


def main(lg, tests):
    t0 = time.time()
    games = D.load_games(lg)
    base = {s: R.final_prior(games, s - 1, R.Dials()) for s in TUNE + [HOLD]}
    tp = T.TalentPrior(lg, games, SEASONS, train=SEASONS[:-1], new_value=-4.0)
    pri = {s: tp.prior(s, base[s]) for s in base}
    print(f"\n######## {lg.upper()} ########  (avg miss in points; lower is better)")

    if "A" in tests:
        rows = []
        for pg in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]:
            d = R.Dials(prior_games=pg)
            nd, fl = hold(games, pri[HOLD], lg, d)
            rows.append({"worth N games": pg, "tune Nov-Jan": early(games, pri, lg, d),
                         "2025-26 Nov-Dec": nd, "2025-26 full": fl})
        print("\nA. Prior strength\n" + pd.DataFrame(rows).to_string(index=False))

    if "B" in tests:
        ret = {s: RS.returning(lg, s) for s in base}
        rows = []
        for pg in sorted({CURRENT_PG[lg], max(1, CURRENT_PG[lg] // 2), CURRENT_PG[lg] * 2}):
            for k in [0, 1, 2, 3]:
                pw = {}
                for s in base:
                    r = ret[s].reindex(pri[s].index)
                    w = (1 + k * (r - r.median())).clip(0.25, 3).fillna(1.0)
                    pw[s] = pri[s].assign(w=w)
                    pw[s].attrs = dict(pri[s].attrs)
                d = R.Dials(prior_games=pg)
                nd, fl = hold(games, pw[HOLD], lg, d)
                rows.append({"worth N games": pg, "continuity k": k, "tune Nov-Jan": early(games, pw, lg, d),
                             "2025-26 Nov-Dec": nd, "2025-26 full": fl})
        print("\nB. Continuity-weighted prior (k=0 is today's model; k=2: everyone back = "
              "~2.4x as many games, nobody back = ~0.4x)\n" + pd.DataFrame(rows).to_string(index=False))

    if "C" in tests:
        d = R.Dials(prior_games=CURRENT_PG[lg])
        seasons = [2023, 2024, 2025]
        rows = []
        for k in [None, 400, 200, 100, 50, 25]:
            ph = {}
            for s in seasons + [HOLD]:
                ph[s] = pri[s].copy()
                ph[s].attrs = dict(pri[s].attrs)
                if k is not None:
                    ph[s].attrs["team_hca"] = team_hca(games, s, k)
            nd, fl = hold(games, ph[HOLD], lg, d)
            rows.append({"shrink k": "off (today)" if k is None else k,
                         "tune full season": full(games, ph, lg, d, seasons),
                         "2025-26 Nov-Dec": nd, "2025-26 full": fl})
        print("\nC. Team-specific home court (smaller k = trust each team's own history more)\n"
              + pd.DataFrame(rows).to_string(index=False))
        th = team_hca(games, HOLD, 100) * 0.69  # per 100 -> points at ~69 possessions
        print("   biggest team effects for 2025-26 at k=100 (extra pts beyond league):")
        names = D.team_names(games)
        top = pd.concat([th.nlargest(5), th.nsmallest(5)])
        print("   " + ", ".join(f"{names.get(t, t)} {v:+.1f}" for t, v in top.items()))
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    args = sys.argv[1:]
    leagues = [a for a in args if a in D.LEAGUES] or D.LEAGUES
    tests = [a for a in args if a in "ABC" and len(a) == 1] or ["A", "B", "C"]
    pd.set_option("display.width", 200)
    for lg in leagues:
        main(lg, tests)
