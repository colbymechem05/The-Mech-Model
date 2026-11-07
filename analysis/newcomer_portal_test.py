"""Two transfer questions (men's, 2021-22..2025-26), test only, nothing adopted.
A. Non-D1 transfers (the JUCO idea): CBBD has no JUCO data. Its rosters include ~340-460
   D2/D3/NAIA teams that played D1 opponents, so D1 newcomers found on one of those rosters
   the season before = non-D1 transfers. Are they better than the model's typical newcomer?
B. Portal ratings (247 stars/rating) for D1-to-D1 transfers: do they predict next-season
   value beyond the box-score value the model already carries over?
Data: data/rosters, data/transfers (scripts/fetch_cbbd_rosters.py, fetch_cbbd_portal.py).
Run: python analysis/newcomer_portal_test.py"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as D        # noqa: E402
import players as P     # noqa: E402
import recruits as RC   # noqa: E402
import talent as T      # noqa: E402

LG, SEASONS = "mbb", [2021, 2022, 2023, 2024, 2025, 2026]


def fit(x, cols):
    A = np.column_stack([np.ones(len(x))] + [x[c] for c in cols])
    w = np.sqrt(x.minutes.values)
    return np.linalg.lstsq(A * w[:, None], x.value.values * w, rcond=None)[0]


def rmse(x, cols, coef):
    A = np.column_stack([np.ones(len(x))] + [x[c] for c in cols])
    return float(np.average((x.value - A @ coef) ** 2, weights=x.minutes) ** 0.5)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    tp = T.TalentPrior(LG, D.load_games(LG), SEASONS, train=SEASONS[:-1], new_value=-4.0,
                       conf_feature=True, recruit_mode="feature")
    p = tp.players.copy()
    p["key"] = p.name.map(RC.norm_name)

    # A. non-D1 transfers vs. other newcomers
    ros = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/rosters/*.csv")))])
    ros["key"] = ros.name.map(RC.norm_name)
    nd, rec, rows = ros[ros.conference.isna()], RC.load(LG), []
    for s in SEASONS[1:]:
        cur = p[(p.season == s) & (p.minutes > 0)]
        new = cur[~cur.athlete_id.isin(set(p[p.season < s].athlete_id))].copy()
        new["nonD1"] = new.key.isin(set(nd[nd.season.isin([s - 1, s - 2])].key))
        new["ranked"] = new.athlete_id.isin(RC.match(rec[rec.season == s], P.season_stats(LG, s)).athlete_id.dropna())
        rows.append(new)
    n = pd.concat(rows)
    n["group"] = np.select([n.ranked, n.nonD1], ["ranked freshman", "non-D1 transfer"], "other newcomer")
    print("A. First D1 season, by newcomer type (value = net pts per 100 per player slot)")
    print(n.groupby("group").apply(lambda x: pd.Series({
        "players": len(x), "avg_min_share": x.min_share.mean(),
        "value": np.average(x.value, weights=x.minutes)}), include_groups=False).round(2).to_string())
    print(f"model's typical newcomer value: {tp.v_typical:.2f}")

    # B. portal ratings for D1 transfers (portal year Y -> season Y + 1)
    po = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/transfers/*.csv")))])
    po["key"] = (po["first"].astype(str) + " " + po["last"].astype(str)).map(RC.norm_name)
    po["season"] = po.year + 1
    rows = []
    for s in SEASONS[1:]:
        cur = p[(p.season == s) & (p.minutes > 0)][["athlete_id", "team_id", "key", "value", "minutes"]]
        prev = (p[(p.season == s - 1) & (p.minutes > 0)].sort_values("minutes")
                .drop_duplicates("athlete_id", keep="last")[["athlete_id", "team_id", "value"]])
        m = cur.merge(prev, on="athlete_id", suffixes=("", "_old"))
        m = m[m.team_id != m.team_id_old].merge(
            po[po.season == s][["key", "stars", "rating"]].drop_duplicates("key"), on="key", how="left")
        rows.append(m.assign(season=s))
    t = pd.concat(rows)
    x = t.dropna(subset=["rating"])
    print(f"\nB. D1-to-D1 transfers: {len(t)}, with a portal rating: {len(x)}")
    tr, te = x[x.season < 2026], x[x.season == 2026]
    for cols in [["value_old"], ["value_old", "rating"]]:
        c = fit(tr, cols)
        print(f"  new value ~ {' + '.join(cols):22s} coef {np.round(c, 2)}  "
              f"RMSE fit 2022-25 {rmse(tr, cols, c):.3f}  holdout 2025-26 {rmse(te, cols, c):.3f}")
    print(x.groupby("stars").apply(lambda d: pd.Series({
        "n": len(d), "value_before": np.average(d.value_old, weights=d.minutes),
        "value_after": np.average(d.value, weights=d.minutes)}), include_groups=False).round(2).to_string())
