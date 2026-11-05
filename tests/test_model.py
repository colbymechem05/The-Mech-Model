"""Run: python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import daily
import data as D
import export
import odds
import ratings as R
import talent as T

HAVE_DATA = (D.ROOT / "data" / "games_mbb.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run src/data.py first")


def fake_league(n_teams=40, seed=0, hca=4.0, noise=10.0):
    """Double round robin with known true ratings."""
    rng = np.random.default_rng(seed)
    O, Dd, Tt = rng.normal(0, 8, n_teams), rng.normal(0, 8, n_teams), rng.normal(0, 3, n_teams)
    O, Dd, Tt = O - O.mean(), Dd - Dd.mean(), Tt - Tt.mean()
    rows, day = [], pd.Timestamp("2025-11-03")
    for i in range(n_teams):
        for j in range(n_teams):
            if i == j:
                continue
            poss = 68 + Tt[i] + Tt[j] + rng.normal(0, 2)
            eh = 100 + O[i] + Dd[j] + hca / 2 + rng.normal(0, noise)
            ea = 100 + O[j] + Dd[i] - hca / 2 + rng.normal(0, noise)
            rows.append(dict(game_id=len(rows), season=2026, date=day + pd.Timedelta(days=len(rows) % 120),
                             home_id=i, away_id=j, home=f"T{i}", away=f"T{j}",
                             home_pts=eh * poss / 100, away_pts=ea * poss / 100, poss=poss,
                             minutes=40, neutral=False, home_d1=True, away_d1=True))
    g = pd.DataFrame(rows)
    g["margin"] = g.home_pts - g.away_pts
    return g, O, Dd, Tt, hca


def test_possessions_formula():
    assert D.possessions(60, 10, 12, 20) == pytest.approx(60 - 10 + 12 + 0.475 * 20)


def test_fit_recovers_known_ratings():
    g, O, Dd, Tt, hca = fake_league()
    r, lg = R.fit(g, g.date.max() + pd.Timedelta(days=1), None, R.Dials(hca_games=1.0))
    r = r.sort_index()
    assert np.corrcoef(r.O, O)[0, 1] > 0.98
    assert np.corrcoef(r.D, Dd)[0, 1] > 0.98
    assert np.corrcoef(r["T"], Tt)[0, 1] > 0.98
    assert np.abs(r.net - (O - Dd)).mean() < 1.5           # points per 100
    assert lg["hca"] == pytest.approx(hca, abs=0.6)
    assert lg["avg_eff"] == pytest.approx(100, abs=0.5)


def test_fit_never_sees_future_games():
    g, *_ = fake_league()
    day = pd.Timestamp("2025-12-15")
    a, _ = R.fit(g, day, None, R.Dials())
    changed = g.copy()
    changed.loc[changed.date >= day, "home_pts"] += 50      # tamper with today + future
    b, _ = R.fit(changed, day, None, R.Dials())
    pd.testing.assert_frame_equal(a, b)


def test_backtest_predictions_ignore_later_results():
    g, *_ = fake_league()
    a = R.backtest(g, 2026, None, R.Dials(), "mbb", end="2025-12-20")
    changed = g.copy()
    changed.loc[changed.date > pd.Timestamp("2025-12-20"), "home_pts"] += 50
    b = R.backtest(changed, 2026, None, R.Dials(), "mbb", end="2025-12-20")
    pd.testing.assert_frame_equal(a, b)


def test_predict_matches_hand_math():
    r = pd.DataFrame({"O": [15.0, 5.0], "D": [-12.0, -10.0], "T": [3.0, -1.0]}, index=["A", "B"])
    lg = {"avg_eff": 100.0, "avg_tempo": 68.0, "hca": 4.0}
    p = R.predict(r, lg, ["A"], ["B"], [False], 11.7).iloc[0]
    tempo = 68 + 3 - 1
    home = (100 + 15 - 10 + 2) * tempo / 100
    away = (100 + 5 - 12 - 2) * tempo / 100
    assert p.tempo == pytest.approx(tempo)
    assert p.spread_home == pytest.approx(home - away)
    assert p.p_home == pytest.approx(norm.cdf((home - away) / 11.7))
    neutral = R.predict(r, lg, ["A"], ["B"], [True], 11.7).iloc[0]
    assert p.spread_home - neutral.spread_home == pytest.approx(4.0 * tempo / 100)


def test_team_home_court_adds_to_spread():
    r = pd.DataFrame({"O": [0.0, 0.0], "D": [0.0, 0.0], "T": [0.0, 0.0]}, index=[1, 2])
    lg = {"avg_eff": 100.0, "avg_tempo": 70.0, "hca": 4.0}
    base = R.predict(r, lg, [1], [2], [False], 11.7).spread_home.item()
    extra = R.predict(r, {**lg, "team_hca": pd.Series({1: 2.0})}, [1], [2], [False], 11.7).spread_home.item()
    neutral = R.predict(r, {**lg, "team_hca": pd.Series({1: 2.0})}, [1], [2], [True], 11.7).spread_home.item()
    assert base == pytest.approx(4.0 * 0.70) and extra == pytest.approx(6.0 * 0.70) and neutral == 0


def test_team_home_court_recovered():
    g = pd.concat([fake_league(n_teams=30, seed=s, hca=4.0, noise=4.0)[0]
                   .assign(season=s, date=lambda d, s=s: d.date + pd.DateOffset(years=s - 2026),
                           game_id=lambda d, s=s: d.game_id + 10000 * s)
                   for s in (2023, 2024, 2025)], ignore_index=True)
    boost = g.home_id == 0                       # team 0 gets +6 per 100 extra at home
    poss = g.poss
    g.loc[boost, "home_pts"] += 3 * poss[boost] / 100
    g.loc[boost, "away_pts"] -= 3 * poss[boost] / 100
    g["margin"] = g.home_pts - g.away_pts
    th = R.team_hca(g, 2026, k=0, first=2023)
    assert th[0] == pytest.approx(6.0, abs=1.5)            # found the real effect
    assert th.idxmax() == 0 and th.drop(0).std() < 1.5     # everyone else is just noise


def test_prior_counts_as_n_games():
    g, *_ = fake_league(n_teams=10)
    prior = pd.DataFrame({"O": 20.0, "D": 0.0, "T": 0.0}, index=range(10))
    early = g.date.min() + pd.Timedelta(days=1)
    r_weak, _ = R.fit(g, early, prior, R.Dials(prior_games=0.5))
    r_strong, _ = R.fit(g, early, prior, R.Dials(prior_games=50))
    # a stronger prior keeps ratings closer to the prior
    assert (r_strong.O - 20).abs().mean() < (r_weak.O - 20).abs().mean()


def test_roster_value_minutes_and_newcomers():
    tp = T.TalentPrior.__new__(T.TalentPrior)
    tp.new_value, tp.d1 = -4.0, {1, 2}
    tp.players = pd.DataFrame({"season": 2025, "athlete_id": range(7), "team_id": 1,
                               "minutes": 1000, "value": 2.0, "min_share": 1.0})
    # team 1: six returners at a full slot each (6 > 5 -> scaled to 5)
    # team 2: one transfer (from team 1) + one freshman -> 4 slots of newcomers
    roster = pd.DataFrame({"athlete_id": [0, 1, 2, 3, 4, 5, 6, 99], "team_id": [1] * 6 + [2, 2]})
    rv = tp.roster_value(2026, roster)
    assert rv.loc[1, "known_slots"] == pytest.approx(5)
    assert rv.loc[1, "roster_value"] == pytest.approx(5 * 2.0)
    assert rv.loc[2, "roster_value"] == pytest.approx(1 * 2.0 + 4 * -4.0)
    assert rv.loc[2, "transfer_slots"] == pytest.approx(1)


def test_log_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "LOGS", tmp_path)
    lines = pd.DataFrame({"game_id": [1, 2], "my_line": [-3.5, -7.0]})
    assert daily.append_log("mbb", lines) == 2
    changed = pd.DataFrame({"game_id": [2, 3], "my_line": [-1.0, -2.0]})  # game 2 again
    assert daily.append_log("mbb", changed) == 1
    log = pd.read_csv(tmp_path / "predictions_mbb.csv")
    assert list(log.game_id) == [1, 2, 3]
    assert log.loc[log.game_id == 2, "my_line"].item() == -7.0       # original kept


def test_vegas_line_parsing():
    assert odds.home_margin({"details": "DUKE -6.5"}, "DUKE", "UNC") == 6.5      # home favored
    assert odds.home_margin({"details": "DUKE -6.5"}, "UNC", "DUKE") == -6.5     # away favored
    assert odds.home_margin({"details": "EVEN"}, "A", "B") == 0.0
    flags = {"details": "", "spread": -4.0, "awayTeamOdds": {"favorite": True}}
    assert odds.home_margin(flags, "A", "B") == -4.0                            # fallback to flags
    assert odds.home_margin({"details": "OFF"}, "A", "B") is None


def test_vegas_parse_skips_started_games():
    def event(gid, state, details):
        return {"id": str(gid), "competitions": [{
            "status": {"type": {"state": state}},
            "odds": [{"details": details, "overUnder": 140.5, "provider": {"name": "Book"}}],
            "competitors": [{"homeAway": "home", "team": {"abbreviation": "AUB"}},
                            {"homeAway": "away", "team": {"abbreviation": "SELA"}}]}]}
    board = {"events": [event(1, "pre", "AUB -18.5"), event(2, "in", "AUB -18.5"), event(3, "pre", "OFF")]}
    v = odds.parse(board, pd.Timestamp("2026-11-04"))
    assert list(v.game_id) == [1] and v.vegas_home.item() == 18.5 and v.vegas_total.item() == 140.5


def test_vegas_grading(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "LOGS", tmp_path)
    monkeypatch.setattr(odds, "LOGS", tmp_path)
    # Mech: home by 10 / away by 2 / home by 1.   Vegas: home by 6 / home by 1 / home by 1.5
    pd.DataFrame({"game_id": [1, 2, 3], "date": ["2026-11-04"] * 3, "away": list("abc"), "home": list("xyz"),
                  "favorite": ["x", "b", "z"], "my_line": [-10.0, -2.0, -1.0], "fav_win_pct": [80.0, 56.0, 53.0],
                  "spread_home": [10.0, -2.0, 1.0], "p_home": [0.8, 0.44, 0.53]}
                 ).to_csv(tmp_path / "predictions_mbb.csv", index=False)
    odds.save("mbb", pd.DataFrame({"game_id": [1, 2, 3], "date": ["2026-11-04"] * 3,
                                   "vegas_home": [6.0, 1.0, 1.5], "vegas_total": [140.0] * 3,
                                   "provider": ["Book"] * 3, "fetched_at": ["x"] * 3}))
    finals = pd.DataFrame({"game_id": [1, 2, 3], "home_pts": [80, 70, 70], "away_pts": [72, 75, 69]})
    monkeypatch.setattr(D, "load_games", lambda lg: finals.assign(margin=finals.home_pts - finals.away_pts))
    x = export.grade("mbb").set_index("game_id")
    assert list(x.vegas_line) == [-6.0, 1.0, -1.5]   # from the Mech favorite's side (Vegas had b as a 1-pt dog)
    assert list(x.miss) == [2.0, 3.0, 0.0] and list(x.vegas_miss) == [2.0, 6.0, 0.5]
    # game 1: edge +4 on home, home won by 8 vs 6 -> W.  game 2: edge -3 on away, away won -> W.
    assert list(x[x.pick].pick_result) == ["W", "W"] and not x.loc[3, "pick"]
    s = export.record("mbb")["summary"]
    assert s["vegas_graded"] == 3 and s["pick_w"] == 2 and s["pick_l"] == 0
    assert s["closer_pct"] == pytest.approx(100 * (0.5 + 1 + 1) / 3, abs=0.1)   # tie, Mech, Mech


@needs_data
@pytest.mark.parametrize("league", ["mbb", "wbb"])
def test_real_games_are_clean(league):
    g = D.load_games(league)
    assert g.game_id.is_unique
    pace40 = g.poss * 40 / g.minutes          # catches broken box scores; real games
    assert pace40.between(45, 125).all()      # top out ~118 (track meets vs tiny schools)
    assert ((g.minutes - 40) % 5 == 0).all() and (g.minutes >= 40).all()
    assert (g.margin == g.home_pts - g.away_pts).all()
    assert not (g.home_id == g.away_id).any()
    n_d1 = len(set(g[g.home_d1].home_id) | set(g[g.away_d1].away_id))
    assert 355 <= n_d1 <= 375


@needs_data
def test_d1_filter():
    d1 = D.d1_teams("mbb")
    assert 2509 in d1          # Purdue
    assert 3229 not in d1      # Virginia-Lynchburg (NAIA)


@needs_data
def test_daily_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "OUT", tmp_path / "out")
    monkeypatch.setattr(daily, "LOGS", tmp_path / "logs")
    daily.run(pd.Timestamp("2026-11-04"), refresh=False)
    for lg in D.LEAGUES:
        r = pd.read_csv(tmp_path / "out" / f"ratings_{lg}.csv")
        assert len(r) > 350 and r["rank"].is_monotonic_increasing and r.net.notna().all()
        assert r.net.is_monotonic_decreasing
        lines = pd.read_csv(tmp_path / "out" / f"lines_{lg}_2026-11-04.csv")
        assert len(lines) > 0 and (lines.my_line <= 0).all()
        assert lines.fav_win_pct.between(50, 100).all()
