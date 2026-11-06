"""Run: python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import daily
import data as D
import export
import odds
import recruits as RC
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


def test_conference_pooling_moves_a_whole_conference():
    """Conference B's priors are all 5 too high. Only teams 10-11 play cross-conference games;
    with pooling, the rest of B moves down too (without it, they don't)."""
    teams = list(range(20))
    conf = pd.Series(["A"] * 10 + ["B"] * 10, index=teams)
    rows = []
    for k in range(6):                               # B teams 10, 11 lose to A as expected (even)
        for b in (10, 11):
            rows.append(dict(game_id=len(rows), season=2026, date=pd.Timestamp("2025-11-05") + pd.Timedelta(days=k),
                             home_id=k, away_id=b, home="", away="", home_pts=70.0, away_pts=70.0, poss=70.0,
                             minutes=40, neutral=True, home_d1=True, away_d1=True))
    g = pd.DataFrame(rows)
    g["margin"] = g.home_pts - g.away_pts
    prior = pd.DataFrame({"O": [0.0] * 10 + [2.5] * 10, "D": [0.0] * 10 + [-2.5] * 10, "T": 0.0}, index=teams)
    prior.attrs = {"hca": 0.0, "conf": conf}
    off, _ = R.fit(g, pd.Timestamp("2025-12-01"), prior, R.Dials(prior_games=6, conf_games=0))
    on, _ = R.fit(g, pd.Timestamp("2025-12-01"), prior, R.Dials(prior_games=6, conf_games=10))
    assert off.loc[15, "net"] == pytest.approx(5.0)            # unplayed B team: untouched
    assert on.loc[15, "net"] < 4.0                             # pooled: pulled down with its conference
    assert on.loc[5, "net"] == pytest.approx(off.loc[5, "net"], abs=1.0)


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
    tp.new_value, tp.d1, tp.new_coef = -4.0, {1, 2}, (-4.0, 0.0)
    tp.truth = pd.DataFrame({"season": [2025, 2025], "team_id": [1, 2], "net": [0.0, 0.0]})
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
    tp.new_coef = (-4.0, 0.2)       # program-based newcomers: team 2 was +10 last season
    tp.truth.loc[1, "net"] = 10.0
    assert tp.roster_value(2026, roster).loc[2, "roster_value"] == pytest.approx(1 * 2.0 + 4 * -2.0)


def test_buy_game_boost():
    conf = pd.Series({1: "SEC", 2: "SWAC", 3: "ACC"})
    nov, jan = pd.Timestamp("2026-11-10"), pd.Timestamp("2027-01-10")
    b = daily.buy_boost("mbb", nov, [1, 1, 2, 1], [2, 3, 1, 2], [False, False, False, True], [20.0] * 4, conf)
    assert list(b) == [pytest.approx(2.33), 0, 0, 0]          # only Power home vs non-Power, not neutral
    assert daily.buy_boost("wbb", nov, [1], [2], [False], [20.0], conf)[0] == pytest.approx(0.61 + 1.6)
    assert daily.buy_boost("mbb", jan, [1], [2], [False], [20.0], conf)[0] == 0   # Nov-Dec only


def test_recruit_name_matching():
    assert RC.norm_name("Kenyon Martin Jr.") == "kenyon martin"
    assert "ace bailey" in RC._variants('Arious "Ace" Bailey')
    stats = pd.DataFrame({"athlete_id": [1, 2, 3], "team_id": [10, 20, 30], "minutes": [500, 300, 100],
                          "name": ["Ace Bailey", "Trey Kaufman-Renn", "John Smith"]})
    rec = pd.DataFrame({"player": ['Arious "Ace" Bailey', "Trey Kaufman", "Nobody Here"],
                        "team_id": [99, 20, 30]})            # Bailey signed elsewhere, played for 10
    m = RC.match(rec, stats)
    assert list(m.played_team.fillna(-1)) == [10, 20, -1]


def test_log_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "LOGS", tmp_path)
    lines = pd.DataFrame({"game_id": [1, 2], "my_line": [-3.5, -7.0]})
    assert daily.append_log("mbb", lines) == 2
    changed = pd.DataFrame({"game_id": [2, 3], "my_line": [-1.0, -2.0]})  # game 2 again
    assert daily.append_log("mbb", changed) == 1
    log = pd.read_csv(tmp_path / "predictions_mbb.csv")
    assert list(log.game_id) == [1, 2, 3]
    assert log.loc[log.game_id == 2, "my_line"].item() == -7.0       # original kept
    # a new column shows up later (this broke the Nov 5 run): file stays readable
    newer = pd.DataFrame({"game_id": [4], "tip": ["19:00"], "my_line": [-1.5]})
    assert daily.append_log("mbb", newer) == 1
    log = pd.read_csv(tmp_path / "predictions_mbb.csv")
    assert list(log.game_id) == [1, 2, 3, 4] and log.tip.isna().sum() == 3
    assert log.loc[log.game_id == 2, "my_line"].item() == -7.0


def test_ratings_snapshot_one_block_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "LOGS", tmp_path)
    r = pd.DataFrame({"rank": [1, 2], "net": [20.0, 10.0], "O": [10, 5], "D": [-10, -5],
                      "tempo": [68, 70], "games": [1, 1]}, index=[11, 22])
    daily.snapshot("mbb", pd.Timestamp("2026-11-05"), r)
    daily.snapshot("mbb", pd.Timestamp("2026-11-05"), r.assign(net=[21.0, 9.0]))   # rerun same day
    daily.snapshot("mbb", pd.Timestamp("2026-11-06"), r)
    h = pd.read_csv(tmp_path / "ratings_history_mbb.csv")
    assert len(h) == 4 and list(h.date.unique()) == ["2026-11-05", "2026-11-06"]
    assert h[(h.date == "2026-11-05") & (h.team_id == 11)].net.item() == 21.0


def test_force_cannot_change_a_logged_line(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "LOGS", tmp_path)
    logged = pd.DataFrame({"game_id": [1], "tip": ["19:00"], "my_line": [-3.0]})
    daily.append_log("mbb", logged)
    remade = pd.DataFrame({"game_id": [1, 2], "tip": ["19:00", "20:00"], "my_line": [-8.0, -5.0]})
    out = daily.canonical("mbb", remade).set_index("game_id")
    assert out.loc[1, "my_line"] == -3.0 and out.loc[2, "my_line"] == -5.0


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
    # Meech: home by 10 / away by 2 / home by 1.   Vegas: home by 6 / home by 1 / home by 1.5
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
    assert list(x.vegas_line) == [-6.0, 1.0, -1.5]   # from the Meech favorite's side (Vegas had b as a 1-pt dog)
    assert list(x.miss) == [2.0, 3.0, 0.0] and list(x.vegas_miss) == [2.0, 6.0, 0.5]
    # game 1: edge +4 on home, home won by 8 vs 6 -> W.  game 2: edge -3 on away, away won -> W.
    assert list(x[x.pick].pick_result) == ["W", "W"] and not x.loc[3, "pick"]
    s = export.record("mbb")["summary"]
    assert s["vegas_graded"] == 3 and s["pick_w"] == 2 and s["pick_l"] == 0
    assert s["closer_pct"] == pytest.approx(100 * (0.5 + 1 + 1) / 3, abs=0.1)   # tie, Meech, Meech
    # a line logged after tip is not graded (audit Nov 6)
    log = pd.read_csv(tmp_path / "predictions_mbb.csv")
    log["made_at"] = ["2026-11-04T08:00-06:00", "2026-11-04T08:00-06:00", "2026-11-04T20:00-06:00"]
    log.to_csv(tmp_path / "predictions_mbb.csv", index=False)
    monkeypatch.setattr(export, "tip_times", lambda lg: pd.Series(["2026-11-04T19:00-05:00"] * 3, index=[1, 2, 3]))
    assert list(export.grade("mbb").game_id) == [1, 2]


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
    monkeypatch.setattr(daily, "NOW", lambda: pd.Timestamp("2026-11-04 10:00", tz="UTC"))  # before tips
    daily.run(pd.Timestamp("2026-11-04"), refresh=False)
    for lg in D.LEAGUES:
        r = pd.read_csv(tmp_path / "out" / f"ratings_{lg}.csv")
        assert len(r) > 350 and r["rank"].is_monotonic_increasing and r.net.notna().all()
        assert r.net.is_monotonic_decreasing
        lines = pd.read_csv(tmp_path / "out" / f"lines_{lg}_2026-11-04.csv")
        assert len(lines) > 0 and (lines.my_line <= 0).all()
        assert lines.fav_win_pct.between(50, 100).all()
    # a second run the same day keeps the lines as first made
    f = tmp_path / "out" / "lines_mbb_2026-11-04.csv"
    first = pd.read_csv(f)
    first.loc[0, "my_line"] = -99.0
    first.to_csv(f, index=False)
    daily.run(pd.Timestamp("2026-11-04"), refresh=False)
    assert pd.read_csv(f).loc[0, "my_line"] == -99.0
    # --force remakes the file, but logged games are published exactly as logged (audit Nov 6)
    daily.run(pd.Timestamp("2026-11-04"), refresh=False, force=True)
    log = pd.read_csv(tmp_path / "logs" / "predictions_mbb.csv").set_index("game_id")
    pub = pd.read_csv(f).set_index("game_id")
    assert (pub.my_line == log.my_line.reindex(pub.index)).all()
    # once games have tipped, a late run makes no new lines for them
    monkeypatch.setattr(daily, "NOW", lambda: pd.Timestamp("2026-11-06", tz="UTC"))
    for g in (tmp_path / "out").glob("lines_*"):
        g.unlink()
    (tmp_path / "logs" / "predictions_wbb.csv").unlink()
    daily.run(pd.Timestamp("2026-11-04"), refresh=False)
    assert not (tmp_path / "logs" / "predictions_wbb.csv").exists()
