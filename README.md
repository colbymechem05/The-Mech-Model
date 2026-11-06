# The Meech Model

Predictive power ratings and daily game lines for men's (`mbb`) and women's (`wbb`) D1 college basketball:
opponent-adjusted efficiency, roster-based preseason ratings, and a logged track record.

**Website:** https://colbymechem05.github.io/The-Meech-Model/

## Automatic daily run
GitHub Actions (`.github/workflows/daily.yml`) runs every morning at 7:17 AM Central (plus a 3:47 PM Vegas-line refresh): downloads last night's
results, grades yesterday's lines, rates every D1 team, makes today's lines, rebuilds the website data, runs
the checks, and commits `output/`, `logs/` and `docs/data/` back to the repo. Run it by hand from the **Actions** tab →
**Daily run** → **Run workflow**.

## Each new season (once a year, late October)
1. Add the new season to `SEASONS` in `src/data.py` (e.g. 2028 = the 2027-28 season). Everything else follows.
2. The recruiting fetch runs on its own every Oct 1. The daily schedule only runs November-April.

## Run it
```
python src/daily.py                 # today: refresh data, rate every D1 team, make today's lines
python src/daily.py 2026-11-04      # a specific date
python src/daily.py --no-refresh    # use cached data
python -m pytest tests -q           # run the checks
```
Needs Python 3.11+ and `pip install -r requirements.txt`. Data downloads automatically (free ESPN data via sportsdataverse).

## Files
| File | What it does |
|---|---|
| `src/data.py` | Downloads game data; builds one clean games table per league (possessions, OT minutes, D1 flags) |
| `src/ratings.py` | **The engine:** opponent-adjusted offense/defense/tempo + measured home court; predictions; walk-forward backtests |
| `src/players.py` | Player season stats (also used as rosters) |
| `src/talent.py` | Player ratings (Game Score, scaled to team strength) + roster-talent preseason prior |
| `src/rosters.py` | Returning-minutes share (shown in the ratings output) |
| `src/daily.py` | Daily run: ratings, today's lines, append-only prediction log |
| `src/odds.py` | Vegas lines from ESPN's scoreboard → `logs/vegas_{lg}.csv` (last pre-tip line per game) |
| `src/export.py` | Website data: one JSON per dataset in `docs/data/` (lines, rankings, conferences, track record, info) |
| `docs/index.html` | The Meech Model website (GitHub Pages serves the `docs/` folder) |
| `analysis/` | One-off studies behind each model decision (home court, talent prior, recruits, buy games, pace, Elo baseline, audit checks); each saves its results next to it |
| `tests/` | Automated checks (rating recovery on a fake league, no-peeking, prediction math, log, end-to-end) |

Outputs: `output/ratings_{lg}.csv`, `output/lines_{lg}_{date}.csv`, `logs/predictions_{lg}.csv` (never overwritten).

## How a rating works
`net` = points per 100 possessions better than an average D1 team on a neutral floor.
`O` = offense above average; `D` = points allowed above average (negative is good); `tempo` = possessions per 40 min.
Spread = efficiency gap x expected tempo / 100, +/- home court. Win % = normal curve with SD ~11.7 (men) / 12.4 (women).

## Settings (live model, Nov 6, 2026)
| Setting | Value | Chosen on |
|---|---|---|
| Preseason prior | roster talent (returning players + transfers from old box scores, newcomers at -4, ranked freshmen by recruiting tier) + last season's rating + conference; worth 6 games (men), 4 (women) | 2021-22 to 2024-25 |
| Conference pooling | shared conference offense/defense term, strength 30 | 2022-23 to 2024-25 |
| Home court | each team's own edge from the last 4 seasons, shrunk toward the league value (k = 200) | 2021-22 to 2024-25 |
| Buy-game boost (Nov-Dec, Power home vs. non-Power visitor) | men's +2.33; women's +0.61 + 0.08 x spread | **all five seasons incl. 2025-26** (refit after the 2025-26 check) |
| Recency weighting / blowout cap | off (no gain / hurt) | 2021-22 to 2024-25 |

## Accuracy
**2025-26 walk-forward check** (each day uses only earlier games; settings chosen on earlier seasons; buy-game
boost not included). This is the benchmark the site shows next to the live record:

| | Men's | Women's |
|---|---|---|
| Avg miss, full season | 9.14 pts | 9.65 pts |
| Winner picked | 72.1% | 76.6% |

Caveats (audit, Nov 6, 2026):
- The buy-game boost the live model uses was refit with 2025-26 included, so the live model is not exactly the one
  that was checked. Adding the boost fitted on 2021-22..2024-25 only improved the 2025-26 check by about
  0.005 pts (men's) / 0.006 (women's), so the gap is tiny, but it is not an untouched holdout anymore.
- Backtest rosters come from every player listed in that season's box scores; the live model only knows players
  listed so far. Rebuilding 2025-26 with rosters known after 3 or 14 days changed the full-season miss by at most
  0.006 pts (`analysis/roster_asof_test.py`).
- **The 2026-27 live track record is the only fully untouched test.** Lines are logged before tip (games that
  already tipped never get a line, and any logged after tip are excluded from grading) and are never edited.
- "Vegas" = the last pre-tip line captured from ESPN by the daily runs (7:17 AM and 3:47 PM Central), not
  necessarily the sportsbook's final closing number.
- Schedule times are fixed in UTC: 7:17 AM Central in standard time (Nov-early Mar), 8:17 AM after daylight
  time starts. GitHub can also start scheduled runs late; the 8:47 and 9:47 backups cover that.
