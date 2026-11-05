# The Mech Model

Predictive power ratings and daily game lines for men's (`mbb`) and women's (`wbb`) D1 college basketball:
opponent-adjusted efficiency, roster-based preseason ratings, and a logged track record.

## Automatic daily run
GitHub Actions (`.github/workflows/daily.yml`) runs every morning at 7:17 AM Central: downloads last night's
results, grades yesterday's lines, rates every D1 team, makes today's lines, rebuilds the dashboard data, runs
the checks, and commits `output/` and `logs/` back to the repo. Run it by hand from the **Actions** tab →
**Daily run** → **Run workflow**.

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
| `src/elo.py` | Elo baseline: the simple benchmark the engine must beat |
| `src/daily.py` | Daily run: ratings, today's lines, append-only prediction log |
| `src/export.py` | Dashboard data: one JSON per dataset in `output/dash/` (lines, rankings, conferences, track record, info) |
| `dashboard/index.html` | The Mech Model dashboard page |
| `analysis/` | Studies: home court & blowouts, talent prior test |
| `tests/` | Automated checks (rating recovery on a fake league, no-peeking, prediction math, log, end-to-end) |

Outputs: `output/ratings_{lg}.csv`, `output/lines_{lg}_{date}.csv`, `logs/predictions_{lg}.csv` (never overwritten).

## How a rating works
`net` = points per 100 possessions better than an average D1 team on a neutral floor.
`O` = offense above average; `D` = points allowed above average (negative is good); `tempo` = possessions per 40 min.
Spread = efficiency gap x expected tempo / 100, +/- home court. Win % = normal curve with SD ~11.7 (men) / 12.4 (women).

## Settings (tuned on 2021-22 to 2024-25, tested on 2025-26)
| Setting | Value |
|---|---|
| Preseason prior | roster talent (returning players + transfers rated from old box scores + newcomers at -4) blended with last season's team rating; worth 8 games (men), 3 (women) |
| Recency weighting | off (no gain) |
| Blowout cap | off (capping hurt, especially for women's) |
| Home court | measured each full season (~3.3 men, ~3.2 women), carried into the next season firmly |

## Accuracy, 2025-26 (never used for tuning)
| | Men's | Women's |
|---|---|---|
| Avg miss, Nov-Dec | 9.70 pts | 10.41 pts |
| Avg miss, Feb-Apr | 8.82 pts | 9.24 pts |
| Avg miss, full season | 9.19 pts | 9.70 pts |
| Elo baseline, Feb-Apr | 8.93 pts | 9.85 pts |
