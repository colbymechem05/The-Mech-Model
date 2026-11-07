"""Move one day's logged lines for games that have NOT tipped yet out of the prediction log, so the
next daily run remakes them. They're kept, with the reason, in logs/superseded_{league}.csv (nothing
is deleted). Use only for a documented mistake, never for a game that has started.

Run: python scripts/supersede_lines.py 2026-11-07 "reason text" [--dry]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import daily  # noqa: E402
import export as E  # noqa: E402

if __name__ == "__main__":
    day, reason, dry = sys.argv[1], sys.argv[2], "--dry" in sys.argv
    now = daily.NOW()
    for lg in ["mbb", "wbb"]:
        f = ROOT / "logs" / f"predictions_{lg}.csv"
        log = pd.read_csv(f)
        tip = pd.to_datetime(log.game_id.map(E.tip_times(lg)), utc=True, errors="coerce")
        move = (log.date == day) & tip.notna() & (tip > now)
        started = (log.date == day) & ~move
        if started.any():
            print(f"{lg}: keeping {started.sum()} lines for games that have started or have no tip time")
        if dry:
            print(f"{lg}: would move {move.sum()} untipped lines from {day}")
            continue
        out = log[move].assign(superseded_at=now.isoformat(timespec="minutes"), reason=reason)
        sf = ROOT / "logs" / f"superseded_{lg}.csv"
        out.to_csv(sf, mode="a", header=not sf.exists(), index=False)
        log[~move].to_csv(f, index=False)
        print(f"{lg}: moved {move.sum()} untipped lines from {day} to {sf.name}")
