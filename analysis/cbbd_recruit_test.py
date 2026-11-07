"""Men's freshmen: RSCI top 100 (scraped) vs. CollegeBasketballData.com 247 composite top 100.
Same walk-forward test as analysis/recruit_test.py (tune 2022-23..2024-25, hold out 2025-26).
The RSCI files were removed after this test (still in git history before Nov 6, 2026),
so only the saved results remain meaningful. Run: python analysis/cbbd_recruit_test.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D              # noqa: E402
import talent as T            # noqa: E402
from recruit_test import LG, SEASONS, run   # noqa: E402

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    g = D.load_games(LG)
    confs = {s: D.conferences(LG, s) for s in SEASONS}
    rows = []
    for src in ["rsci", "cbbd"]:
        tp = T.TalentPrior(LG, g, SEASONS, train=SEASONS[:-1], new_value=-4.0, conf_feature=True,
                           recruit_mode="feature", recruit_source=src)
        rows.append({"source": src, **run(g, tp, confs), "recruit weight": tp.coef[-1]})
        print(f"\n{src}: what a freshman adds by tier")
        print(tp.tier_table.round(2).to_string(), flush=True)
    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
