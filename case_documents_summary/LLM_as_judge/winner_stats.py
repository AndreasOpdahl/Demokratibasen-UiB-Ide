# LLM_as_judge/winner_stats.py
"""
Steg 3: Les judge_winners_<A>_vs_<B>.csv
        ➜ hvor mange A-, B- og X-utfall per kriterium og totalt
"""

import csv, pathlib
from collections import Counter, defaultdict

# ---------- SETT MODELLNAVN ----------
LABEL_A = "gemini"
LABEL_B = "claude"
# ─────────────────────────────────────────────

ROOT      = pathlib.Path(__file__).resolve().parent
IN_CSV    = ROOT / f"judge_winners_{LABEL_A}_vs_{LABEL_B}.csv"
OUT_CSV   = ROOT / f"winner_stats_{LABEL_A}_vs_{LABEL_B}.csv"

CRITERIA  = ["koherens", "konsistens", "flyt", "relevans", "totalt"]

def main() -> None:
    # dict: crit -> Counter({'A':…, 'B':…, 'X':…})
    stats = defaultdict(Counter)

    with IN_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for crit in CRITERIA:
                col = "vinner_" + crit
                if col not in row:
                    continue
                stats[crit][row[col].strip()] += 1

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kriterium",
            f"seiere_{LABEL_A}",
            f"seiere_{LABEL_B}",
            "uavgjort",
            f"{LABEL_A}_andel",
            f"{LABEL_B}_andel",
        ])

        for crit in CRITERIA:
            a = stats[crit].get("A", 0)
            b = stats[crit].get("B", 0)
            x = stats[crit].get("X", 0)
            played = a + b                # kamper som ikke endte X (uavgjort)
            a_pct = a / played if played else 0
            b_pct = b / played if played else 0
            writer.writerow([crit, a, b, x, f"{a_pct:.3f}", f"{b_pct:.3f}"])

    print("Ferdig – statistikk lagret i", OUT_CSV.relative_to(ROOT))

if __name__ == "__main__":
    main()