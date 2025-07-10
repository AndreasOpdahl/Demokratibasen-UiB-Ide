"""
Les judge_scores_*.csv ➜ avgjør vinner A/B/X (uavgjort)
• per kriterium
• samlet med vektet gjennomsnitt
"""

import csv, pathlib

LABEL_A = "gemini"
LABEL_B = "claude"

ROOT      = pathlib.Path(__file__).resolve().parent
IN_CSV    = ROOT / f"judge_scores_{LABEL_A}_vs_{LABEL_B}.csv"
OUT_CSV   = ROOT / f"judge_winners_{LABEL_A}_vs_{LABEL_B}.csv"

WEIGHTS = {
    "koherens": 0.20,
    "konsistens": 0.30,
    "flyt": 0.15,
    "relevans": 0.35
}

def winner(a: int, b: int) -> str:
    return "A" if a > b else ("B" if b > a else "X")

def main() -> None:
    rows_out = []

    with IN_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Noen ganger følger det med en Pandas-indeks-kolonne – hopp over den
            if "dokument_id" not in row:
                continue

            did = row["dokument_id"]

            # beregn vinner per kriterium 
            per_crit = {}
            for crit in WEIGHTS.keys():
                col_a = f"{crit}_{LABEL_A}"
                col_b = f"{crit}_{LABEL_B}"
                per_crit[crit] = winner(float(row[col_a]), float(row[col_b]))

            # beregn total score
            total_A = sum(
                WEIGHTS[crit] * float(row[f"{crit}_{LABEL_A}"])
                for crit in WEIGHTS
            )
            total_B = sum(
                WEIGHTS[crit] * float(row[f"{crit}_{LABEL_B}"])
                for crit in WEIGHTS
            )
            total_w = winner(total_A, total_B)

            rows_out.append([
                did,
                per_crit["koherens"],
                per_crit["konsistens"],
                per_crit["flyt"],
                per_crit["relevans"],
                total_w,
            ])

    # Skriv resultatfil 
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dokument_id",
            f"vinner_koherens",
            f"vinner_konsistens",
            f"vinner_flyt",
            f"vinner_relevans",
            "vinner_totalt",
        ])
        writer.writerows(rows_out)

    print("Ferdig – resultat lagret i", OUT_CSV.relative_to(ROOT))

if __name__ == "__main__":
    main()