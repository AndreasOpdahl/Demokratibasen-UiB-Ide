# LLM_as_judge/compute_winners.py
"""
Steg 2: Les judge_scores.csv ➜ avgjør vinner A/B/X (uavgjort)
• per kriterium
• samlet (lik vekt ¼)
"""

import csv, pathlib

ROOT      = pathlib.Path(__file__).resolve().parent
IN_CSV    = ROOT / "judge_scores.csv"
OUT_CSV   = ROOT / "judge_winners.csv"

WEIGHTS = {
    "koherens": 0.20,
    "konsistens": 0.30,
    "flyt": 0.15,
    "relevans": 0.35
}

def winner(a: int, b: int) -> str:
    return "A" if a > b else ("B" if b > a else "X")

def main():
    rows_out = []

    with IN_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = row["dokument_id"]
            coh_w = winner(int(row["koherens_A"]),  int(row["koherens_B"]))
            con_w = winner(int(row["konsistens_A"]), int(row["konsistens_B"]))
            flu_w = winner(int(row["flyt_A"]),  int(row["flyt_B"]))
            rel_w = winner(int(row["relevans_A"]),  int(row["relevans_B"]))

            # samlet vekt
            total_A = (
                WEIGHTS["koherens"]   * float(row["koherens_A"])   +
                WEIGHTS["konsistens"] * float(row["konsistens_A"]) +
                WEIGHTS["flyt"]       * float(row["flyt_A"])       +
                WEIGHTS["relevans"]   * float(row["relevans_A"])
            )

            total_B = (
                WEIGHTS["koherens"]   * float(row["koherens_B"])   +
                WEIGHTS["konsistens"] * float(row["konsistens_B"]) +
                WEIGHTS["flyt"]       * float(row["flyt_B"])       +
                WEIGHTS["relevans"]   * float(row["relevans_B"])
            )
            tot_w = winner(total_A, total_B)

            rows_out.append([did, coh_w, con_w, flu_w, rel_w, tot_w])

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dokument_id",
            "vinner_koherens",
            "vinner_konsistens",
            "vinner_flyt",
            "vinner_relevans",
            "vinner_totalt"
        ])
        writer.writerows(rows_out)

    print("Ferdig. Resultat lagret i", OUT_CSV.relative_to(ROOT))

if __name__ == "__main__":
    main()