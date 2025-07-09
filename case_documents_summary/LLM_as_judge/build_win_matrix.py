"""
Bygg en N×N matrise over seiers-andeler mellom alle modeller
(brukt 'totalt'-kriteriet fra winner_stats_<A>_vs_<B>.csv).
"""

import csv, pathlib, re, itertools
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
PAT  = re.compile(r"winner_stats_(.+?)_vs_(.+?)\.csv")

# Finn alle stats-filer og hvilke modeller som finnes
files = list(ROOT.glob("winner_stats_*_vs_*.csv"))
pairs = [PAT.match(f.name).groups() for f in files]
models = sorted({m for pair in pairs for m in pair})

# matrise: dict[(i,j)] = prosent modell i vant over j   (0–1)
win = {(i, j): None for i in models for j in models}

for a, b in pairs:
    path = ROOT / f"winner_stats_{a}_vs_{b}.csv"
    with path.open(encoding="utf-8") as f:
        row_tot = next(r for r in csv.DictReader(f) if r["kriterium"] == "totalt")

    # kolonnenavn følger mønsteret <label>_andel
    win[(a, b)] = float(row_tot[f"{a}_andel"])
    win[(b, a)] = float(row_tot[f"{b}_andel"])

# Egen-diagonalen = NaN 
for m in models:
    win[(m, m)] = float("nan")

# Lag DataFrame, skriv CSV 
matrix = pd.DataFrame(
    [[win[(i, j)] for j in models] for i in models],
    index=models, columns=models
)

out_csv = ROOT / "model_win_matrix.csv"
matrix.to_csv(out_csv, float_format="%.3f")
print("Matrise lagret ➜", out_csv.relative_to(ROOT))
print("\n", matrix.round(3))