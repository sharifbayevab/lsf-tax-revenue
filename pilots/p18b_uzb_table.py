"""Write paper/tables/tab_uzb.tex from results/uzb/annual_check.csv (p18_uzb_annual.py)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
r = pd.read_csv(ROOT / "results" / "uzb" / "annual_check.csv")

ROWS = [  # (label, method, nominal counterpart for the win count)
    ("Na\\\"ive (last year)", "Naive", None),
    ("Drift", "Drift", None),
    ("AutoTheta, nominal", "AutoTheta nominal", None),
    ("Chronos-2, nominal", "Chronos-2 nominal", None),
    ("Level layer alone (flat real)", "Level only (survey, e=0)", None),
    ("AutoTheta + level layer", "AutoTheta + level (survey, e=0)", "AutoTheta nominal"),
    ("LSF (Chronos-2 + level layer)", "LSF (survey, e=0)", "Chronos-2 nominal"),
    ("LSF, naive price path", "LSF (naive, e=0)", "Chronos-2 nominal"),
]

piv = r.pivot(index="method", columns="target", values="total_ape")
lines = []
for i, (lab, m, ref) in enumerate(ROWS):
    s = r[r.method == m]
    wins = "--" if ref is None else f"{int((piv.loc[m] < piv.loc[ref]).sum())} of {piv.shape[1]}"
    lines.append(f"{lab} & {100 * s.total_ape.mean():.1f} & {100 * s.total_err.mean():+.1f} & "
                 f"{100 * s.leaf_wape.mean():.1f} & {wins} \\\\")
    if i == 3:
        lines.append("\\midrule")
out = ROOT / "paper" / "tables" / "tab_uzb.tex"
out.write_text("\n".join(lines) + "\n")
print(out.read_text())

# Table 1 row (panels)
y = pd.read_csv(ROOT / "results" / "uzb" / "annual_check.csv")
row = (f"Uzbekistan$^\\S$ & Min. Finance, data.egov.uz (national) & 1 & 12 & 12 & "
       f"2015--2025 & {y.origin.nunique()} & CC BY-SA \\\\\n")
(ROOT / "paper" / "tables" / "tab_data_uzb.tex").write_text(row)
print(row)
