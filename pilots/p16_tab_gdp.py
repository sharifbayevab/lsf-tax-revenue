"""Table for P14: national-total WAPE (%) under GDP-growth multiplier elasticity e = 0 / 0.5 / 1 and bias of the
12-month sum at e = 1. Reads results/{br_rt,ar,super}/p14_gdp_elast.csv -> paper/tables/tab_gdp.tex"""
import os, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
t = pd.concat(pd.read_csv(f"{R}/results/{d}/p14_gdp_elast.csv") for d in ["br_rt", "ar", "super"])
PAN = {"br": "Brazil (tax)", "ar": "Argentina (tax)", "super": "Argentina (supermarkets)"}
MOD = {"chronos": "Chronos-2", "theta": "AutoTheta", "lgbm": "LightGBM", "tirex": "TiRex", "timesfm": "TimesFM-2.5",
       "rec_chronos_chronos_focus_rake_tax": "Chronos-2, raked", "rec_chronos_chronos_focus_rake_tax_buoy": "Chronos-2, raked + gap"}
rows, last = [], None
for (pan, mod), g in t.groupby(["panel", "model"], sort=False):
    g = g.set_index("e"); key = mod.split("_real_")[0] if "_real_" in mod else mod
    w = [100 * g.WAPE_total[e] for e in (0.0, 0.5, 1.0)]; b = min(w)
    cells = [(r"\textbf{%.1f}" % x) if x == b else "%.1f" % x for x in w]
    name = PAN[pan] if pan != last else ""; last = pan
    rows.append(f"{name} & {MOD[key]} & " + " & ".join(cells) + f" & {100*g.bias12[1.0]:+.1f} \\\\")
open(f"{R}/paper/tables/tab_gdp.tex", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
