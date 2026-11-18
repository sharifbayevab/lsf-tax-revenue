"""Level-layer ablation: deflation x inflation path x GDP anchor, for Chronos-2 and Theta (bottom-up, no reconciliation)."""
import pandas as pd
from common import load_panel, RES, mase_scales, evaluate, DATASET
from p4_oracle import scores
S = {"br": "focus", "ar": "rem", "super": "rem"}.get(DATASET, "naive")
cfg = [("nominal", "{n}"), ("real, naive infl.", "{m}_real_naive"), ("real, naive infl. x GDP", "{m}_real_naive_gdp1"),
       (f"real, survey infl.", "{m}_real_" + S), (f"real, survey infl. x GDP", "{m}_real_" + S + "_gdp1")]
df = load_panel(); sc = mase_scales(df); rows = []
for model, nom in [("Chronos-2", "c2_xl_tax"), ("Theta", "theta")]:
    for lab, pat in cfg:
        n = pat.format(n=nom, m="chronos" if model == "Chronos-2" else "theta")
        f = RES / f"fc_{n}.parquet"
        if not f.exists(): rows.append({"model": model, "level layer": lab, "file": n}); continue
        fc = pd.read_parquet(f); r, _ = evaluate(fc, df, sc, by_h=False); s = scores(fc, df)
        rows.append({"model": model, "level layer": lab, "WAPE_bottom": r["WAPE_bottom"], "WAPE_state": r["WAPE_state"],
                     "WAPE_total": r["WAPE_total"], "APE_12m": s[1], "bias": s[3]})
t = pd.DataFrame(rows); print(DATASET); print(t.round(3).to_string(index=False)); t.to_csv(RES / "p8_ablation.csv", index=False)
