"""Four-country summary (BR, AR, ES, PE): bottom / total WAPE + 12m APE for the common model ladder,
plus test-period inflation level/volatility. Writes results/four_country.csv"""
import os, subprocess, json, numpy as np, pandas as pd
rows = []
for d in ["br", "ar", "es", "pe"]:
    os.environ["DATASET"] = d
    import importlib, common, macro, p4_oracle
    importlib.reload(common); importlib.reload(macro); importlib.reload(p4_oracle)
    from common import load_panel, RES, mase_scales, evaluate
    S = {"br": "focus", "ar": "rem", "super": "rem"}.get(d, "naive")
    lad = [("ETS", "ets"), ("AutoARIMA", "arima"), ("AutoTheta", "theta"), ("LightGBM real", "lgbm_real_naive"),
           ("Chronos-2 nominal", "c2_xl_tax"), ("Theta + level layer", f"theta_real_{S}_gdp1"),
           ("Chronos-2 + level layer", f"chronos_real_{S}_gdp1")]
    df = load_panel(); sc = mase_scales(df); P = macro.ipca_index()
    infl = (P / P.shift(12) - 1).loc["2019-12-01":"2025-12-01"] * 100
    for lab, n in lad:
        f = RES / f"fc_{n}.parquet"
        if not f.exists(): continue
        fc = pd.read_parquet(f); r, _ = evaluate(fc, df, sc, by_h=False); s = p4_oracle.scores(fc, df)
        rows.append({"country": d, "model": lab, "bottom": r["WAPE_bottom"], "state": r["WAPE_state"],
                     "total": r["WAPE_total"], "APE12": s[1], "bias": s[3],
                     "infl_mean": infl.mean(), "infl_sd": infl.std()})
t = pd.DataFrame(rows)
for k in ["bottom", "total", "APE12"]:
    print(k); print(t.pivot(index="model", columns="country", values=k).round(3).to_string())
print(t.groupby("country")[["infl_mean", "infl_sd"]].first().round(1))
t.to_csv("../results/four_country.csv", index=False)
