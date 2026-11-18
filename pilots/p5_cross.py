"""Cross-country summary: same model ladder on Brazil (br) and Argentina (ar)."""
import os, sys, pandas as pd
from common import load_panel, RES, mase_scales, evaluate, DATASET
from p4_oracle import scores
SURV = {"br": "focus", "ar": "rem", "super": "rem"}.get(DATASET, "naive")
LADDER = [("seasonal naive", "snaive"), ("ETS", "ets"), ("AutoARIMA", "arima"), ("AutoTheta", "theta"),
          ("Chronos-2 nominal (xl by tax)", "c2_xl_tax"),
          ("Chronos-2 real, naive infl.", "chronos_real_naive"),
          ("Theta real, survey infl. x GDP", f"theta_real_{SURV}_gdp1"),
          ("Chronos-2 real, survey infl. x GDP", f"chronos_real_{SURV}_gdp1"),
          ("  + rake to national taxes", f"rec_chronos_chronos_{SURV}_rake_tax"),
          ("  + adaptive rake rule", f"rec_chronos_chronos_{SURV}_rake_sel")]
df = load_panel(); sc = mase_scales(df); rows = {}
for lab, n in LADDER:
    f = RES / f"fc_{n}.parquet"
    if not f.exists(): print("missing", n); continue
    fc = pd.read_parquet(f); r, _ = evaluate(fc, df, sc, by_h=False); s = scores(fc, df)
    rows[lab] = {"MASE_med": r["MASE_med"], "WAPE_bottom": r["WAPE_bottom"], "WAPE_tax": r["WAPE_tax"],
                 "WAPE_state": r["WAPE_state"], "WAPE_total": r["WAPE_total"], "APE_12m": s[1], "bias": s[3]}
t = pd.DataFrame(rows).T; print(DATASET); print(t.round(3).to_string()); t.to_csv(RES / "cross_table.csv")
