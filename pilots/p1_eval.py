"""Evaluate all fc_*.parquet in results/ -> results/p1_table.csv"""
import pandas as pd, numpy as np, sys
from common import load_panel, evaluate, mase_scales, RES
df = load_panel(); sc = mase_scales(df)
rows = []
for f in sorted(RES.glob("fc_*.parquet")):
    fc = pd.read_parquet(f)
    r, _ = evaluate(fc, df, sc); r["model"] = f.stem[3:]; rows.append(r)
t = pd.DataFrame(rows).set_index("model").drop(columns="n")
t = t.sort_values("MASE")
pd.set_option("display.width", 250)
print(t.round(3).to_string())
t.to_csv(RES / "p1_table.csv")
