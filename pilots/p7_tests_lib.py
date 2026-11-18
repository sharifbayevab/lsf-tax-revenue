"""Paired circular moving-block bootstrap over consecutive quarterly origins (overlapping 12m windows share
targets) for WAPE differences (proposed - baseline), bottom and total level. Block lengths 1 (naive), 4 (one year
of origins, primary), 8 (sensitivity)."""
import numpy as np, pandas as pd
from common import load_panel, RES, DATASET

def per_origin(n, df):
    m = pd.read_parquet(RES / f"fc_{n}.parquet").merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    b = m.assign(ae=(m.y - m.yhat).abs(), ay=m.y.abs()).groupby("origin")[["ae", "ay"]].sum()
    t = m.groupby(["origin", "ds"])[["y", "yhat"]].sum()
    tt = t.assign(ae=(t.y - t.yhat).abs()).groupby("origin")[["ae", "y"]].sum()
    return b, tt

def ci(A, B, key_ae, key_y, rng, R=5000, L=4):
    o = A.index; n = len(o); d = []
    for _ in range(R):
        starts = rng.integers(0, n, int(np.ceil(n / L)))
        i = np.concatenate([(np.arange(s, s + L) % n) for s in starts])[:n]
        d.append(A[key_ae].iloc[i].sum() / A[key_y].iloc[i].sum() - B[key_ae].iloc[i].sum() / B[key_y].iloc[i].sum())
    pt = A[key_ae].sum() / A[key_y].sum() - B[key_ae].sum() / B[key_y].sum()
    return pt, *np.quantile(d, [.025, .975]), (A[key_ae] < B[key_ae]).mean()

