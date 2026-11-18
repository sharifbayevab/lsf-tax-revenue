"""Shared protocol for the Brazil tax-revenue pilots.

Rolling-origin evaluation: for each origin T (last observed month), every model
sees y[<=T] only and forecasts T+1..T+12. Origins are quarterly from 2019-12 to
2025-09 (24 origins); the last one ends exactly at the data end (2026-09).
Forecast files: results/fc_<model>.parquet with columns
  unique_id, origin, ds, h, yhat
"""
from pathlib import Path
import os
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "br")   # br = Brazil RFB (main), ar = Argentina provincial (external)
PANEL = ROOT / ("data/processed/brazil_panel.parquet" if DATASET == "br" else f"data/processed/ext_{ {'ar': 'argentina', 'es': 'spain', 'pe': 'peru', 'super': 'super'}.get(DATASET, DATASET)}_panel.parquet")
RES = ROOT / ("results" if DATASET == "br" else f"results/{DATASET}")
if os.environ.get("KEPT") == "kept_rt":
    RES = RES / "rt" if DATASET != "br" else ROOT / "results/br_rt"
RES.mkdir(exist_ok=True, parents=True)
H = 12
SEASON = 12
ORIGINS = pd.date_range("2019-12-01", {"br": "2025-09-01", "super": "2025-06-01"}.get(DATASET, "2024-12-01"), freq="3MS")
COVID_ORIGINS = pd.to_datetime(["2019-12-01", "2020-03-01"])


KEPT = os.environ.get("KEPT", "kept" if DATASET == "br" else "kept")  # "kept_rt" = series filter decided on data <= first origin


def load_panel(kept_only=True):
    df = pd.read_parquet(PANEL)
    if kept_only:
        df = df[df[KEPT]]
    df = df[["unique_id", "UF", "tax", "ds", "y"]].sort_values(["unique_id", "ds"])
    return df.reset_index(drop=True)


def wide(df):
    """ds x unique_id matrix."""
    return df.pivot(index="ds", columns="unique_id", values="y").sort_index()


def train_slice(df, origin):
    return df[df.ds <= origin]


def save_fc(fc, name):
    fc = fc[["unique_id", "origin", "ds", "h", "yhat"]].copy()
    fc.to_parquet(RES / f"fc_{name}.parquet", index=False)
    print(f"saved {name}: {len(fc)} rows")


def mase_scales(df):
    """Per (series, origin) in-sample seasonal-naive MAE on the training window."""
    W = wide(df)
    rows = []
    for o in ORIGINS:
        tr = W.loc[:o]
        d = (tr - tr.shift(SEASON)).abs().iloc[SEASON:]
        sc = d.mean()
        rows.append(pd.DataFrame({"unique_id": sc.index, "origin": o, "scale": sc.values}))
    return pd.concat(rows)


def evaluate(fc, df, scales=None, by_h=True):
    """Return dict of headline metrics for one forecast table."""
    meta = df[["unique_id", "UF", "tax"]].drop_duplicates()
    m = fc.merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")
    m = m.merge(meta, on="unique_id")
    if scales is None:
        scales = mase_scales(df)
    m = m.merge(scales, on=["unique_id", "origin"])
    m["ae"] = (m.y - m.yhat).abs()
    m["mase"] = m.ae / m.scale.replace(0, np.nan)
    out = {}
    out["MASE"] = m.mase.mean()
    out["MASE_med"] = m.mase.median()
    out["WAPE_bottom"] = m.ae.sum() / m.y.abs().sum()
    # national by tax (bottom-up sums of the same forecasts)
    nt = m.groupby(["tax", "origin", "ds"])[["y", "yhat"]].sum()
    out["WAPE_tax"] = (nt.y - nt.yhat).abs().sum() / nt.y.abs().sum()
    # state totals
    st = m.groupby(["UF", "origin", "ds"])[["y", "yhat"]].sum()
    out["WAPE_state"] = (st.y - st.yhat).abs().sum() / st.y.abs().sum()
    tot = m.groupby(["origin", "ds"])[["y", "yhat"]].sum()
    out["WAPE_total"] = (tot.y - tot.yhat).abs().sum() / tot.y.abs().sum()
    cov = m[m.origin.isin(COVID_ORIGINS)]
    out["MASE_covid"] = cov.mase.mean()
    calm = m[~m.origin.isin(COVID_ORIGINS)]
    out["MASE_calm"] = calm.mase.mean()
    if by_h:
        for lo, hi in [(1, 3), (4, 6), (7, 12)]:
            out[f"MASE_h{lo}-{hi}"] = m[(m.h >= lo) & (m.h <= hi)].mase.mean()
    out["n"] = len(m)
    return out, m


def future_dates(origin, h=H):
    return pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
