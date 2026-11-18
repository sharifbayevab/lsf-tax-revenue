"""P12: 'level as covariate' — the natural competitor to LSF's multiplicative level layer.
Chronos-2 forecasts NOMINAL receipts with log price index as a covariate: past = realised log CPI, future = the same
origin-dated expected price path LSF uses (survey path for BR/AR, naive elsewhere). Same grouping (by tax,
cross-learning) and median as LSF. Variant 'cov_real': deflated target + the same covariate, re-inflated with the path
(separates encoding from multiplicative recombination).
Review #5 (M3) encoding/scaling cells: 'cov_level' = nominal target, CPI level (not log) as covariate;
'log_nom' = log nominal target, no covariate; 'log_cov' = log nominal target + log CPI covariate (both exponentiated
back; in logs a price-level shift is additive, which tests whether instance scaling is the obstacle).
usage: DATASET=br|ar|es|pe p12_covariate.py [cov_nom|cov_real|cov_level|log_nom|log_cov]  -> results/<ds>/fc_c2_<variant>_<path>.parquet"""
import sys, os, numpy as np, pandas as pd
from common import load_panel, ORIGINS, H, save_fc, DATASET
from macro import ipca_index, focus, expected_path

PATH = {"br": "focus", "ar": "rem", "super": "rem"}.get(DATASET, "naive")


def main(variant):
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")
    df = load_panel(); P = ipca_index(); F = focus("IPCA")
    df["lp"] = df.ds.map(P) if variant == "cov_level" else np.log(df.ds.map(P))
    if variant == "cov_real":
        df["y"] = df.y / df.ds.map(P)
    if variant.startswith("log_"):
        floor = df[df.y > 0].groupby("unique_id").y.min().reindex(df.unique_id).values
        df["y"] = np.log(np.maximum(df.y.values, floor))   # non-positive months -> series minimum (<=10% by the filter)
    use_cov = variant != "log_nom"
    out = []
    for o in ORIGINS:
        e = expected_path(o, P, F, H, PATH)
        tr = df[df.ds <= o]
        parts = []
        for _, g in tr.groupby("tax"):
            ids = g.unique_id.unique()
            fv = e.values if variant == "cov_level" else np.log(e.values)
            fut = pd.DataFrame({"unique_id": np.repeat(ids, H), "ds": np.tile(e.index, len(ids)),
                                "lp": np.tile(fv, len(ids))})
            cols = ["unique_id", "ds", "y", "lp"] if use_cov else ["unique_id", "ds", "y"]
            p = pipe.predict_df(g[cols], future_df=fut if use_cov else None, id_column="unique_id",
                                timestamp_column="ds", target="y", prediction_length=H, quantile_levels=[0.5],
                                batch_size=512, cross_learning=True)
            parts.append(p.rename(columns={"0.5": "yhat"})[["unique_id", "ds", "yhat"]])
        p = pd.concat(parts)
        if variant == "cov_real":
            p["yhat"] = p.yhat * p.ds.map(e)
        if variant.startswith("log_"):
            p["yhat"] = np.exp(p.yhat)
        p["origin"] = o; p["h"] = p.groupby("unique_id").cumcount() + 1
        out.append(p); print(DATASET, variant, o.date(), flush=True)
    save_fc(pd.concat(out), f"c2_{variant}_{PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cov_nom")
