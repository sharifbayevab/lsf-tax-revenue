"""P12: 'level as covariate' — the natural competitor to LSF's multiplicative level layer.
Chronos-2 forecasts NOMINAL receipts with log price index as a covariate: past = realised log CPI, future = the same
origin-dated expected price path LSF uses (survey path for BR/AR, naive elsewhere). Same grouping (by tax,
cross-learning) and median as LSF. Variant 'cov_real': deflated target + the same covariate, re-inflated with the path
(separates encoding from multiplicative recombination).
usage: DATASET=br|ar|es|pe p12_covariate.py [cov_nom|cov_real]  -> results/<ds>/fc_c2_<variant>_<path>.parquet"""
import sys, os, numpy as np, pandas as pd
from common import load_panel, ORIGINS, H, save_fc, DATASET
from macro import ipca_index, focus, expected_path

PATH = {"br": "focus", "ar": "rem", "super": "rem"}.get(DATASET, "naive")


def main(variant):
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")
    df = load_panel(); P = ipca_index(); F = focus("IPCA")
    df["lp"] = np.log(df.ds.map(P))
    if variant == "cov_real":
        df["y"] = df.y / df.ds.map(P)
    out = []
    for o in ORIGINS:
        e = expected_path(o, P, F, H, PATH)
        tr = df[df.ds <= o]
        parts = []
        for _, g in tr.groupby("tax"):
            ids = g.unique_id.unique()
            fut = pd.DataFrame({"unique_id": np.repeat(ids, H), "ds": np.tile(e.index, len(ids)),
                                "lp": np.tile(np.log(e.values), len(ids))})
            p = pipe.predict_df(g[["unique_id", "ds", "y", "lp"]], future_df=fut, id_column="unique_id",
                                timestamp_column="ds", target="y", prediction_length=H, quantile_levels=[0.5],
                                batch_size=512, cross_learning=True)
            parts.append(p.rename(columns={"0.5": "yhat"})[["unique_id", "ds", "yhat"]])
        p = pd.concat(parts)
        if variant == "cov_real":
            p["yhat"] = p.yhat * p.ds.map(e)
        p["origin"] = o; p["h"] = p.groupby("unique_id").cumcount() + 1
        out.append(p); print(DATASET, variant, o.date(), flush=True)
    save_fc(pd.concat(out), f"c2_{variant}_{PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cov_nom")
