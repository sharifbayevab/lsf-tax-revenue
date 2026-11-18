"""P1 Chronos-2 zero-shot. Variants:
  indep     - each series alone
  xl_tax    - cross-learning inside groups = same tax, all 27 states
  xl_state  - cross-learning inside groups = same state, all taxes
  xl_all    - cross-learning in fixed random batches of ~100 series (control)
"""
import sys, time
import numpy as np, pandas as pd, torch
from chronos import Chronos2Pipeline
from common import load_panel, ORIGINS, H, save_fc, RES

def main(variant, device):
    df = load_panel()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    if variant == "indep":
        groups = [df.unique_id.unique()]
    elif variant == "xl_tax":
        groups = [g.unique_id.unique() for _, g in df.groupby("tax")]
    elif variant == "xl_state":
        groups = [g.unique_id.unique() for _, g in df.groupby("UF")]
    elif variant == "xl_all":
        ids = df.unique_id.unique().copy(); np.random.default_rng(0).shuffle(ids)
        groups = np.array_split(ids, int(np.ceil(len(ids) / 100)))
    out = []; t0 = time.time()
    for o in ORIGINS:
        tr = df[df.ds <= o]
        for g in groups:
            sub = tr[tr.unique_id.isin(g)][["unique_id", "ds", "y"]]
            p = pipe.predict_df(sub, id_column="unique_id", timestamp_column="ds", target="y",
                                prediction_length=H, quantile_levels=[0.1, 0.5, 0.9],
                                batch_size=max(len(g), 1) if variant != "indep" else 256,
                                cross_learning=(variant != "indep"))
            p = p.rename(columns={"0.5": "yhat", "0.1": "q10", "0.9": "q90"})
            p["origin"] = o
            p["h"] = p.groupby("unique_id").cumcount() + 1
            out.append(p[["unique_id", "origin", "ds", "h", "yhat", "q10", "q90"]])
        print(variant, o.date(), f"{time.time()-t0:.0f}s", flush=True)
    fc = pd.concat(out)
    fc.to_parquet(RES / f"fcq_c2_{variant}.parquet", index=False)
    save_fc(fc, f"c2_{variant}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mps")
