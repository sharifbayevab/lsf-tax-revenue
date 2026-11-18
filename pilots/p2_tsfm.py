"""P2 zero-shot TSFMs on real-terms series (deflate by IPCA, re-inflate with trailing inflation).
usage: p2_tsfm.py timesfm|tirex|chronos2"""
import sys, time, numpy as np, pandas as pd, torch
from common import load_panel, ORIGINS, H, save_fc, RES, DATASET
PATH = {"ar": "rem", "super": "rem"}.get(DATASET, "naive")
from macro import ipca_index, focus, expected_path

def get_model(name):
    if name == "timesfm":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True,
                  use_continuous_quantile_head=True, force_flip_invariance=True,
                  infer_is_positive=False, fix_quantile_crossing=True))
        def f(ctx):
            pt, q = m.forecast(horizon=H, inputs=ctx)
            return np.asarray(q)[:, :, 5]  # median of [mean, q10..q90]
        return f
    if name == "tirex":
        from tirex import load_model
        m = load_model("NX-AI/TiRex")
        def f(ctx):
            L = max(len(c) for c in ctx)
            X = torch.full((len(ctx), L), float("nan"))
            for i, c in enumerate(ctx): X[i, L - len(c):] = torch.tensor(c, dtype=torch.float32)
            q, mean = m.forecast(context=X, prediction_length=H)
            return q[:, :, 4].cpu().numpy()  # median of 9 quantiles
        return f
    raise ValueError(name)

def main(name):
    df = load_panel(); P = ipca_index(); F = focus("IPCA")
    df["yr"] = df.y / df.ds.map(P)
    f = get_model(name); out = []; t0 = time.time()
    for o in ORIGINS:
        tr = df[df.ds <= o]
        ids, ctx = [], []
        for u, g in tr.groupby("unique_id"):
            ids.append(u); ctx.append(g.yr.values.astype(np.float32))
        yh = f(ctx)
        months = pd.date_range(o + pd.offsets.MonthBegin(1), periods=H, freq="MS")
        path = expected_path(o, P, F, H, PATH).values
        fc = pd.DataFrame({"unique_id": np.repeat(ids, H), "ds": np.tile(months, len(ids)),
                           "h": np.tile(np.arange(1, H + 1), len(ids)),
                           "yhat": (yh * path[None, :]).ravel(), "origin": o})
        out.append(fc); print(name, o.date(), f"{time.time()-t0:.0f}s", flush=True)
    save_fc(pd.concat(out), f"{name}_real_{PATH}")

if __name__ == "__main__":
    main(sys.argv[1])
