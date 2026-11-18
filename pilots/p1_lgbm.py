"""P1 global LightGBM, direct multi-horizon, one model per origin.
Sample = (series, anchor t, horizon h). Features: 24 lags of y/s_t, where s_t = mean |y| over
t-11..t; tax, UF (categorical); target month; h; yoy growth of 12m sums.
Target: y[t+h]/s_t. Loss L1.
"""
import os
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import load_panel, wide, ORIGINS, H, save_fc

L = int(os.environ.get("LGBM_L", 24))  # lags; Argentina uses 12 (only 36 months at first origin)

def build(W, taxes, ufs, anchors, with_target=True):
    Y = W.values  # T x N
    idx = {d: i for i, d in enumerate(W.index)}
    X, y, meta = [], [], []
    N = Y.shape[1]
    for a in anchors:
        i = idx[a]
        if i < L - 1 + 12:
            continue
        lags = Y[i - L + 1:i + 1][::-1]          # L x N, lag0 first
        s = np.abs(Y[i - 11:i + 1]).mean(0) + 1e-6
        prev = np.abs(Y[i - 23:i - 11]).sum(0) + 1e-6
        yoy = Y[i - 11:i + 1].sum(0) / prev
        base = np.column_stack([lags.T / s[:, None], yoy, np.log(s)])
        for h in range(1, H + 1):
            j = i + h
            if with_target and j >= len(Y):
                continue
            tm = (a.month + h - 1) % 12 + 1
            f = np.column_stack([base, np.full(N, h), np.full(N, tm), taxes, ufs])
            X.append(f)
            meta.append(np.column_stack([np.arange(N), np.full(N, h)]))
            if with_target:
                y.append(Y[j] / s)
    X = np.vstack(X); meta = np.vstack(meta)
    y = np.concatenate(y) if with_target else None
    return X, y, meta

def main(deflate=None):
    df = load_panel()
    W = wide(df).fillna(0.0)
    if deflate:
        from macro import ipca_index, focus, expected_path
        P = ipca_index(); F = focus("IPCA")
        W = W.div(P.reindex(W.index), axis=0)
    ids = W.columns
    meta_s = df[["unique_id", "tax", "UF"]].drop_duplicates().set_index("unique_id").loc[ids]
    taxes = meta_s.tax.astype("category").cat.codes.values
    ufs = meta_s.UF.astype("category").cat.codes.values
    ncol = L + 2 + 4
    cat = [ncol - 2, ncol - 1]
    params = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
                  num_threads=8)
    out = []; t0 = time.time()
    for o in ORIGINS:
        Wt = W.loc[:o]
        # training anchors: all t with t+h <= o is enforced by with_target (j < len)
        X, y, _ = build(Wt, taxes, ufs, Wt.index, True)
        ok = np.isfinite(y) & (np.abs(y) < 50)
        ds = lgb.Dataset(X[ok], y[ok], categorical_feature=cat, free_raw_data=False)
        m = lgb.train(params, ds, num_boost_round=600)
        Xp, _, mp = build(Wt, taxes, ufs, [Wt.index[-1]], False)
        s = np.abs(Wt.values[-12:]).mean(0) + 1e-6
        pred = m.predict(Xp) * s[mp[:, 0]]
        fc = pd.DataFrame({"unique_id": ids[mp[:, 0]], "h": mp[:, 1], "yhat": pred, "origin": o})
        fc["ds"] = [o + pd.DateOffset(months=int(h)) for h in fc.h]
        if deflate:
            fc["yhat"] = fc.yhat * fc.ds.map(expected_path(o, P, F, H, deflate))
        out.append(fc)
        print("lgbm", o.date(), X.shape, f"{time.time()-t0:.0f}s", flush=True)
    save_fc(pd.concat(out), "lgbm" + (f"_real_{deflate}" if deflate else ""))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
