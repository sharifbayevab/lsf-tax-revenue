"""P3b: two-axis (state x tax) hierarchical reconciliation of base forecasts.
Nodes: TOT, T|<tax> (19), S|<UF> (27), bottom UF|tax (472).
Base: bottom file (default chronos_real_naive_gdp1) + aggregate file fc_agg_<model>.
Methods (all leakage-free: weights use only errors whose targets are observed by the origin):
  bu       bottom-up
  ols      OLS projection
  wls_s    structural WLS (w = #bottom series under node)
  wls_v    WLS with node variance = mean squared past realised base error (fallback wls_s)
  mint     MinT with shrinkage covariance of past realised scaled errors (fallback wls_v)
usage: p3_reconcile.py <bottom_name> <agg_model>   -> results/fc_rec_<agg>_<method>.parquet + table
"""
import sys, numpy as np, pandas as pd
from common import load_panel, ORIGINS, H, RES, evaluate, mase_scales, save_fc
from p3_agg import agg_panel


def summing(df):
    b = sorted(df.unique_id.unique())
    meta = df[["unique_id", "UF", "tax"]].drop_duplicates().set_index("unique_id").loc[b]
    taxes, ufs = sorted(meta.tax.unique()), sorted(meta.UF.unique())
    rows = [np.ones(len(b))]
    rows += [(meta.tax.values == t).astype(float) for t in taxes]
    rows += [(meta.UF.values == u).astype(float) for u in ufs]
    names = ["TOT"] + ["T|" + t for t in taxes] + ["S|" + u for u in ufs]
    S = np.vstack(rows + [np.eye(len(b))])
    return S, names + b, b


def shrink_cov(E):
    """Schafer-Strimmer shrinkage towards diagonal; E: (n_obs, n_nodes) errors."""
    n = E.shape[0]
    Ec = E - E.mean(0)
    sd = np.sqrt((Ec ** 2).mean(0)) + 1e-12
    Z = Ec / sd
    R = (Z.T @ Z) / n
    W = np.einsum("ki,kj->kij", Z, Z)
    varR = W.var(0) * n / (n - 1) ** 2 if n > 1 else np.ones_like(R)
    off = ~np.eye(len(sd), dtype=bool)
    lam = np.clip(varR[off].sum() / (R[off] ** 2).sum(), 0, 1)
    Rs = (1 - lam) * R; np.fill_diagonal(Rs, 1.0)
    return Rs * np.outer(sd, sd), lam


def rake(Yb, S, nt, ns, Ttax, Tst, it=50):
    """Iterative proportional fitting of the bottom (state x tax) to tax and/or state margins, per horizon."""
    Mt, Ms = S[1:nt], S[nt:nt + ns]
    X = Yb.copy() + 1e-9
    for _ in range(it if (Ttax is not None and Tst is not None) else 1):
        if Ttax is not None:
            X *= (Mt.T @ (Ttax / (Mt @ X)))
        if Tst is not None:
            X *= (Ms.T @ (Tst / (Ms @ X)))
    return X


def project(S, Yh, Winv=None, Sigma=None):
    """Yh: (n_all, H). Returns reconciled bottom (n_b, H)."""
    if Sigma is not None:
        Si = np.linalg.pinv(Sigma)
        G = np.linalg.solve(S.T @ Si @ S, S.T @ Si)
    else:
        G = np.linalg.solve((S.T * Winv) @ S, S.T * Winv)
    return G @ Yh


def main(bottom_name, agg_model):
    df = load_panel()
    S, nodes, b = summing(df)
    nb = len(b)
    n_tax = sum(n.startswith("T|") for n in nodes); n_st = sum(n.startswith("S|") for n in nodes)
    A = agg_panel(df)
    truth = pd.concat([A[["unique_id", "ds", "y"]], df[["unique_id", "ds", "y"]]])
    truth = truth.pivot(index="unique_id", columns="ds", values="y").loc[nodes]
    fb = pd.read_parquet(RES / f"fc_{bottom_name}.parquet")
    fa = pd.read_parquet(RES / f"fc_agg_{agg_model}.parquet")
    base = pd.concat([fa, fb])
    ws = S.sum(1)  # structural weights
    hist = []      # list of (origin, target ds, node-error vector, node-scale)
    outs = {m: [] for m in ["bu", "ols", "wls_s", "wls_v", "wls_p", "wls_pv", "mint", "scale_tot", "rake", "rake_tax", "rake_state", "rake_sel"]}
    hist_sel = []  # (target ds, |err direct| per tax, |err BU| per tax)
    lam_log = []; sel_log = []
    for o in ORIGINS:
        g = base[base.origin == o].pivot(index="unique_id", columns="ds", values="yhat").loc[nodes]
        months = g.columns
        Yh = g.values
        # realised past errors available at origin o
        past = [e for e in hist if e[1] <= o]
        rec = {}
        rec["bu"] = Yh[-nb:]
        Yb0 = np.clip(Yh[-nb:], 0, None)
        nt, ns = 1 + n_tax, n_st
        Ttot = Yh[0]; Ttax = Yh[1:nt]; Tst = Yh[nt:nt + ns]
        rec["scale_tot"] = Yh[-nb:] * (Ttot / (S[0] @ Yh[-nb:]))[None, :]
        rec["rake"] = rake(Yb0, S, nt, ns, Ttax * (Ttot / Ttax.sum(0)), Tst * (Ttot / Tst.sum(0)))
        rec["rake_tax"] = rake(Yb0, S, nt, ns, Ttax, None)
        # adaptive rule: per tax, rake to the direct national forecast only if its past realised error is lower
        Mt = S[1:nt]; BUtax = Mt @ Yh[-nb:]
        ps = [e for e in hist_sel if e[0] <= o]
        if len(ps) >= 6:
            use = np.vstack([e[1] for e in ps]).mean(0) < np.vstack([e[2] for e in ps]).mean(0)
        else:
            use = np.zeros(nt - 1, dtype=bool)
        sel_log.append((o, int(use.sum())))
        rec["rake_sel"] = rake(Yb0, S, nt, ns, np.where(use[:, None], Ttax, BUtax), None)
        rec["rake_state"] = rake(Yb0, S, nt, ns, None, Tst)
        rec["ols"] = project(S, Yh, Winv=np.ones(len(nodes)))
        rec["wls_s"] = project(S, Yh, Winv=1 / ws)
        lvl = np.abs(Yh).mean(1) + 1e-9          # current base level per node
        rec["wls_p"] = project(S, Yh, Winv=1 / lvl ** 2)   # proportional: equal relative error variance
        if len(past) >= 12:
            E = np.vstack([e[2] for e in past])
            v = (E ** 2).mean(0) + 1e-9
            rec["wls_v"] = project(S, Yh, Winv=1 / v)
            rv = ((E / np.vstack([e[3] for e in past])) ** 2).mean(0) + 1e-9   # past relative error variance
            rec["wls_pv"] = project(S, Yh, Winv=1 / (rv * lvl ** 2))
            # MinT on errors scaled by node level (heteroskedastic over time) -> rescale to current level
            Lv = np.vstack([e[3] for e in past])
            Es = E / Lv
            C, lam = shrink_cov(Es)
            cur = np.abs(Yh).mean(1) + 1e-9
            rec["mint"] = project(S, Yh, Sigma=C * np.outer(cur, cur)); lam_log.append((o, lam, len(past)))
        else:
            rec["wls_v"] = rec["wls_s"]; rec["mint"] = rec["wls_s"]; rec["wls_pv"] = rec["wls_p"]
        for m, Yb in rec.items():
            outs[m].append(pd.DataFrame({"unique_id": np.repeat(b, H), "ds": np.tile(months, nb),
                                         "h": np.tile(np.arange(1, H + 1), nb), "yhat": Yb.ravel(), "origin": o}))
        # store base errors for targets of this origin (become usable once ds <= later origin)
        for j, d in enumerate(months):
            if d in truth.columns:
                y = truth[d].values
                hist_sel.append((d, np.abs(y[1:nt] - Yh[1:nt, j]), np.abs(y[1:nt] - BUtax[:, j])))
                hist.append((o, d, y - Yh[:, j], np.abs(truth.loc[:, :o].iloc[:, -12:]).mean(1).values + 1e-9))
    print("mint shrinkage (origin, lambda, n):", [(str(a.date()), round(l, 2), n) for a, l, n in lam_log[:3]], "...")
    print("rake_sel: #taxes raked per origin:", [n for _, n in sel_log])
    sc = mase_scales(df)
    rows = {}
    for m, parts in outs.items():
        fc = pd.concat(parts)
        name = f"rec_{bottom_name.split('_')[0]}_{agg_model}_{m}" if m != "bu" else f"rec_bu_{bottom_name}"
        save_fc(fc, name)
        r, _ = evaluate(fc, df, sc, by_h=False); rows[name] = r
    # direct (unreconciled) aggregate accuracy
    m = fa.merge(A[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    m["lvl"] = m.unique_id.str[:2].map({"TO": "total", "T|": "tax", "S|": "state"})
    d = m.assign(ae=(m.y - m.yhat).abs()).groupby("lvl")[["ae", "y"]].sum()
    print(f"direct {agg_model} aggregate WAPE:", (d.ae / d.y.abs()).round(4).to_dict())
    t = pd.DataFrame(rows).T[["MASE", "MASE_med", "WAPE_bottom", "WAPE_tax", "WAPE_state", "WAPE_total"]]
    print(t.round(4).to_string())
    t.to_csv(RES / f"p3_table_{agg_model}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "chronos_real_naive_gdp1",
         sys.argv[2] if len(sys.argv) > 2 else "theta")
