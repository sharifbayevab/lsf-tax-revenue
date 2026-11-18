"""P6: leakage-free rolling empirical-quantile intervals with finite-sample split-conformal rank adjustment
(errors are serially dependent / overlapping, so nominal guarantees do NOT hold — coverage is reported empirically) for budget aggregates (total, national-by-tax, state).
For origin o, node level L and horizon h: past relative errors r = y/yhat - 1 of the same level and horizon whose
targets are observed by o (ds <= o), pooled over nodes of the level; interval yhat*(1+q_lo, 1+q_hi).
Origins without >= MIN past errors are skipped (reported). Metrics: coverage, mean relative width, Winkler score / y.
Review #5 (M9) variant 'infl': log errors log(y/yhat) divided by the expected cumulative log inflation from the origin
to the target month, log(Phat(T+h)/P(T)), taken from the same origin-dated price path as the forecast; ranks as above;
interval yhat*exp(q*s) with s of the current origin. Under unstable inflation errors scale with expected price growth.
usage: p6_conformal.py <fc_name> [<fc_name> ...]   (DATASET env selects country)"""
import sys, numpy as np, pandas as pd
from common import load_panel, RES, DATASET, H
PATH = {"br": "focus", "ar": "rem", "super": "rem"}.get(DATASET, "naive")


def infl_scale(L):
    from macro import ipca_index, focus, expected_path
    P = ipca_index(); F = focus("IPCA"); out = {}
    for o in L.origin.unique():
        e = expected_path(pd.Timestamp(o), P, F, H, PATH)
        for ds, v in e.items():
            out[(pd.Timestamp(o), ds)] = np.log(v / P.loc[pd.Timestamp(o)])
    return pd.Series([out[(o, d)] for o, d in zip(L.origin, L.ds)], index=L.index)

MIN = 8


def levels(fc, df):
    m = fc.merge(df[["unique_id", "ds", "y", "UF", "tax"]], on=["unique_id", "ds"])
    out = []
    for lvl, keys in [("total", []), ("tax", ["tax"]), ("state", ["UF"])]:
        g = m.groupby(keys + ["origin", "ds", "h"])[["y", "yhat"]].sum().reset_index()
        g["node"] = g[keys].astype(str).agg("|".join, axis=1) if keys else "TOT"
        out.append(g.assign(level=lvl)[["level", "node", "origin", "ds", "h", "y", "yhat"]])
    return pd.concat(out).reset_index(drop=True)


def conformal(L, alpha, scaled=False, aci=0.0, infl=None):
    """scaled: relative errors divided by the node's own past mean |r| (locally adaptive);
    aci: step size of adaptive conformal inference on the level's miscoverage (feedback only from realised targets)."""
    L = L.copy(); L["r"] = L.y / L.yhat - 1
    if infl is not None:
        L["s"] = infl.clip(lower=1e-4); L["r"] = np.log(L.y / L.yhat.where(L.yhat > 0)) / L.s   # log needs yhat > 0
    L["lo"] = np.nan; L["hi"] = np.nan
    a_t = {lvl: alpha for lvl in L.level.unique()}
    for o in sorted(L.origin.unique()):
        for lvl in L.level.unique():
            if aci > 0:   # update alpha with realised miscoverage of earlier intervals whose targets are now known
                done = L[(L.level == lvl) & (L.ds <= o) & (L.ds > o - pd.DateOffset(months=3)) & L.lo.notna()]
                if len(done):
                    miss = (~((done.y >= done.lo) & (done.y <= done.hi))).mean()
                    a_t[lvl] = float(np.clip(a_t[lvl] + aci * (alpha - miss), 0.001, 0.5))
            for h in range(1, 13):
                idx = L.index[(L.level == lvl) & (L.origin == o) & (L.h == h)]
                P = L[(L.level == lvl) & (L.h == h) & (L.ds <= o)]
                if len(P) < MIN or len(idx) == 0:
                    continue
                g = L.loc[idx]
                if scaled:
                    sP = P.groupby("node").r.apply(lambda x: x.abs().mean())
                    z = P.r / P.node.map(sP).replace(0, np.nan)
                    sc = g.node.map(sP).fillna(sP.median()).values
                else:
                    z = P.r; sc = 1.0
                zz = np.asarray(z.dropna()) if hasattr(z, "dropna") else np.asarray(z); n = len(zz)
                # finite-sample split-conformal ranks (equal-tailed): ceil((n+1)(1-a/2)) / n, clipped to [0,1]
                qhi = min(1.0, np.ceil((n + 1) * (1 - a_t[lvl] / 2)) / n); qlo = max(0.0, 1 - qhi)
                ql, qh = np.quantile(zz, qlo, method="lower"), np.quantile(zz, qhi, method="higher")
                if infl is not None:
                    L.loc[idx, "lo"] = g.yhat * np.exp(ql * g.s); L.loc[idx, "hi"] = g.yhat * np.exp(qh * g.s)
                else:
                    L.loc[idx, "lo"] = g.yhat * (1 + ql * sc); L.loc[idx, "hi"] = g.yhat * (1 + qh * sc)
    return L


def report(name):
    df = load_panel(); fc = pd.read_parquet(RES / f"fc_{name}.parquet"); L = levels(fc, df)
    rows = []
    L = L[L.y > 0]   # zero-receipt nodes (reporting gaps) excluded from interval scoring
    S = infl_scale(L)
    for a, variant in [(x, v) for x in (0.2, 0.05) for v in ("pooled", "scaled", "scaled+aci", "infl")]:
        C = conformal(L, a, scaled=variant in ("scaled", "scaled+aci"), aci=0.05 if "aci" in variant else 0,
                      infl=S if variant == "infl" else None).dropna(subset=["lo"])
        C["cov"] = (C.y >= C.lo) & (C.y <= C.hi)
        C["w"] = (C.hi - C.lo) / C.y
        pen = (2 / a) * ((C.lo - C.y).clip(lower=0) + (C.y - C.hi).clip(lower=0))
        C["wink"] = ((C.hi - C.lo) + pen) / C.y
        for lvl, g in C.groupby("level"):
            rows.append({"model": name, "variant": variant, "nominal": 1 - a, "level": lvl, "coverage": g["cov"].mean(),
                         "rel_width": g.w.median(), "winkler_rel": g.wink.mean(), "n_origins": g.origin.nunique(),
                         "n_cells": len(g)})
            if lvl == "total":
                for y, gy in g.groupby(g.origin.dt.year):
                    rows.append({"model": name, "variant": variant, "nominal": 1 - a, "level": f"total_{y}",
                                 "coverage": gy["cov"].mean(), "rel_width": gy.w.median(), "n_cells": len(gy)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = pd.concat([report(n) for n in sys.argv[1:]])
    print(DATASET); print(t.round(3).to_string(index=False))
    t.to_csv(RES / "p6_conformal.csv", index=False)
