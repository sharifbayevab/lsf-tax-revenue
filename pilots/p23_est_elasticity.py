"""P23 (review #5, M7): fixed unit elasticity of receipts to expected real GDP growth vs an elasticity estimated from
pre-origin history (Brazil; Argentina has three years of history before the first origin).

At origin o: annual real growth of panel receipts (deflated by IPCA, complete years only) regressed on realised real GDP
growth (BCB SGS 7326) with an intercept, over years Y >= 2006 whose GDP is published by o (first release in March of
Y+1, the rule of p4b_buoyancy.py; current revised series, hence quasi-real-time). Pooled panel and four tax groups
(income, turnover, import, other); 'shrunk' moves the group slopes halfway to one. The slope replaces e in the growth
anchor of p1c_gdp_anchor.py applied to the saved e=0 forecasts.
usage: python p23_est_elasticity.py -> results/review5/p23_elasticity.csv, results/review5/p23_slopes.csv"""
import os, importlib
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)
os.environ["DATASET"] = "br"; os.environ["KEPT"] = "kept_rt"
import common, macro, p1c_gdp_anchor, p4_oracle
for m in (common, macro, p1c_gdp_anchor, p4_oracle): importlib.reload(m)

GROUPS = {"income": ["IRPF", "IRPJ_DEM", "IRPJ_FIN", "IRRF_CAP", "IRRF_OUT", "IRRF_REM", "IRRF_TRAB", "CSLL_DEM", "CSLL_FIN"],
          "turnover": ["COFINS_DEM", "COFINS_FIN", "PIS_DEM", "PIS_FIN"],
          "import": ["II", "IPI_IMP"], "other": ["CIDE", "IOF", "IPI_OUT", "OUTRAS"]}
TAX2G = {t: g for g, ts in GROUPS.items() for t in ts}


def slope(real_annual, G, last):
    a = real_annual[(real_annual.index <= last)]
    gr = a.pct_change().dropna(); gr = gr[gr.index >= 2006]
    x = G.reindex(gr.index); ok = x.notna()
    X = np.column_stack([np.ones(ok.sum()), x[ok].values])
    return float(np.linalg.lstsq(X, gr[ok].values, rcond=None)[0][1])


def main():
    df = common.load_panel(); P = macro.ipca_index(); G = p4_oracle.realised_gdp(); F = macro.focus("PIB")
    df["g"] = df.tax.map(TAX2G); assert df.g.notna().all()
    df["yr"] = df.y / df.ds.map(P)
    full = df.groupby("ds").filter(lambda x: True)
    months_per_year = df.groupby(df.ds.dt.year).ds.nunique()
    complete = months_per_year[months_per_year == 12].index
    ann = df[df.ds.dt.year.isin(complete)].groupby([df.ds.dt.year, "g"]).yr.sum().unstack()
    ann_all = ann.sum(1)
    act = df[["unique_id", "ds", "y"]]; meta = df[["unique_id", "g"]].drop_duplicates().set_index("unique_id").g
    res, slopes = [], []
    for model in ["chronos", "theta"]:
        f0 = pd.read_parquet(common.RES / f"fc_{model}_real_focus.parquet")
        out = {k: [] for k in ["e0", "e1", "pooled", "group", "group_shrunk"]}
        for o, g in f0.groupby("origin"):
            last = o.year - 1 if o.month >= 3 else o.year - 2
            bp = slope(ann_all, G, last)
            bg = {k: slope(ann[k], G, last) for k in GROUPS}
            if model == "chronos":
                slopes.append(dict(origin=o, pooled=bp, **bg))
            fac = lambda e: g.ds.map(p1c_gdp_anchor.gdp_factor(o, F, e)).values
            out["e0"].append(g)
            out["e1"].append(g.assign(yhat=g.yhat * fac(1.0)))
            out["pooled"].append(g.assign(yhat=g.yhat * fac(bp)))
            gg = g.unique_id.map(meta)
            for lab, sh in [("group", 0.0), ("group_shrunk", 0.5)]:
                y = g.yhat.values.copy()
                for k, b in bg.items():
                    msk = (gg == k).values
                    y[msk] = y[msk] * fac((1 - sh) * b + sh * 1.0)[msk]
                out[lab].append(g.assign(yhat=y))
        for lab, parts in out.items():
            fc = pd.concat(parts).merge(act, on=["unique_id", "ds"])
            t = fc.groupby(["origin", "ds"])[["y", "yhat"]].sum()
            s12 = t.groupby("origin")[["y", "yhat"]].sum()
            res.append(dict(model=model, variant=lab, wape_total=100 * (t.y - t.yhat).abs().sum() / t.y.abs().sum(),
                            wape_bottom=100 * (fc.y - fc.yhat).abs().sum() / fc.y.abs().sum(),
                            bias12=100 * (s12.yhat / s12.y - 1).mean()))
    r = pd.DataFrame(res); s = pd.DataFrame(slopes)
    r.to_csv(os.path.join(OUT, "p23_elasticity.csv"), index=False); s.to_csv(os.path.join(OUT, "p23_slopes.csv"), index=False)
    print(r.round(2).to_string(index=False)); print(s.iloc[[0, -1]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
