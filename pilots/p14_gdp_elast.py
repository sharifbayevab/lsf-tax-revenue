"""P14 (review #3, K1): GDP-growth multiplier elasticity readout, e in {0, 0.5, 1}, on SAVED forecasts (no refit).
Real-terms forecasts re-inflated with the survey price path (files fc_<model>_real_<path>) are multiplied by
gdp_factor(origin, F, e). For Brazil the full pipeline (rake to national-by-tax, + buoyancy gap) is divided by the
e=1 factor it already contains and multiplied by the e factor (a common multiplier commutes with raking).
Note: the BR buoyancy gap is defined relative to GDP; with e=0 it is kept as is (sensitivity, not re-derived).
usage: DATASET=br|ar|super KEPT=... python p14_gdp_elast.py  -> results/<ds>/p14_gdp_elast.csv"""
import numpy as np, pandas as pd
from common import load_panel, RES, DATASET, evaluate, mase_scales
from macro import focus
from p1c_gdp_anchor import gdp_factor

PATH = {"br": "focus", "ar": "rem", "super": "rem"}[DATASET]
ES = [0.0, 0.5, 1.0]


def apply(fc, F, e, base_e=0.0):
    parts = []
    for o, g in fc.groupby("origin"):
        f = gdp_factor(o, F, e) / gdp_factor(o, F, base_e)
        parts.append(g.assign(yhat=g.yhat * g.ds.map(f).values))
    return pd.concat(parts)


def ape12(m):
    tot = m.groupby(["origin", "ds"])[["y", "yhat"]].sum().reset_index()
    a = tot.groupby("origin").agg(y=("y", "sum"), yhat=("yhat", "sum"), n=("ds", "size"))
    a = a[a.n == 12]
    return ((a.yhat - a.y).abs() / a.y).mean(), ((a.yhat - a.y) / a.y).mean()


def main():
    df = load_panel(); sc = mase_scales(df); F = focus("PIB")
    runs = {f"{m}_real_{PATH}": 0.0 for m in ["chronos", "theta", "lgbm", "tirex", "timesfm"]}
    if DATASET == "br":
        runs["rec_chronos_chronos_focus_rake_tax"] = 1.0
        runs["rec_chronos_chronos_focus_rake_tax_buoy"] = 1.0
    rows = []
    for name, base_e in runs.items():
        p = RES / f"fc_{name}.parquet"
        if not p.exists():
            continue
        fc = pd.read_parquet(p)
        for e in ES:
            out, m = evaluate(apply(fc, F, e, base_e), df, sc, by_h=False)
            a, b = ape12(m)
            rows.append(dict(panel=DATASET, model=name, e=e, WAPE_bottom=out["WAPE_bottom"],
                             WAPE_total=out["WAPE_total"], APE12=a, bias12=b, n=out["n"]))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "p14_gdp_elast.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
