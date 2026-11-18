"""P19 (review #5, M1): an economic rule with no fitted model. The deflated value of the same month one year earlier
(seasonal naive on real receipts) times the same origin-dated expected price path and, where a growth survey exists,
the unit-elasticity growth anchor. Saved as fc_snaive_real_<path>[_gdp1] so that it is scored like every other model.
usage: python p19_econ_baseline.py -> results/<ds>/fc_snaive_real_*.parquet, results/review5/p19_econ_baseline.csv"""
import os, importlib
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)
RUNS = [("br", None, "naive", True), ("br", "kept_rt", "focus", True), ("ar", None, "rem", True),
        ("super", None, "rem", True), ("es", None, "naive", False), ("pe", None, "naive", False)]


def env(d, kept):
    os.environ["DATASET"] = d
    if kept: os.environ["KEPT"] = kept
    else: os.environ.pop("KEPT", None)
    import common, macro, p1c_gdp_anchor
    for m in (common, macro, p1c_gdp_anchor): importlib.reload(m)
    return common, macro, p1c_gdp_anchor


def main():
    rows = []
    for d, kept, path, gdp in RUNS:
        c, mc, ga = env(d, kept); df = c.load_panel(); P = mc.ipca_index(); F = mc.focus("IPCA")
        FG = mc.focus("PIB") if gdp else None
        df["yr"] = df.y / df.ds.map(P); W = df.pivot(index="ds", columns="unique_id", values="yr")
        out = []
        for o in c.ORIGINS:
            e = mc.expected_path(o, P, F, c.H, path)
            g = ga.gdp_factor(o, FG, 1.0) if gdp else pd.Series(1.0, index=e.index)
            for h, ds in enumerate(e.index, start=1):
                base = W.loc[ds - pd.DateOffset(years=1 * ((h - 1) // 12 + 1))]
                out.append(pd.DataFrame({"unique_id": base.index, "origin": o, "ds": ds, "h": h,
                                         "yhat": base.values * e[ds] * g[ds]}))
        name = f"snaive_real_{path}" + ("_gdp1" if gdp else "")
        fc = pd.concat(out); c.save_fc(fc, name)
        r, _ = c.evaluate(fc, df, None, by_h=False)
        t = fc.merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"]).groupby("origin")[["y", "yhat"]].sum()
        rows.append(dict(panel=d, kept=kept or "kept", fc=name, wape_bottom=100 * r["WAPE_bottom"],
                         wape_total=100 * r["WAPE_total"], ape12=100 * (t.yhat / t.y - 1).abs().mean(),
                         bias12=100 * (t.yhat / t.y - 1).mean()))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "p19_econ_baseline.csv"), index=False)


if __name__ == "__main__":
    main()
