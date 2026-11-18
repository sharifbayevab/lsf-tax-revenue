"""P20 (review #5, M2): accuracy on target months that no pretraining corpus can contain.
Target months from 2025-11 onward, after the public release of the Chronos-2 checkpoint (repository created
2025-10-30; TiRex and TimesFM-2.5 were released earlier in 2025). Same saved forecasts as in the paper, all origins whose
horizon reaches these months. Brazil: level layer with naive path x growth anchor (the multi-model comparison of
Table 2, full-sample filter). Supermarkets: REM level layer, plus nominal models.
usage: python p20_postrelease.py -> results/review5/p20_postrelease.csv"""
import os, importlib
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)
CUT = pd.Timestamp("2025-11-01")
SETS = {"br": ("Brazil", None, [("Chronos-2 (LSF)", "chronos_real_naive_gdp1"), ("TiRex", "tirex_real_naive_gdp1"),
                                ("TimesFM-2.5", "timesfm_real_naive_gdp1"), ("LightGBM", "lgbm_real_naive_gdp1"),
                                ("LSTM, tuned", "LSTM_tuned_real_naive_gdp1"), ("N-HiTS, tuned", "NHITS_tuned_real_naive_gdp1"),
                                ("AutoTheta", "theta_real_naive_gdp1")]),
        "super": ("Supermarkets", None, [("Chronos-2 (LSF)", "chronos_real_rem_gdp1"), ("AutoTheta + level", "theta_real_rem_gdp1"),
                                         ("LightGBM + level", "lgbm_real_rem_gdp1"), ("Chronos-2, nominal", "c2_xl_tax"),
                                         ("AutoTheta, nominal", "theta"), ("LightGBM, nominal", "lgbm"), ("AutoARIMA", "arima")])}


def env(d, kept):
    os.environ["DATASET"] = d
    if kept: os.environ["KEPT"] = kept
    else: os.environ.pop("KEPT", None)
    import common; importlib.reload(common)
    return common


def main():
    rows = []
    for d, (pname, kept, models) in SETS.items():
        c = env(d, kept); df = c.load_panel()
        ref = None
        for lab, n in models:
            p = c.RES / f"fc_{n}.parquet"
            if not p.exists():
                print("missing", d, n); continue
            m = pd.read_parquet(p).merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
            for win, mm in [("post-release", m[m.ds >= CUT]), ("all", m)]:
                t = mm.groupby(["origin", "ds"])[["y", "yhat"]].sum()
                bot = mm.assign(ae=(mm.y - mm.yhat).abs()).groupby("ds")[["ae"]].sum()
                rows.append(dict(panel=pname, model=lab, window=win, months=mm.ds.nunique(), origins=mm.origin.nunique(),
                                 wape_bottom=100 * (mm.y - mm.yhat).abs().sum() / mm.y.abs().sum(),
                                 wape_total=100 * (t.y - t.yhat).abs().sum() / t.y.abs().sum()))
            post = m[m.ds >= CUT].assign(ae=lambda x: (x.y - x.yhat).abs()).groupby("ds").ae.sum()
            if ref is None: ref = post
            else: rows[-2]["c2_wins_months"] = f"{int((ref < post.reindex(ref.index)).sum())}/{len(ref)}"
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "p20_postrelease.csv"), index=False)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
