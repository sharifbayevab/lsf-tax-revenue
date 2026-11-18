"""P1b: forecast deflated receipts, re-inflate with an expected price path.
usage: p1b_real.py chronos|theta  focus|naive
Review #5 (M8): env GROUP=tax (default, cross-learning within a tax) | indep (no cross-learning) | region (groups =
regions) | random (fixed random groups of ~100 series) | taxre (default grouping re-run, for timing); non-default groups are saved with suffix _g<GROUP> and the
wall-clock time is appended to results/<ds>/timing_attention.csv."""
import sys, os, time
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
import numpy as np, pandas as pd
from common import load_panel, ORIGINS, H, save_fc, RES
from macro import ipca_index, focus, expected_path

def main(model, mode):
    df = load_panel(); P = ipca_index(); F = focus("IPCA")
    df["yr"] = df.y / df.ds.map(P)
    if model == "chronos":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")
    else:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoTheta
    GROUP = os.environ.get("GROUP", "tax")
    if GROUP == "random":
        ids = df.unique_id.unique().copy(); np.random.default_rng(0).shuffle(ids)
        gmap = {u: k // 100 for k, u in enumerate(ids)}
    else:
        gmap = None
    gcol = {"tax": "tax", "taxre": "tax", "region": "UF", "indep": "tax", "random": "rg"}[GROUP]  # taxre = re-run of the default for timing
    df["rg"] = df.unique_id.map(gmap) if gmap else 0
    out = []; t0 = time.time()
    for o in ORIGINS:
        tr = df[df.ds <= o][["unique_id", "tax", "UF", "rg", "ds", "yr"]]
        if model == "chronos":
            ps = []
            for _, g in tr.groupby(gcol):
                p = pipe.predict_df(g[["unique_id", "ds", "yr"]], id_column="unique_id", timestamp_column="ds",
                                    target="yr", prediction_length=H, quantile_levels=[0.5],
                                    batch_size=len(g.unique_id.unique()), cross_learning=(GROUP != "indep"))
                ps.append(p.rename(columns={"0.5": "yhat"})[["unique_id", "ds", "yhat"]])
            p = pd.concat(ps)
        else:
            from statsforecast.models import SeasonalNaive
            sf = StatsForecast(models=[AutoTheta(season_length=12)], freq="MS", n_jobs=-1,
                               fallback_model=SeasonalNaive(season_length=12))
            p = sf.forecast(df=tr[["unique_id", "ds", "yr"]].rename(columns={"yr": "y"}), h=H).rename(columns={"AutoTheta": "yhat"})
        path = expected_path(o, P, F, H, mode)
        p["yhat"] = p.yhat * p.ds.map(path)
        p["origin"] = o; p["h"] = p.groupby("unique_id").cumcount() + 1
        out.append(p)
        print(model, mode, o.date(), flush=True)
    suf = "" if GROUP == "tax" else f"_g{GROUP}"
    save_fc(pd.concat(out), f"{model}_real_{mode}{suf}")  # mode in naive, naive3, focus, rem
    if model == "chronos":
        with open(RES / "timing_attention.csv", "a") as fh:
            fh.write(f"{mode},{GROUP},{time.time()-t0:.1f},{len(ORIGINS)}\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
