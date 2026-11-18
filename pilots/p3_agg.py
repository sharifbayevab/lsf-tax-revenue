"""P3a: base forecasts fitted directly on aggregate nodes (national total, national-by-tax, state totals).
Same real-terms protocol as p1b (deflate by IPCA, naive re-inflation) + Focus real-GDP anchor e=1.
usage: p3_agg.py chronos|theta   -> results/fc_agg_<model>.parquet (unique_id = node id)"""
import sys, os
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
import numpy as np, pandas as pd
from common import load_panel, ORIGINS, H, RES
from macro import ipca_index, focus, expected_path
from p1c_gdp_anchor import gdp_factor


def agg_panel(df):
    tot = df.groupby("ds", as_index=False).y.sum().assign(unique_id="TOT", level="total")
    tax = df.groupby(["tax", "ds"], as_index=False).y.sum()
    tax = tax.assign(unique_id="T|" + tax.tax, level="tax").drop(columns="tax")
    st = df.groupby(["UF", "ds"], as_index=False).y.sum()
    st = st.assign(unique_id="S|" + st.UF, level="state").drop(columns="UF")
    return pd.concat([tot, tax, st])[["unique_id", "level", "ds", "y"]]


def main(model):
    A = agg_panel(load_panel()); P = ipca_index(); FI = focus("IPCA"); FG = focus("PIB")
    A["yr"] = A.y / A.ds.map(P)
    if model == "chronos":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")
    else:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoTheta
    out = []
    for o in ORIGINS:
        tr = A[A.ds <= o]
        if model == "chronos":
            ps = []
            for _, g in tr.groupby("level"):
                p = pipe.predict_df(g[["unique_id", "ds", "yr"]], id_column="unique_id", timestamp_column="ds",
                                    target="yr", prediction_length=H, quantile_levels=[0.5],
                                    batch_size=g.unique_id.nunique(), cross_learning=True)
                ps.append(p.rename(columns={"0.5": "yhat"})[["unique_id", "ds", "yhat"]])
            p = pd.concat(ps)
        else:
            sf = StatsForecast(models=[AutoTheta(season_length=12)], freq="MS", n_jobs=-1)
            p = sf.forecast(df=tr[["unique_id", "ds", "yr"]].rename(columns={"yr": "y"}), h=H)
            p = p.rename(columns={"AutoTheta": "yhat"})
        p["yhat"] = p.yhat * p.ds.map(expected_path(o, P, FI, H, os.environ.get("PATHMODE", "naive"))) * p.ds.map(gdp_factor(o, FG, 1.0))
        p["origin"] = o; p["h"] = p.groupby("unique_id").cumcount() + 1
        out.append(p); print(model, o.date(), flush=True)
    fc = pd.concat(out)[["unique_id", "origin", "ds", "h", "yhat"]]
    fc.to_parquet(RES / f"fc_agg_{model}{os.environ.get('AGGSUF', '')}.parquet", index=False)
    print("saved", len(fc))


if __name__ == "__main__":
    main(sys.argv[1])
