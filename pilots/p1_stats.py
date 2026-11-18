"""P1 local statistical baselines: SNaive, AutoETS, AutoARIMA, AutoTheta (statsforecast)."""
import sys, time, os
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
import pandas as pd, numpy as np
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS, AutoARIMA, AutoTheta
from common import load_panel, ORIGINS, H, SEASON, save_fc

def main(which):
    df = load_panel()[["unique_id", "ds", "y"]]
    models = {"snaive": [SeasonalNaive(SEASON)],
              "ets": [AutoETS(season_length=SEASON)],
              "arima": [AutoARIMA(season_length=SEASON)],
              "theta": [AutoTheta(season_length=SEASON)]}[which]
    out = []
    t0 = time.time()
    for o in ORIGINS:
        tr = df[df.ds <= o]
        sf = StatsForecast(models=models, freq="MS", n_jobs=-1)
        fc = sf.forecast(df=tr, h=H)
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        fc = fc.rename(columns={col: "yhat"})
        fc["origin"] = o
        fc["h"] = fc.groupby("unique_id").cumcount() + 1
        out.append(fc)
        print(which, o.date(), f"{time.time()-t0:.0f}s", flush=True)
    save_fc(pd.concat(out), which)

if __name__ == "__main__":
    main(sys.argv[1])
