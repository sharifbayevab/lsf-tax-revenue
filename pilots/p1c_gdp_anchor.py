"""P1c: post-hoc real-growth anchor. yhat * prod over months of (1+g_Y)^(1/12), g_Y = Focus real GDP
expectation for the target year (vintage <= end of origin month); elasticity e in {0.5,1}.
Applied to an existing real-terms forecast file."""
import sys, numpy as np, pandas as pd
from common import RES, ORIGINS, H, save_fc
from macro import focus

def gdp_factor(origin, F, e=1.0):
    if F is None:  # no GDP expectations for this dataset -> no anchor
        return pd.Series(1.0, index=pd.date_range(origin + pd.offsets.MonthBegin(1), periods=H, freq="MS"))
    cut = origin + pd.offsets.MonthEnd(0)
    v = F[F.Data <= cut]; v = v[v.Data == v.Data.max()].set_index("year").Mediana / 100
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    f, cur = [], 1.0
    for m in months:
        cur *= (1 + e * v.get(m.year, v.iloc[-1])) ** (1 / 12); f.append(cur)
    return pd.Series(f, index=months)

if __name__ == "__main__":
    src, e = sys.argv[1], float(sys.argv[2])
    F = focus("PIB"); fc = pd.read_parquet(RES / f"fc_{src}.parquet")
    parts = []
    for o, g in fc.groupby("origin"):
        g = g.copy(); g["yhat"] = g.yhat * g.ds.map(gdp_factor(o, F, e)); parts.append(g)
    save_fc(pd.concat(parts), f"{src}_gdp{e:g}")
