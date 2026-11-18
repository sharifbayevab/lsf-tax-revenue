"""P4b: buoyancy-gap correction (Brazil). At origin o, gap = expanding mean over complete years Y <= last of
(real growth of national kept-panel receipts in Y) - (real GDP growth in Y); GDP for year Y is treated as known from
March of Y+1 (IBGE first release is early March). Forecast multiplied by (1+gap)^(h/12).
CAVEAT: uses the current (revised) BCB SGS 7326 series, not first-release vintages -> quasi-real-time sensitivity.
usage: p4b_buoyancy.py <fc_name> [...]  -> results/fc_<fc_name>_buoy.parquet"""
import sys, numpy as np, pandas as pd
from common import load_panel, RES, save_fc
from macro import ipca_index
from p4_oracle import realised_gdp


def gap_series(df, P, G, origin):
    # receipts for complete years up to the origin only
    d = df[df.ds <= origin]
    t = d.groupby("ds").y.sum() / P.reindex(sorted(d.ds.unique())).values
    a = t.groupby(t.index.year).agg(["sum", "size"]); a = a[a["size"] == 12]["sum"]
    return (a.pct_change() - G.reindex(a.index)).dropna()


def main(name):
    df = load_panel(); P = ipca_index(); G = realised_gdp()
    fc = pd.read_parquet(RES / f"fc_{name}.parquet"); parts, log = [], []
    for o, g in fc.groupby("origin"):
        last = o.year - 1 if o.month >= 3 else o.year - 2
        gap = gap_series(df, P, G, o); gap = gap[gap.index <= last]
        b = gap.mean() if len(gap) else 0.0
        log.append((str(o.date()), round(b * 100, 2)))
        parts.append(g.assign(yhat=g.yhat * (1 + b) ** (g.h / 12)))
    print(name, "gap % by origin (first/last):", log[:2], log[-2:])
    save_fc(pd.concat(parts), f"{name}_buoy")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
