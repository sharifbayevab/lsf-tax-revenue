"""P4: decomposition of budget-level error into macro surprises vs model error (Brazil).
Takes a forecast made with naive inflation path + Focus GDP anchor (e=1) and swaps in ex-post realised paths:
  infl  : realised IPCA instead of the naive trailing-12m path
  gdp   : realised annual real GDP growth (BCB SGS 7326) instead of the Focus vintage (2026 targets keep Focus)
  both  : both swaps.  NOT a forecast — an upper bound showing how much error is macro surprise.
usage: p4_oracle.py <fc_name>"""
import sys, json, numpy as np, pandas as pd
from common import load_panel, RES, H, ROOT, ORIGINS
from macro import ipca_index, focus, expected_path
from p1c_gdp_anchor import gdp_factor


def realised_gdp():
    d = pd.DataFrame(json.load(open(ROOT / "data/raw/gdp_real_7326.json")))
    return pd.Series(d.valor.astype(float).values / 100, index=pd.to_datetime(d.data, dayfirst=True).dt.year)


def realised_factor(o, FG, G):
    """Same construction as gdp_factor but with realised growth where available."""
    cut = o + pd.offsets.MonthEnd(0)
    v = FG[FG.Data <= cut]; v = v[v.Data == v.Data.max()].set_index("year").Mediana / 100
    months = pd.date_range(o + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    f, cur = [], 1.0
    for m in months:
        g = G.get(m.year, v.get(m.year, v.iloc[-1]))
        cur *= (1 + g) ** (1 / 12); f.append(cur)
    return pd.Series(f, index=months)


def scores(fc, df):
    m = fc.merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    t = m.groupby(["origin", "ds"])[["y", "yhat"]].sum()
    wape = (t.y - t.yhat).abs().sum() / t.y.sum()
    a = t.groupby("origin").sum(); full = t.groupby("origin").size() == 12
    ape = (a.yhat / a.y - 1)[full]
    bias = (a.yhat.sum() / a.y.sum() - 1)
    return wape, ape.abs().mean(), ape.abs().median(), bias


def main(name):
    df = load_panel(); P = ipca_index(); FI = focus("IPCA"); FG = focus("PIB"); G = realised_gdp()
    fc = pd.read_parquet(RES / f"fc_{name}.parquet")
    adj_i, adj_g = [], []
    for o in ORIGINS:
        months = pd.date_range(o + pd.offsets.MonthBegin(1), periods=H, freq="MS")
        e = expected_path(o, P, FI, H, "naive")
        real = P.reindex(months)
        adj_i.append(pd.DataFrame({"origin": o, "ds": months, "ai": (real / e).values}))
        adj_g.append(pd.DataFrame({"origin": o, "ds": months,
                                   "ag": (realised_factor(o, FG, G) / gdp_factor(o, FG, 1.0)).values}))
    A = pd.concat(adj_i).merge(pd.concat(adj_g), on=["origin", "ds"])
    A["ai"] = A.ai.fillna(1.0)  # months beyond available IPCA: no swap
    f = fc.merge(A, on=["origin", "ds"])
    rows = {}
    for k, mult in {"base": 1, "infl": f.ai, "gdp": f.ag, "both": f.ai * f.ag}.items():
        g = f.assign(yhat=f.yhat * mult)
        rows[k] = dict(zip(["WAPE_total_monthly", "APE_12m_mean", "APE_12m_median", "bias"], scores(g, df)))
    t = pd.DataFrame(rows).T.round(4); print(name); print(t.to_string())
    return t


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
