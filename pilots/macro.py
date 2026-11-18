"""Real-time price index and Focus-based expected price paths (no look-ahead).
At origin T (last observed receipts month) we assume IPCA up to T is known and use the
latest Focus vintage dated <= last day of month T."""
import json, numpy as np, pandas as pd
from common import ROOT, DATASET

RAW = ROOT / "data/raw"

def ipca_index():
    if DATASET == "es":   # INE table 24077, general index (2021 = 100)
        d = pd.read_csv(RAW / "ext/spain_cpi.csv", sep=";", encoding="utf-8-sig")
        d = d[d.iloc[:, 1].str.strip() == "Índice"]
        P = pd.Series(d.Total.str.replace(",", ".").astype(float).values,
                      index=pd.to_datetime(d.Periodo.str.replace("M", "-") + "-01")).sort_index().dropna()
        return P / P.loc["2006-12-01"]
    if DATASET == "pe":   # BCRP/INEI Lima CPI (Dec 2021 = 100)
        mm = dict(Ene=1, Feb=2, Mar=3, Abr=4, May=5, Jun=6, Jul=7, Ago=8, Sep=9, Set=9, Oct=10, Nov=11, Dic=12)
        d = json.load(open(RAW / "ext/peru_cpi.json"))["periods"]
        idx = [pd.Timestamp(int(x["name"].split(".")[1]), mm[x["name"].split(".")[0]], 1) for x in d]
        P = pd.Series(pd.to_numeric([x["values"][0] for x in d], errors="coerce"), index=idx).sort_index().dropna()
        return P / P.iloc[0]
    if DATASET in ("ar", "super"):   # INDEC national CPI (datos.gob.ar 148.3_INIVELNAL_DICI_M_26), base Dec-2016
        d = pd.DataFrame(json.load(open(RAW / "ext/ar_ipc_indec.json"))["data"], columns=["ds", "v"])
        P = d.assign(ds=pd.to_datetime(d.ds)).set_index("ds").v.astype(float)
        return P / P.loc["2016-12-01"]
    d = pd.DataFrame(json.load(open(RAW / "ipca_433.json")))
    d["ds"] = pd.to_datetime(d.data, dayfirst=True)
    d["r"] = d.valor.astype(float) / 100
    P = (1 + d.set_index("ds").r).cumprod()
    return P / P.loc["2004-12-01"]  # base Dec-2004 = 1

def focus(indicator="IPCA"):
    if DATASET in ("ar", "super"):   # BCRA REM vintages (data/processed/ar_rem.parquet), keyed by publication date
        r = pd.read_parquet(ROOT / "data/processed/ar_rem.parquet")
        if indicator == "PIB":
            g = r[r.variable == "gdp_year"]
            return pd.DataFrame({"Data": g.pub_date.values, "year": g.target.dt.year.values, "Mediana": g["median"].values})
        return r
    if DATASET != "br":
        return None
    f = {"IPCA": "focus_IPCA.json", "PIB": "focus_PIBTotal.json"}[indicator]
    d = pd.DataFrame(json.load(open(RAW / f))["value"])
    d["Data"] = pd.to_datetime(d.Data); d["year"] = d.DataReferencia.astype(int)
    return d

def expected_path(origin, P, F, h=12, mode="focus"):
    """Expected price index for origin+1..origin+h."""
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    PT = P.loc[origin]
    if mode == "naive":  # trailing 12m inflation carried forward
        g = (P.loc[origin] / P.loc[origin - pd.DateOffset(months=12)]) ** (1 / 12)
        return pd.Series(PT * g ** np.arange(1, h + 1), index=months)
    if mode == "naive3":  # trailing 3m inflation carried forward (faster reaction to disinflation)
        g = (P.loc[origin] / P.loc[origin - pd.DateOffset(months=3)]) ** (1 / 3)
        return pd.Series(PT * g ** np.arange(1, h + 1), index=months)
    cut = origin + pd.offsets.MonthEnd(0)
    if mode in ("rem", "rem_last"):
        # latest REM vintage published by origin month-end (usually the previous survey month). Its explicit monthly
        # medians cover the next ~6 months after the SURVEY month; remaining months: "rem" = uniform rate that
        # completes the vintage's 12m-ahead median; "rem_last" = carry the last explicit monthly median.
        v = F[F.pub_date <= cut]; v = v[v.vintage_month == v.vintage_month.max()]
        vm = v.vintage_month.iloc[0]
        mm = v[v.variable == "cpi_m"].set_index("target")["median"] / 100
        c12 = v[v.variable == "cpi_12m"]["median"].iloc[0] / 100
        rest = ((1 + c12) / (1 + mm).prod()) ** (1 / (12 - len(mm))) - 1
        if mode == "rem_last":
            rest = mm.sort_index().iloc[-1]
        rates = {}
        for k in range(1, 25):
            m = vm + pd.DateOffset(months=k)
            rates[m] = mm.get(m, rest)
        cur, path = P.loc[origin], []
        for m in months:
            cur = cur * (1 + rates.get(m, rest)); path.append(cur)
        return pd.Series(path, index=months)
    v = F[F.Data <= cut]
    v = v[v.Data == v.Data.max()].set_index("year").Mediana / 100
    path, cur = [], PT
    for m in months:
        y = m.year
        pi = v.get(y, v.iloc[-1])
        if y == origin.year:  # remaining-year inflation implied by the annual expectation
            ytd = P.loc[origin] / P.loc[pd.Timestamp(f"{y-1}-12-01")]
            rem = 12 - origin.month
            g = ((1 + pi) / ytd) ** (1 / rem) if rem > 0 else 1.0
        else:
            g = (1 + pi) ** (1 / 12)
        cur = cur * g; path.append(cur)
    return pd.Series(path, index=months)
