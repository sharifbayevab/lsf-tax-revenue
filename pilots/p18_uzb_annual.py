"""Uzbekistan: annual national check of the level layer (open data only).

Receipts: Ministry of Finance / data.egov.uz, state-budget revenue by tax,
annual 2015-2025 (CC BY-SA), dataset 62d151ae0f4de3c96a299272.
CPI and GDP growth: SIAT (stat.uz), see p17_uzb_siat.py.
Inflation expectations: Central Bank of Uzbekistan household survey,
next-12-month expectation.

Only annual national receipts by tax are public, so the shape layer forecasts
one step of an annual real series; the test is about the level layer.
Origin T = mid-January of year T+1: receipts and CPI of year T are known, the
latest survey used is the one for November of year T, the growth anchor is
real GDP growth over the first nine months of year T (SIAT, current vintage).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UZB = ROOT / "data" / "uzb"
OUT = ROOT / "results" / "uzb"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2015, 2026))
ORIGINS = list(range(2019, 2025))          # forecast years 2020..2025
MONTHS_UZ = ["Yanvar", "Fevral", "Mart", "Aprel", "May", "Iyun", "Iyul", "Avgust",
             "Sentyabr", "Oktyabr", "Noyabr", "Dekabr"]


def num(s):
    s = str(s).strip()
    if s in ("-", "", "nan"):
        return 0.0
    return float(s.replace("\xa0", "").replace(" ", "").replace(",", "."))


def load_receipts():
    rows = json.loads((UZB / "egov" / "revenue_by_tax_annual_62d151ae0f4de3c96a299272.json").read_text())
    raw = {r["Korsatkichlar"].replace("▼", "").strip(): [num(r[f"{y}Y"]) for y in YEARS] for r in rows}
    g = lambda k: np.array(next(v for n, v in raw.items() if n.startswith(k)))
    direct, indirect, resource = g("Bevosita"), g("Bilvosita"), g("Resurs")
    leaves = {
        "PIT": g("Jismoniy shaxslardan"),
        "CIT": g("Foyda solig") + g("Yuridik shaxslardan olinadigan foyda"),
        "VAT": g("Qo‘shilgan qiymat"),
        "Excise": g("Aksiz"),
        "Customs": g("Bojxona"),
        "Property": g("Mol-mulk"),
        "Land": g("Yer solig"),
        "Subsoil": g("Yer qa’ridan") + g("Yer osti"),
        "Water": g("Suv resurslaridan"),
    }
    leaves["Direct other"] = direct - leaves["PIT"] - leaves["CIT"]
    leaves["Indirect other"] = indirect - leaves["VAT"] - leaves["Excise"] - leaves["Customs"]
    leaves["Resource other"] = resource - sum(leaves[k] for k in ("Property", "Land", "Subsoil", "Water"))
    df = pd.DataFrame(leaves, index=YEARS)
    assert np.allclose(df.sum(1).values, direct + indirect + resource)
    return df


def load_cpi():
    a = pd.read_csv(UZB / "siat" / "1285_cpi_monthly_mom_2010_2020.csv")
    b = pd.read_csv(UZB / "siat" / "1286_cpi_monthly_mom_coicop2018.csv")
    a = a[a.name_ru.str.startswith("Сводный")]
    b = b[b.name_ru.str.startswith("Сводный")]
    m = pd.concat([a, b])[["period", "value"]].drop_duplicates("period")
    m["date"] = pd.PeriodIndex(m.period.str.replace("-M", "-"), freq="M")
    m = m.sort_values("date").set_index("date").value / 100
    idx = m.cumprod()
    return idx  # monthly index, 2010-01 based


def load_expectations():
    x = pd.read_excel(UZB / "cbu" / "inflation_expectations.xlsx", header=None).iloc[2:]
    out = {}
    for lab, hh in zip(x[0], x[1]):
        mon, yr = str(lab).rsplit(" ", 1)
        out[pd.Period(f"{yr}-{MONTHS_UZ.index(mon.strip()) + 1:02d}", "M")] = float(hh) / 100
    return pd.Series(out).sort_index()


def load_growth_9m():
    g = pd.read_csv(UZB / "siat" / "1857_grp_quarterly_growth.csv")
    g = g[g.code == 1700]
    return {int(p[:4]): v / 100 - 1 for p, v in zip(g.period, g.value) if p.endswith("Q3")}


def theta_1step(y):
    from statsforecast.models import AutoTheta
    if len(y) < 3:
        return y[-1]
    try:
        return float(AutoTheta(season_length=1).forecast(y=np.asarray(y, float), h=1)["mean"][0])
    except Exception:
        return y[-1]


def chronos_1step(pipe, hist):
    """hist: DataFrame index=years, columns=series; joint forecast with group attention."""
    long = hist.stack().rename("y").reset_index()
    long.columns = ["year", "unique_id", "y"]
    long["ds"] = pd.to_datetime(long.year.astype(str) + "-12-31")
    p = pipe.predict_df(long[["unique_id", "ds", "y"]], id_column="unique_id", timestamp_column="ds",
                        target="y", prediction_length=1, quantile_levels=[0.5],
                        batch_size=hist.shape[1], cross_learning=True)
    return p.set_index("unique_id")["0.5"].reindex(hist.columns)


def main(device="mps"):
    Y = load_receipts()
    cpi = load_cpi()
    pbar = cpi.groupby(cpi.index.year).mean()
    exp = load_expectations()
    g9 = load_growth_9m()
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    recs = []
    for T in ORIGINS:
        hist = Y.loc[:T]
        X = hist.div(pbar.loc[hist.index], axis=0)                 # real receipts
        p_dec = cpi[pd.Period(f"{T}-12", "M")]
        # survey path: Nov survey, constant monthly rate through year T+1
        pi_s = exp[pd.Period(f"{T}-11", "M")]
        # naive path: trailing Dec/Dec inflation
        pi_n = p_dec / cpi[pd.Period(f"{T-1}-12", "M")] - 1
        ratio = {}
        for name, pi in (("survey", pi_s), ("naive", pi_n)):
            pbar_next = p_dec * np.mean([(1 + pi) ** (m / 12) for m in range(1, 13)])
            ratio[name] = pbar_next / pbar[T]
        G = 1 + g9[T]
        c2_nom = chronos_1step(pipe, hist)
        c2_real = chronos_1step(pipe, X)
        fc = {}
        fc["Naive"] = hist.loc[T]
        prev = hist.loc[T - 1]
        gr = (hist.loc[T] / prev.where(prev > 0)).clip(0.5, 2.0).fillna(1.0)
        fc["Drift"] = hist.loc[T] * gr
        fc["AutoTheta nominal"] = pd.Series({c: theta_1step(hist[c].values) for c in hist}, dtype=float)
        fc["Chronos-2 nominal"] = c2_nom
        for path in ("survey", "naive"):
            for e in (0, 1):
                lev = pbar[T] * ratio[path] * G ** e
                tag = f"{path}, e={e}"
                fc[f"Level only ({tag})"] = X.loc[T] * lev
                fc[f"AutoTheta + level ({tag})"] = pd.Series(
                    {c: theta_1step(X[c].values) for c in X}, dtype=float) * lev
                fc[f"LSF ({tag})"] = c2_real * lev
        act = Y.loc[T + 1]
        for m, f in fc.items():
            f = f.clip(lower=0)
            tot_err = f.sum() / act.sum() - 1
            wape = (f - act).abs().sum() / act.sum()
            recs.append({"origin": T, "target": T + 1, "method": m, "total_err": tot_err,
                         "total_ape": abs(tot_err), "leaf_wape": wape})
        print(T, f"pi_survey={pi_s:.3f} pi_naive={pi_n:.3f} g9={g9[T]:.3f}", flush=True)
    r = pd.DataFrame(recs)
    r.to_csv(OUT / "annual_check.csv", index=False)
    s = r.groupby("method").agg(total_ape=("total_ape", "mean"), total_bias=("total_err", "mean"),
                                leaf_wape=("leaf_wape", "mean")).sort_values("total_ape")
    s.to_csv(OUT / "annual_check_summary.csv")
    pd.set_option("display.width", 160)
    print((s * 100).round(1).to_string())
    print(r.pivot(index="method", columns="target", values="total_err").mul(100).round(1)
          .loc[s.index].to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mps")
