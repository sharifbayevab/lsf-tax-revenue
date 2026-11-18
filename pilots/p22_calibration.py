"""P22 (review #5, M7): does the growth anchor only offset a bias? Compare it with a bias calibration of the shape
forecast computed from past errors only.

Calibration at origin o: c_o = sum(actual) / sum(forecast) over all earlier forecasts of the same method whose target
month is observed by o (ds <= o), pooled over series and horizons; applied to every forecast of origin o when at
least four earlier origins contribute, otherwise c_o = 1. Evaluated on origins from 2020-12 (all variants on the same
origins). Panels: Brazil (kept_rt, Focus path) and Argentina taxes (REM path).
usage: python p22_calibration.py  -> results/review5/p22_calibration.csv, paper/tables/tab_calib.tex"""
import os, importlib
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)
FIRST = pd.Timestamp("2020-12-01")
PANELS = [("br", "kept_rt", "focus", "Brazil (tax)"), ("ar", None, "rem", "Argentina (tax)")]
MODELS = [("chronos", "Chronos-2"), ("theta", "AutoTheta")]


def env(d, kept):
    os.environ["DATASET"] = d
    if kept: os.environ["KEPT"] = kept
    else: os.environ.pop("KEPT", None)
    import common; importlib.reload(common)
    return common


def calibrate(fc, act):
    m = fc.merge(act, on=["unique_id", "ds"], how="left")
    parts = []
    for o, g in fc.groupby("origin"):
        past = m[(m.origin < o) & (m.ds <= o) & m.y.notna()]
        c = past.y.sum() / past.yhat.sum() if past.origin.nunique() >= 4 else 1.0
        parts.append(g.assign(yhat=g.yhat * c))
    return pd.concat(parts)


def score(fc, act):
    m = fc[fc.origin >= FIRST].merge(act, on=["unique_id", "ds"])
    t = m.groupby(["origin", "ds"])[["y", "yhat"]].sum()
    wape = (t.y - t.yhat).abs().sum() / t.y.abs().sum()
    s12 = t.groupby("origin")[["y", "yhat"]].sum()
    s12 = s12[m.groupby("origin").h.max() == 12]
    bias = (s12.yhat / s12.y - 1).mean()
    per_o = t.assign(ae=(t.y - t.yhat).abs()).groupby("origin")[["ae", "y"]].sum()
    return wape, bias, per_o


def boot(A, B, R=5000, L=4, seed=0):
    rng = np.random.default_rng(seed); n = len(A); d = np.empty(R)
    for r in range(R):
        s = rng.integers(0, n, int(np.ceil(n / L)))
        i = np.concatenate([(np.arange(x, x + L) % n) for x in s])[:n]
        d[r] = A.ae.iloc[i].sum() / A.y.iloc[i].sum() - B.ae.iloc[i].sum() / B.y.iloc[i].sum()
    return np.quantile(d, [.025, .975])


def main():
    rows, tex = [], []
    for d, kept, path, name in PANELS:
        c = env(d, kept); df = c.load_panel(); act = df[["unique_id", "ds", "y"]]
        for mk, mname in MODELS:
            base = f"{mk}_real_{path}"
            f0 = pd.read_parquet(c.RES / f"fc_{base}.parquet"); f1 = pd.read_parquet(c.RES / f"fc_{base}_gdp1.parquet")
            res = {}
            for lab, f in [("e0", f0), ("e1", f1), ("e0_cal", calibrate(f0, act)), ("e1_cal", calibrate(f1, act))]:
                res[lab] = score(f, act)
            lo0, hi0 = boot(res["e1"][2], res["e0"][2]); lo1, hi1 = boot(res["e1_cal"][2], res["e0_cal"][2])
            r = dict(panel=d, model=mk, **{f"wape_{k}": 100 * v[0] for k, v in res.items()},
                     **{f"bias12_{k}": 100 * v[1] for k, v in res.items()},
                     anchor_gain=100 * (res["e1"][0] - res["e0"][0]), anchor_gain_lo=100 * lo0, anchor_gain_hi=100 * hi0,
                     anchor_gain_cal=100 * (res["e1_cal"][0] - res["e0_cal"][0]),
                     anchor_gain_cal_lo=100 * lo1, anchor_gain_cal_hi=100 * hi1, n_origins=len(res["e0"][2]))
            rows.append(r); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            sig = lambda lo, hi: not (lo < 0 < hi)
            fmt = lambda x, lo, hi: (r"\textbf{%+.2f}" % x if sig(lo, hi) else "%+.2f" % x) + f" [{lo:+.2f}, {hi:+.2f}]"
            tex.append(f"{name} & {mname} & {r['wape_e0']:.1f} & {r['wape_e1']:.1f} & {r['wape_e0_cal']:.1f} & "
                       f"{r['wape_e1_cal']:.1f} & {fmt(r['anchor_gain'], lo0 * 100, hi0 * 100)} & "
                       f"{fmt(r['anchor_gain_cal'], lo1 * 100, hi1 * 100)} \\\\")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "p22_calibration.csv"), index=False)
    open(os.path.join(ROOT, "paper/tables/tab_calib.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
