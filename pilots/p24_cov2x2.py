"""P24 (review #5, M3): where the price information enters, as a 2x2 design for frozen Chronos-2.
Target nominal / deflated (deflated forecasts are re-inflated with the same origin-dated path) x CPI covariate
absent / present (log CPI, realised in the past and the expected path in the future). Encoding cells: CPI level as
covariate, log nominal target with and without the log-CPI covariate. National-total WAPE; paired moving-block
bootstrap contrasts (L=4, 5000 replicates).
usage: python p24_cov2x2.py -> paper/tables/tab_cov.tex, results/review5/p24_cov2x2.csv"""
import os, importlib
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)
PANELS = [("ar", None, "rem", "Argentina (tax)"), ("super", None, "rem", "Argentina (supermarkets)"),
          ("br", "kept_rt", "focus", "Brazil (tax)"), ("es", None, "naive", "Spain (tax)"), ("pe", None, "naive", "Peru (tax)")]


def env(d, kept):
    os.environ["DATASET"] = d
    if kept: os.environ["KEPT"] = kept
    else: os.environ.pop("KEPT", None)
    import common, p7_tests_lib
    for m in (common, p7_tests_lib): importlib.reload(m)
    return common, p7_tests_lib


def main():
    rows, tex = [], []
    for d, kept, P, name in PANELS:
        c, tl = env(d, kept); df = c.load_panel()
        cells = {"nom": "c2_xl_tax", "nom_cov": f"c2_cov_nom_{P}", "real": f"chronos_real_{P}", "real_cov": f"c2_cov_real_{P}",
                 "cov_level": f"c2_cov_level_{P}", "log_nom": f"c2_log_nom_{P}", "log_cov": f"c2_log_cov_{P}"}
        po, w = {}, {}
        for k, n in cells.items():
            if (c.RES / f"fc_{n}.parquet").exists():
                _, po[k] = tl.per_origin(n, df); w[k] = 100 * po[k].ae.sum() / po[k].y.sum()
        def contrast(a, b):
            if a not in po or b not in po: return (np.nan,) * 3
            pt, lo, hi, _ = tl.ci(po[a], po[b], "ae", "y", np.random.default_rng(0), L=4)
            return 100 * pt, 100 * lo, 100 * hi
        con = {"deflate_nocov": contrast("real", "nom"), "deflate_cov": contrast("real_cov", "nom_cov"),
               "cov_in_nominal": contrast("nom_cov", "nom"), "cov_in_deflated": contrast("real_cov", "real"),
               "log_nom_vs_nom": contrast("log_nom", "nom"), "log_cov_vs_log_nom": contrast("log_cov", "log_nom"),
               "real_vs_log_cov": contrast("real", "log_cov")}
        rows.append(dict(panel=d, path=P, **{f"wape_{k}": v for k, v in w.items()},
                         **{f"{k}_{s}": v[i] for k, v in con.items() for i, s in enumerate(["pt", "lo", "hi"])}))
        print(d, {k: round(v, 2) for k, v in w.items()}, {k: tuple(round(x, 2) for x in v) for k, v in con.items()}, flush=True)
        def cell(v):
            pt, lo, hi = v; sig = not (lo < 0 < hi)
            return (r"\textbf{" if sig else "") + f"{pt:+.1f}" + ("}" if sig else "") + f" [{lo:+.1f}, {hi:+.1f}]"
        tex.append(f"{name} & {P} & {w['nom']:.1f} & {w['nom_cov']:.1f} & {w['real']:.1f} & {w['real_cov']:.1f} & "
                   f"{cell(con['deflate_nocov'])} & {cell(con['deflate_cov'])} \\\\")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "p24_cov2x2.csv"), index=False)
    open(os.path.join(ROOT, "paper/tables/tab_cov.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
