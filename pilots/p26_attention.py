"""P26 (review #5, M8): does cross-series (group) attention matter in the LSF setting? Chronos-2 on deflated receipts
with the paper's level path (no growth anchor), four groupings: by tax (default), no cross-learning (independent),
by region, random groups of ~100 series. Bottom and national-total WAPE; paired bootstrap of tax vs independent at the
bottom; wall-clock seconds per origin on the Mac (MPS) from results/<ds>/timing_attention.csv. The by-tax accuracy is
taken from the re-run used for timing (`_gtaxre`), so that all four variants come from the same session.
usage: python p26_attention.py -> paper/tables/tab_attn.tex, results/review5/p26_attention.csv"""
import os, importlib
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)
PANELS = [("br", "kept_rt", "focus", "Brazil (tax)"), ("ar", None, "rem", "Argentina (tax)"),
          ("super", None, "rem", "Argentina (supermarkets)"), ("es", None, "naive", "Spain (tax)"), ("pe", None, "naive", "Peru (tax)")]
VARS = [("taxre", "by tax"), ("indep", "independent"), ("region", "by region"), ("random", "random")]


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
        tm = pd.read_csv(c.RES / "timing_attention.csv", header=None, names=["path", "group", "sec", "n"])
        tm = tm[tm.path == P].drop_duplicates("group", keep="last").set_index("group")
        res, po = {}, {}
        for g, lab in VARS:
            b, t = tl.per_origin(f"chronos_real_{P}_g{g}", df); po[g] = (b, t)
            res[g] = (100 * b.ae.sum() / b.ay.sum(), 100 * t.ae.sum() / t.y.sum(), tm.loc[g, "sec"] / tm.loc[g, "n"])
        rng = np.random.default_rng(0)
        pt, lo, hi, _ = tl.ci(po["taxre"][0], po["indep"][0], "ae", "ay", rng, L=4)
        pt2, lo2, hi2, _ = tl.ci(po["taxre"][1], po["indep"][1], "ae", "y", rng, L=4)
        _, frozen = tl.per_origin(f"chronos_real_{P}", df)
        rows.append(dict(panel=d, **{f"{g}_{k}": v for g, r in res.items() for k, v in zip(["bottom", "total", "sec"], r)},
                         tax_vs_indep_bottom=100 * pt, lo=100 * lo, hi=100 * hi,
                         tax_vs_indep_total=100 * pt2, lo_t=100 * lo2, hi_t=100 * hi2,
                         frozen_total=100 * frozen.ae.sum() / frozen.y.sum()))
        print(rows[-1], flush=True)
        sig = not (lo < 0 < hi)
        cell = (r"\textbf{" if sig else "") + f"{100*pt:+.2f}" + ("}" if sig else "") + f" [{100*lo:+.2f}, {100*hi:+.2f}]"
        tex.append(f"{name} & " + " & ".join(f"{res[g][0]:.1f} / {res[g][1]:.1f}" for g, _ in VARS) + f" & {cell} & "
                   + " / ".join(f"{res[g][2]:.2f}" for g, _ in VARS) + r" \\")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "p26_attention.csv"), index=False)
    open(os.path.join(ROOT, "paper/tables/tab_attn.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
