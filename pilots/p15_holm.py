"""P15 (review #3, K2): multiplicity control over the family of headline comparisons (assembled after the experiments).
Paired circular moving-block bootstrap over origins (L=4 primary, R=10000), WAPE difference A - B at the stated
level; two-sided p from the centred bootstrap distribution; Holm and Benjamini-Hochberg adjustment over the family.
usage: python p15_holm.py  -> results/p15_holm.csv, paper/tables/tab_holm.tex"""
import os, importlib, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# id, panel, kept, A, B, level, label
FAMILY = [
    ("H1", "ar", None, "chronos_real_rem_gdp1", "arima", "total", "AR tax: LSF vs best unlayered (ARIMA)"),
    ("H2", "ar", None, "chronos_real_rem", "c2_cov_nom_rem", "total", "AR tax: level outside vs CPI covariate"),
    ("H3", "super", None, "chronos_real_rem", "c2_cov_nom_rem", "total", "AR super: level outside vs CPI covariate"),
    ("H4", "br", "kept_rt", "chronos_real_focus", "c2_cov_nom_focus", "total", "BR: level outside vs CPI covariate"),
    ("H5", "super", None, "chronos_real_rem_gdp1", "arima", "total", "AR super: LSF vs ARIMA"),
    ("H6", "br", "kept_rt", "rec_chronos_chronos_focus_rake_tax_buoy", "ets", "total", "BR: full LSF vs ETS"),
    ("H7", "br", "kept_rt", "rec_chronos_chronos_focus_rake_tax", "rec_bu_chronos_real_focus_gdp1", "total",
     "BR: raking vs bottom-up"),
    ("H8", "br", "kept_rt", "rec_bu_chronos_real_focus_gdp1", "rec_bu_theta_real_focus_gdp1", "bottom",
     "BR: Chronos-2 vs Theta shape, same level"),
    ("H9", "br", "kept_rt", "chronos_real_focus_gdp1", "chronos_real_focus", "total", "BR: GDP multiplier e=1 vs e=0"),
    ("H10", "ar", None, "chronos_real_rem_gdp1", "chronos_real_rem", "total", "AR tax: GDP multiplier e=1 vs e=0"),
]


def env(d, kept):
    os.environ["DATASET"] = d
    if kept: os.environ["KEPT"] = kept
    else: os.environ.pop("KEPT", None)
    import common, p7_tests_lib
    for m in (common, p7_tests_lib): importlib.reload(m)
    return common, p7_tests_lib


def boot(A, B, ka, ky, rng, R=10000, L=4):
    n = len(A); d = np.empty(R)
    for r in range(R):
        s = rng.integers(0, n, int(np.ceil(n / L)))
        i = np.concatenate([(np.arange(x, x + L) % n) for x in s])[:n]
        d[r] = A[ka].iloc[i].sum() / A[ky].iloc[i].sum() - B[ka].iloc[i].sum() / B[ky].iloc[i].sum()
    pt = A[ka].sum() / A[ky].sum() - B[ka].sum() / B[ky].sum()
    p = min(1.0, (np.abs(d - d.mean()) >= abs(pt)).mean() + 1 / R)
    return pt, *np.quantile(d, [.025, .975]), p


def main():
    rows = []
    for hid, d, kept, a, b, lev, lab in FAMILY:
        c, tl = env(d, kept); df = c.load_panel(); rng = np.random.default_rng(0)
        if not (c.RES / f"fc_{b}.parquet").exists() and b == "rec_bu_theta_real_focus_gdp1":
            b = "theta_real_focus_gdp1"
        ba, ta = tl.per_origin(a, df); bb, tb = tl.per_origin(b, df)
        common_o = ta.index.intersection(tb.index)
        if lev == "total": A, B, ka, ky = ta.loc[common_o], tb.loc[common_o], "ae", "y"
        else: A, B, ka, ky = ba.loc[common_o], bb.loc[common_o], "ae", "ay"
        pt, lo, hi, p = boot(A, B, ka, ky, rng)
        rows.append(dict(id=hid, label=lab, panel=d, level=lev, A=a, B=b, n_origins=len(common_o),
                         diff_pp=100 * pt, lo=100 * lo, hi=100 * hi, p=p))
        print(hid, lab, f"{100*pt:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] p={p:.4f}", flush=True)
    t = pd.DataFrame(rows); m = len(t)
    o = np.argsort(t.p.values); holm = np.empty(m); run = 0
    for k, j in enumerate(o):
        run = max(run, min(1, (m - k) * t.p.values[j])); holm[j] = run
    bh = np.empty(m); prev = 1
    for k, j in list(enumerate(o))[::-1]:
        prev = min(prev, t.p.values[j] * m / (k + 1)); bh[j] = prev
    t["p_holm"], t["p_bh"] = holm, bh
    t.to_csv(os.path.join(ROOT, "results/p15_holm.csv"), index=False)
    print(t[["id", "diff_pp", "lo", "hi", "p", "p_holm", "p_bh"]].round(4).to_string(index=False))
    fp = lambda x: "$<$0.001" if x < 0.001 else "%.3f" % x
    bf = lambda x: ("\\textbf{%s}" % fp(x)) if x < .05 else fp(x)
    lines = [f"{r.id} & {r.label} & {r.level} & {r.diff_pp:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] & "
             + fp(r.p) + " & " + bf(r.p_holm) + r" \\" for r in t.itertuples()]
    open(os.path.join(ROOT, "paper/tables/tab_holm.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
