"""LaTeX tables for the paper, generated from saved forecasts (no hand-typed numbers). Output: paper/tables/*.tex"""
import os, importlib, numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(__file__), "../paper/tables"); os.makedirs(OUT, exist_ok=True)


def load(d):
    os.environ["DATASET"] = d
    import common, macro, p4_oracle, p7_tests_lib
    for m in (common, macro, p4_oracle, p7_tests_lib):
        importlib.reload(m)
    return common, macro, p4_oracle, p7_tests_lib


def f3(x): return f"{x:.3f}"
def pct(x): return f"{100*x:.1f}"


def best(col, vals, low=True):
    nums = [float(v) for v in vals]; b = min(nums) if low else max(nums)
    return [r"\textbf{" + v + "}" if abs(x - b) < 1e-9 else v for v, x in zip(vals, nums)]


# ---------------- Table: data ----------------
rows = []
for d, name, src, lic in [("br", "Brazil", "Receita Federal (state)", "open data"),
                           ("ar", "Argentina", "Min. Economía, TOP (province)", "CC BY 4.0"),
                           ("es", "Spain", "AEAT (tax delegation)", "AEAT reuse terms"),
                           ("pe", "Peru$^\\dagger$", "SUNAT A15 (region)", "unverified")]:
    common, macro, p4, _ = load(d); df = common.load_panel()
    rows.append(f"{name} & {src} & {df.UF.nunique()} & {df.tax.nunique()} & {df.unique_id.nunique()} & "
                f"{df.ds.min():%Y-%m}--{df.ds.max():%Y-%m} & {len(common.ORIGINS)} & {lic} \\\\")
open(f"{OUT}/tab_data.tex", "w").write("\n".join(rows) + "\n")

# ---------------- Table: Brazil benchmark ----------------
common, macro, p4, tl = load("br"); df = common.load_panel(); sc = common.mase_scales(df)
BENCH = [("Seasonal naive", "snaive", "nominal"), ("ETS", "ets", "nominal"), ("AutoARIMA", "arima", "nominal"),
         ("AutoTheta", "theta", "nominal"), ("Chronos-2", "c2_xl_tax", "nominal"),
         ("AutoTheta", "theta_real_naive_gdp1", "level"), ("LightGBM (global)", "lgbm_real_naive_gdp1", "level"),
         ("DeepAR", "DeepAR_real_naive_gdp1", "level"), ("TFT", "TFT_real_naive_gdp1", "level"),
         ("PatchTST", "PatchTST_real_naive_gdp1", "level"), ("PatchTST (tuned, 3 seeds)", "PatchTST_tuned_real_naive_gdp1", "level"),
         ("NHITS (tuned, 3 seeds)", "NHITS_tuned_real_naive_gdp1", "level"), ("LSTM (tuned, 3 seeds)", "LSTM_tuned_real_naive_gdp1", "level"),
         ("TimesFM-2.5", "timesfm_real_naive_gdp1", "level"), ("TiRex", "tirex_real_naive_gdp1", "level"),
         ("Chronos-2 (= LSF shape layer)", "chronos_real_naive_gdp1", "level")]
res = []
for lab, n, grp in BENCH:
    f = common.RES / f"fc_{n}.parquet"
    if not f.exists(): continue
    fc = pd.read_parquet(f); r, _ = common.evaluate(fc, df, sc, by_h=False); s = p4.scores(fc, df)
    res.append((lab, grp, r["MASE_med"], r["WAPE_bottom"], r["WAPE_state"], r["WAPE_tax"], r["WAPE_total"], s[1]))
cols = list(zip(*[r[2:] for r in res]))
cols = [best(i, [f3(v) for v in c]) for i, c in enumerate(cols)]
lines, prev = [], None
for i, r in enumerate(res):
    if r[1] != prev:
        lines.append(r"\midrule" if prev else "")
        lines.append(r"\multicolumn{7}{l}{\emph{" + ("Nominal receipts" if r[1] == "nominal" else
                     "Level layer: real terms $\\times$ naive inflation path $\\times$ GDP anchor") + r"}}\\")
        prev = r[1]
    lines.append(f"{r[0]} & " + " & ".join(c[i] for c in cols) + r" \\")
open(f"{OUT}/tab_bench_br.tex", "w").write("\n".join(l for l in lines if l) + "\n")

# ---------------- Table: reconciliation (Brazil) ----------------
REC = [("Bottom-up", "rec_bu_chronos_real_focus_gdp1"), ("OLS", "rec_chronos_chronos_focus_ols"),
       ("WLS, structural", "rec_chronos_chronos_focus_wls_s"), ("WLS, past error variance", "rec_chronos_chronos_focus_wls_v"),
       ("WLS, proportional", "rec_chronos_chronos_focus_wls_p"),
       ("Scale to direct total", "rec_chronos_chronos_focus_scale_tot"), ("Rake to tax and state margins", "rec_chronos_chronos_focus_rake"),
       ("Rake to national tax series", "rec_chronos_chronos_focus_rake_tax"), ("Adaptive rake rule", "rec_chronos_chronos_focus_rake_sel")]
res = []
for lab, n in REC:
    fc = pd.read_parquet(common.RES / f"fc_{n}.parquet"); r, _ = common.evaluate(fc, df, sc, by_h=False)
    res.append((lab, r["MASE"], r["MASE_med"], r["WAPE_bottom"], r["WAPE_state"], r["WAPE_tax"], r["WAPE_total"]))
cols = list(zip(*[r[1:] for r in res])); fmt = [lambda v: f"{v:.2f}"] + [f3] * 5
cols = [best(i, [fmt[i](v) for v in c]) for i, c in enumerate(cols)]
open(f"{OUT}/tab_rec_br.tex", "w").write("\n".join(f"{r[0]} & " + " & ".join(c[i] for c in cols) + r" \\"
                                                   for i, r in enumerate(res)) + "\n")

# ---------------- Table: oracle decomposition (Brazil) ----------------
lines = []
for lab, n in [("Chronos-2 + level, bottom-up", "rec_bu_chronos_real_naive_gdp1"), ("AutoTheta + level", "theta_real_naive_gdp1"),
               ("Chronos-2 + level, rake to national taxes", "rec_chronos_chronos_rake_tax")]:
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        t = p4.main(n)
    lines.append(f"{lab} & " + " & ".join(f"{pct(t.loc[k, 'APE_12m_mean'])}" for k in ["base", "infl", "gdp", "both"])
                 + f" & {pct(t.loc['base', 'bias'])} & {pct(t.loc['both', 'bias'])}" + r" \\")
open(f"{OUT}/tab_oracle_br.tex", "w").write("\n".join(lines) + "\n")

# ---------------- Table: bootstrap comparisons (all countries) ----------------
CMP = [("br", "rec_chronos_chronos_focus_rake_tax_buoy", "ets", "LSF + raking + gap vs ETS"),
       ("br", "rec_chronos_chronos_focus_rake_tax_buoy", "arima", "LSF + raking + gap vs AutoARIMA"),
       ("br", "rec_chronos_chronos_focus_rake_tax_buoy", "theta_real_focus_gdp1_buoy", "LSF + raking + gap vs Theta + same level layer"),
       ("br", "chronos_real_naive_gdp1", "NHITS_tuned_real_naive_gdp1", "LSF vs tuned N-HiTS + same level layer"),
       ("ar", "chronos_real_rem_gdp1", "arima", "LSF vs AutoARIMA"),
       ("ar", "chronos_real_rem_gdp1", "c2_xl_tax", "LSF vs Chronos-2 nominal"),
       ("ar", "chronos_real_rem_gdp1", "theta_real_rem_gdp1", "LSF vs Theta + same level layer"),
       ("es", "chronos_real_naive_gdp1", "ets", "LSF vs ETS"),
       ("es", "chronos_real_naive_gdp1", "lgbm_real_naive", "LSF vs LightGBM + same level layer"),
       ("pe", "chronos_real_naive_gdp1", "theta", "LSF vs AutoTheta")]
NAME = {"br": "BR", "ar": "AR", "es": "ES", "pe": "PE"}
lines = []
for d, a, b, lab in CMP:
    common, macro, p4, tl = load(d); df = common.load_panel(); rng = np.random.default_rng(0)
    pa, ta = tl.per_origin(a, df); pb, tb = tl.per_origin(b, df); cells = []
    for A, B, k in [(pa, pb, "ay"), (ta, tb, "y")]:
        dlt, lo, hi, w = tl.ci(A, B, "ae", k, rng, L=4)
        sig = (hi < 0) or (lo > 0)
        cells.append((r"\textbf{" if sig else "") + f"{100*dlt:+.2f}" + ("}" if sig else "") + f" [{100*lo:+.2f}, {100*hi:+.2f}]")
    lines.append(f"{NAME[d]} & {lab} & " + " & ".join(cells) + r" \\")
open(f"{OUT}/tab_boot.tex", "w").write("\n".join(lines) + "\n")
print("tables written:", sorted(os.listdir(OUT)))
