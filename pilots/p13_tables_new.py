"""Tables for: level outside vs inside (covariate) across 5 panels; LSF-T vs LSF-R; supermarket panel row in data."""
import os, importlib, numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(__file__), "../paper/tables")
def load(d, kept=None):
    os.environ["DATASET"] = d
    if kept: os.environ["KEPT"] = kept
    else: os.environ.pop("KEPT", None)
    import common, macro, p4_oracle, p7_tests_lib
    for m in (common, macro, p4_oracle, p7_tests_lib): importlib.reload(m)
    return common, p4_oracle, p7_tests_lib
NAME = {"ar": "Argentina (tax)", "super": "Argentina (supermarkets)", "br": "Brazil (tax)", "es": "Spain (tax)", "pe": "Peru (tax)"}
SURV = {"br": "focus", "ar": "rem", "super": "rem"}
cov, tr = [], []
for d in ["ar", "super", "br", "es", "pe"]:
    c, p4, tl = load(d, "kept_rt" if d == "br" else None); df = c.load_panel(); P = SURV.get(d, "naive"); rng = np.random.default_rng(0)
    w = lambda n: c.evaluate(pd.read_parquet(c.RES / f"fc_{n}.parquet"), df, None, by_h=False)[0]["WAPE_total"]
    a, b = f"chronos_real_{P}", f"c2_cov_nom_{P}"
    pa, ta = tl.per_origin(a, df); pb, tb = tl.per_origin(b, df)
    dl, lo, hi, _ = tl.ci(ta, tb, "ae", "y", rng, L=4); sig = hi < 0 or lo > 0
    cell = (r"\textbf{" if sig else "") + f"{100*dl:+.1f}" + ("}" if sig else "") + f" [{100*lo:+.1f}, {100*hi:+.1f}]"
    cov.append(f"{NAME[d]} & {P} & {100*w('c2_xl_tax'):.1f} & {100*w(b):.1f} & {100*w(a):.1f} & {cell} \\\\")
    if d == "super": continue
    if d == "br": c, p4, tl = load(d); df = c.load_panel()  # TSFM comparison run on the full-sample filter
    PP = {"br": "naive", "ar": "rem"}.get(d, "naive"); vals = []
    for n in [f"chronos_real_{PP}_gdp1", f"tirex_real_{PP}_gdp1", f"timesfm_real_{PP}_gdp1"]:
        r = c.evaluate(pd.read_parquet(c.RES / f"fc_{n}.parquet"), df, None, by_h=False)[0]
        vals.append((r["WAPE_bottom"], r["WAPE_total"]))
    bb = min(v[0] for v in vals); bt = min(v[1] for v in vals)
    fmt = lambda x, b: (r"\textbf{%.3f}" % x) if abs(x - b) < 1e-9 else "%.3f" % x
    tr.append(f"{NAME[d]} & " + " & ".join(f"{fmt(v[0], bb)} & {fmt(v[1], bt)}" for v in vals) + r" \\")
open(f"{OUT}/tab_cov.tex", "w").write("\n".join(cov) + "\n")
open(f"{OUT}/tab_tr.tex", "w").write("\n".join(tr) + "\n")
c, p4, tl = load("super"); df = c.load_panel()
open(f"{OUT}/tab_data_super.tex", "w").write(
    f"Argentina$^\\ddagger$ & INDEC supermarket sales (area $\\times$ product) & {df.UF.nunique()} & {df.tax.nunique()} & "
    f"{df.unique_id.nunique()} & {df.ds.min():%Y-%m}--{df.ds.max():%Y-%m} & {len(c.ORIGINS)} & CC BY-SA 4.0 \\\\\n")
print(open(f"{OUT}/tab_cov.tex").read()); print(open(f"{OUT}/tab_tr.tex").read())
