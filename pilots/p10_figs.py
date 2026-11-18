"""Paper figures from saved forecast files (no re-estimation). Output: paper/figs/*.pdf + paper/figs/data.json
(the JSON feeds the interactive HTML companion)."""
import os, json, importlib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "../paper/figs"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#8a8986", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "pdf.fonttype": 42})
DATA = {}


def load(d):
    os.environ["DATASET"] = d
    import common, macro, p4_oracle
    for m in (common, macro, p4_oracle):
        importlib.reload(m)
    return common, macro, p4_oracle


def metrics(common, p4, n, df, sc):
    fc = pd.read_parquet(common.RES / f"fc_{n}.parquet")
    r, _ = common.evaluate(fc, df, sc, by_h=False); s = p4.scores(fc, df)
    return dict(bottom=r["WAPE_bottom"], total=r["WAPE_total"], ape12=s[1], bias=s[3])


# ---------- Fig 2: four-country dot plot (WAPE total and bottom) ----------
SURV = {"br": "focus", "ar": "rem", "es": "naive", "pe": "naive"}
NAME = {"br": "Brazil", "ar": "Argentina", "es": "Spain", "pe": "Peru"}
LAD = [("ETS", "ets"), ("AutoARIMA", "arima"), ("AutoTheta", "theta"), ("Chronos-2 (nominal)", "c2_xl_tax"),
       ("LightGBM + level", "lgbm_real_{s}"), ("Theta + level", "theta_real_{s}_gdp1"),
       ("LSF (Chronos-2 + level)", "chronos_real_{s}_gdp1")]
four = {}
for d in ["ar", "br", "es", "pe"]:
    common, macro, p4 = load(d); df = common.load_panel(); sc = common.mase_scales(df); rows = []
    for lab, pat in LAD:
        n = pat.format(s=SURV[d])
        if d == "ar" and n.startswith("lgbm"): n = "lgbm_real_rem_gdp1"
        if d in ("br",) and n.startswith("lgbm"): n = "lgbm_real_naive_gdp1"
        if d in ("es", "pe") and n.startswith("lgbm"): n = "lgbm_real_naive"
        if not (common.RES / f"fc_{n}.parquet").exists(): continue
        rows.append({"model": lab, **metrics(common, p4, n, df, sc)})
    if d == "br":
        rows.append({"model": "LSF + raking + gap (BR)", **metrics(common, p4, "rec_chronos_chronos_focus_rake_tax_buoy", df, sc)})
    four[d] = rows
DATA["four"] = four
fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True)
models = [m for m, _ in LAD] + ["LSF + raking + gap (BR)"]
for ax, d in zip(axs, ["ar", "br", "es", "pe"]):
    rr = {r["model"]: r for r in four[d]}
    for i, m in enumerate(models):
        if m not in rr: continue
        r = rr[m]
        ax.plot([r["total"] * 100, r["bottom"] * 100], [i, i], color=GRID, lw=1.2, zorder=1)
        ax.scatter(r["total"] * 100, i, s=26, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter(r["bottom"] * 100, i, s=26, color=ORANGE, zorder=3, marker="s", edgecolor="white", linewidth=0.8)
    ax.set_title(NAME[d], fontsize=9, color=INK); ax.set_xlabel("WAPE (%)"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(models))); axs[0].set_yticklabels(models); axs[0].invert_yaxis()
axs[0].scatter([], [], s=26, color=BLUE, label="national total"); axs[0].scatter([], [], s=26, color=ORANGE, marker="s", label="region × tax")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.01))
fig.savefig(f"{OUT}/fig_countries.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: level-layer ablation (BR, AR) ----------
STEPS = [("nominal", "{nom}"), ("real,\nnaive π", "{m}_real_naive"), ("+ GDP", "{m}_real_naive_gdp1"),
         ("real,\nsurvey π", "{m}_real_{s}"), ("+ GDP ", "{m}_real_{s}_gdp1")]
abl = {}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, d in zip(axs, ["br", "ar"]):
    common, macro, p4 = load(d); df = common.load_panel(); sc = common.mase_scales(df); abl[d] = {}
    for mod, nom, col, mk in [("chronos", "c2_xl_tax", BLUE, "o"), ("theta", "theta", ORANGE, "s")]:
        ys = [metrics(common, p4, pat.format(nom=nom, m=mod, s=SURV[d]), df, sc)["total"] * 100 for _, pat in STEPS]
        abl[d][mod] = ys
        ax.plot(range(len(STEPS)), ys, color=col, marker=mk, markersize=5, markeredgecolor="white")
        dy = 0 if d == "br" else (-6 if mod == "chronos" else 6)
        ax.annotate("Chronos-2" if mod == "chronos" else "Theta", (len(STEPS) - 1, ys[-1]), xytext=(6, dy),
                    textcoords="offset points", va="center", color=INK, fontsize=7.5)
    ax.set_xticks(range(len(STEPS))); ax.set_xticklabels([s for s, _ in STEPS], fontsize=7)
    ax.set_title(f"{NAME[d]} ({'Focus' if d == 'br' else 'BCRA REM'} survey)", fontsize=9)
    ax.set_ylabel("national-total WAPE (%)"); ax.set_xlim(-0.3, len(STEPS) - 0.2)
DATA["ablation"] = {"steps": [s.replace("\n", " ") for s, _ in STEPS], **abl}
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: Argentina inflation paths (12-month-ahead) ----------
common, macro, p4 = load("ar"); P = macro.ipca_index(); F = macro.focus("IPCA")
rows = []
for o in common.ORIGINS:
    e = {m: macro.expected_path(o, P, F, 12, m).iloc[-1] / P[o] - 1 for m in ["naive", "rem"]}
    a = P.reindex(pd.date_range(o + pd.offsets.MonthBegin(1), periods=12, freq="MS")).iloc[-1] / P[o] - 1
    rows.append({"origin": str(o.date()), "naive": e["naive"] * 100, "rem": e["rem"] * 100, "actual": a * 100})
ip = pd.DataFrame(rows); DATA["ar_inflation"] = rows
fig, ax = plt.subplots(figsize=(3.4, 2.4))
x = pd.to_datetime(ip.origin)
ax.plot(x, ip.actual, color=INK, lw=1.6, label="realised")
ax.plot(x, ip.naive, color=ORANGE, marker="s", markersize=3.5, label="naive (trailing 12m)")
ax.plot(x, ip.rem, color=BLUE, marker="o", markersize=3.5, label="REM survey path")
ax.set_ylabel("12-month-ahead CPI inflation (%)"); ax.set_xlabel("forecast origin"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ar_inflation.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: bias by origin year (BR, AR) ----------
def bias_year(common, n, df):
    m = pd.read_parquet(common.RES / f"fc_{n}.parquet").merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    t = m.groupby("origin")[["y", "yhat"]].sum()
    return ((t.yhat / t.y - 1).groupby(t.index.year).mean() * 100).round(2)
by = {}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, d, sets in [(axs[0], "br", [("ETS", "ets", GRAY, "^"), ("LSF, bottom-up", "rec_bu_chronos_real_focus_gdp1", ORANGE, "s"),
                                    ("LSF + raking + gap (BR)", "rec_chronos_chronos_focus_rake_tax_buoy", BLUE, "o")]),
                    (axs[1], "ar", [("AutoARIMA", "arima", GRAY, "^"), ("Chronos-2 real, naive π", "chronos_real_naive", ORANGE, "s"),
                                    ("LSF (REM path)", "chronos_real_rem_gdp1", BLUE, "o")])]:
    common, macro, p4 = load(d); df = common.load_panel(); by[d] = {}
    for lab, n, col, mk in sets:
        b = bias_year(common, n, df); by[d][lab] = {int(k): float(v) for k, v in b.items()}
        ax.plot(b.index, b.values, color=col, marker=mk, markersize=4, label=lab, markeredgecolor="white")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_title(NAME[d], fontsize=9); ax.set_xlabel("origin year")
    ax.set_ylabel("bias of 12-month sum (%)"); ax.legend(fontsize=6.5, loc="best")
DATA["bias"] = by
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bias.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 6: Brazil buoyancy gap ----------
common, macro, p4 = load("br"); df = common.load_panel(); P = macro.ipca_index(); G = p4.realised_gdp()
t = df.groupby("ds").y.sum() / P.reindex(sorted(df.ds.unique())).values
a = t.groupby(t.index.year).agg(["sum", "size"]); a = a[a["size"] == 12]["sum"]
g = ((a.pct_change() - G.reindex(a.index)) * 100).dropna(); g = g[g.index <= 2025]
DATA["buoyancy_gap"] = {int(k): round(float(v), 2) for k, v in g.items()}
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.bar(g.index, g.values, color=[BLUE if v >= 0 else ORANGE for v in g.values], width=0.7, edgecolor="white", linewidth=0.8)
ax.axhline(0, color=INK2, lw=0.8)
ax.axhline(g[g.index <= 2019].mean(), color=INK2, lw=1, ls="--")
ax.text(0.02, 0.97, f"dashed: 2006–2019 mean {g[g.index <= 2019].mean():+.1f} pp", transform=ax.transAxes,
        fontsize=7, color=INK2, va="top")
for yr in (2021, 2024):
    ax.annotate(f"{g[yr]:+.1f}", (yr, g[yr]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
ax.set_ylabel("receipts minus GDP growth, real (pp)"); ax.set_xlabel("year"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_buoyancy.pdf", bbox_inches="tight"); plt.close(fig)

json.dump(DATA, open(f"{OUT}/data.json", "w"), indent=1, default=float)
print("figures written to", os.path.abspath(OUT))
