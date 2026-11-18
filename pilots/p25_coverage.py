"""P25 (review #5, M6): how much of each source table the retained series cover.
Share = sum of receipts of the retained (kept) series / sum over all region-by-tax cells of the processed source panel,
over the target months of the test window (month after the first origin to the end of the data). Spain is also split
by tax. Uzbekistan: the 12 series (three tax groups) against all state-budget revenue in the source table, 2020-2025.
Inserts a 'Share' column before the licence column of paper/tables/tab_data*.tex (rows written by p11_tables.py,
p13_tables_new.py, p18b_uzb_table.py).
usage: python p25_coverage.py -> results/review5/p25_coverage.csv"""
import os, json, re
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)
PANELS = [("Brazil", "brazil_panel", "kept_rt", "2020-01-01"), ("Argentina", "ext_argentina_panel", "kept", "2020-01-01"),
          ("Spain", "ext_spain_panel", "kept", "2020-01-01"), ("Peru", "ext_peru_panel", "kept", "2020-01-01"),
          ("Argentina$^\\ddagger$", "ext_super_panel", "kept", "2020-01-01")]


def uzb_share():
    rows = json.loads(open(os.path.join(ROOT, "data/uzb/egov/revenue_by_tax_annual_62d151ae0f4de3c96a299272.json")).read())
    num = lambda s: 0.0 if str(s).strip() in ("-", "") else float(str(s).replace("\xa0", "").replace(" ", "").replace(",", "."))
    grp = {r["Korsatkichlar"].replace("▼", "").strip()[:8]: r for r in rows if r["Korsatkichlar"].startswith("▼")}
    yrs = [f"{y}Y" for y in range(2020, 2026)]
    tax = sum(num(grp[k][y]) for k in ("Bevosita", "Bilvosit", "Resurs t") for y in yrs)
    allr = sum(num(r[y]) for r in rows if r["Korsatkichlar"].startswith("▼") for y in yrs) + \
        sum(num(r[y]) for r in rows if r["Korsatkichlar"].strip().startswith("Boshqa daromadlar") for y in yrs)
    return tax / allr


def main():
    out = []
    for name, f, kept, start in PANELS:
        d = pd.read_parquet(os.path.join(ROOT, "data/processed", f + ".parquet"))
        d = d[d.ds >= start]
        share = d.loc[d[kept], "y"].sum() / d.y.sum()
        out.append(dict(panel=name, file=f, kept=int(d[d[kept]].unique_id.nunique()), cells=int(d.unique_id.nunique()),
                        share=100 * share))
        if f == "ext_spain_panel":
            for t, g in d.groupby("tax"):
                out.append(dict(panel=f"Spain {t}", file=f, kept=int(g[g[kept]].unique_id.nunique()),
                                cells=int(g.unique_id.nunique()), share=100 * g.loc[g[kept], "y"].sum() / g.y.sum()))
    out.append(dict(panel="Uzbekistan$^\\S$", file="egov", kept=12, cells=12, share=100 * uzb_share()))
    t = pd.DataFrame(out); t.to_csv(os.path.join(OUT, "p25_coverage.csv"), index=False); print(t.round(1).to_string(index=False))
    sh = {r.panel: r.share for r in t.itertuples()}
    for fn in ["tab_data.tex", "tab_data_super.tex", "tab_data_uzb.tex"]:
        p = os.path.join(ROOT, "paper/tables", fn); lines = open(p).read().rstrip("\n").split("\n"); new = []
        for ln in lines:
            cells = [c.strip() for c in ln.rstrip("\\ ").split("&")]
            key = cells[0].replace("$^\\dagger$", "") if fn == "tab_data.tex" else cells[0]
            if len(cells) == 8:  # not yet inserted
                cells.insert(7, f"{sh[key]:.0f}\\%")
            new.append(" & ".join(cells) + " \\\\")
        open(p, "w").write("\n".join(new) + "\n")
        print(open(p).read())


if __name__ == "__main__":
    main()
