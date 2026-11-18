"""P21 (review #5, M4): robustness of the ten headline comparisons of p15_holm.py to single years.
For each comparison A - B (WAPE, percentage points, same level as in Table 10):
  full difference; range of the difference when all origins of one calendar year are left out (origin year) and when
  all target months of one calendar year are left out (target year); origins won by A; wins inside the four sets of
  non-overlapping origins (December, March, June, September origins: disjoint twelve-month windows).
usage: python p21_loyo.py -> results/review5/p21_loyo.csv, results/review5/p21_by_origin_year.csv, paper/tables/tab_loyo.tex"""
import os, importlib
import numpy as np, pandas as pd
from p15_holm import FAMILY

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/review5"); os.makedirs(OUT, exist_ok=True)


def env(d, kept):
    os.environ["DATASET"] = d
    if kept: os.environ["KEPT"] = kept
    else: os.environ.pop("KEPT", None)
    import common; importlib.reload(common)
    return common


def cells(c, name, df, level):
    m = pd.read_parquet(c.RES / f"fc_{name}.parquet").merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    if level == "bottom":
        m = m.assign(ae=(m.y - m.yhat).abs(), ay=m.y.abs())
        return m.groupby(["origin", "ds"])[["ae", "ay"]].sum()
    t = m.groupby(["origin", "ds"])[["y", "yhat"]].sum()
    return pd.DataFrame({"ae": (t.y - t.yhat).abs(), "ay": t.y.abs()})


def diff(A, B):
    return 100 * (A.ae.sum() / A.ay.sum() - B.ae.sum() / B.ay.sum())


def main():
    rows, years = [], []
    for hid, d, kept, a, b, lev, lab in FAMILY:
        c = env(d, kept); df = c.load_panel()
        if not (c.RES / f"fc_{b}.parquet").exists() and b == "rec_bu_theta_real_focus_gdp1":
            b = "theta_real_focus_gdp1"
        A = cells(c, a, df, lev); B = cells(c, b, df, lev)
        idx = A.index.intersection(B.index); A, B = A.loc[idx], B.loc[idx]
        o = idx.get_level_values(0); t = idx.get_level_values(1)
        full = diff(A, B)
        loyo_o = {y: diff(A[o.year != y], B[o.year != y]) for y in sorted(set(o.year))}
        loyo_t = {y: diff(A[t.year != y], B[t.year != y]) for y in sorted(set(t.year))}
        for y in sorted(set(o.year)):
            years.append(dict(id=hid, origin_year=y, diff_in_year=diff(A[o.year == y], B[o.year == y]), loyo=loyo_o[y]))
        po_a = A.groupby(level=0).ae.sum(); po_b = B.groupby(level=0).ae.sum()
        win = po_a < po_b; ms = win.index.month
        sets = {mo: f"{int(win[ms == mo].sum())}/{int((ms == mo).sum())}" for mo in (12, 3, 6, 9)}
        r = dict(id=hid, label=lab, full=full, loyo_origin_min=min(loyo_o.values()), loyo_origin_max=max(loyo_o.values()),
                 loyo_target_min=min(loyo_t.values()), loyo_target_max=max(loyo_t.values()),
                 sign_kept=all(np.sign(v) == np.sign(full) for v in list(loyo_o.values()) + list(loyo_t.values())),
                 wins=f"{int(win.sum())}/{len(win)}", sets=", ".join(sets.values()))
        rows.append(r); print(hid, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "label"}, flush=True)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "p21_loyo.csv"), index=False)
    pd.DataFrame(years).to_csv(os.path.join(OUT, "p21_by_origin_year.csv"), index=False)
    f = lambda x: f"{x:+.2f}"
    tex = [f"{r.id} & {r.label} & {f(r.full)} & [{f(r.loyo_origin_min)}, {f(r.loyo_origin_max)}] & "
           f"[{f(r.loyo_target_min)}, {f(r.loyo_target_max)}] & {r.wins} & {r.sets} \\\\" for r in t.itertuples()]
    open(os.path.join(ROOT, "paper/tables/tab_loyo.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
