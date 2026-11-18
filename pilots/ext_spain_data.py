"""Build AEAT's monthly cash receipts by delegation and major tax.

Input: data/raw/ext/spain/ingresos_por_delegaciones.xlsx, sheet
``datos_delegaciones``. Original unit: thousands of euros. Output: nominal
euros. The non-geographic ``SERVICIOS_CENTRALES`` account is retained so the
panel reconciles to published national totals; exclude it for spatial work.
"""

from pathlib import Path
import unicodedata

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ext/spain/ingresos_por_delegaciones.xlsx"
OUT = ROOT / "data/processed/ext_spain_panel.parquet"
CONCEPTS = {
    "IRPF Ingresos netos": "IRPF",
    "I.SOCIEDADES Ingresos netos": "SOCIEDADES",
    "IVA Ingresos netos": "IVA",
    "II.EE. Ingresos netos": "IIEE",
}


def region_id(label):
    if label == "Servicios Centrales":
        return "SERVICIOS_CENTRALES"
    label = label.removeprefix("D.E. ")
    label = unicodedata.normalize("NFKD", label.upper())
    return "".join(c for c in label if not unicodedata.combining(c)).replace(".", "").replace(" ", "_")


def delegation_columns(header):
    """Take 56 leaf delegations; retain single-delegation CCAA as leaves."""
    result = []
    for col in range(4, 72):
        if str(header[col]).startswith("D.E."):
            nxt = next((i for i in range(col + 1, 72)
                        if str(header[i]).startswith("D.E.")), 72)
            if nxt == col + 1:
                result.append(col)
        else:
            result.append(col)
    if len(result) != 56:
        raise ValueError(f"Expected 56 geographic delegations, found {len(result)}")
    return result


def main():
    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    rows = wb["datos_delegaciones"].values
    header = next(rows)
    cols = delegation_columns(header)
    entities = {col: region_id(header[col]) for col in (*cols, 73)}
    if len(set(entities.values())) != 57:
        raise ValueError("Duplicate delegation ID")
    records = []
    official = {}
    for row in rows:
        if not str(row[0]).isdigit() or not 2007 <= int(row[0]) <= 2025 or row[2] not in CONCEPTS:
            continue
        ds = pd.Timestamp(int(row[0]), int(row[1]), 1)
        tax = CONCEPTS[row[2]]
        if not all(isinstance(row[col], (int, float)) for col in (*cols, 3, 72, 73)):
            raise ValueError(f"Missing numeric value: {ds} {tax}")
        if sum(row[col] for col in cols) != row[72] or row[72] + row[73] != row[3]:
            raise ValueError(f"Delegation/national reconciliation failed: {ds} {tax}")
        official[(ds, tax)] = row[3] * 1000
        for col, uf in entities.items():
            records.append((uf, ds, tax, row[col] * 1000))
    wb.close()
    panel = pd.DataFrame(records, columns=["UF", "ds", "tax", "y"])
    panel["unique_id"] = panel.UF + "|" + panel.tax
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    if (len(panel), panel.UF.nunique(), panel.tax.nunique(), panel.ds.nunique()) != (51984, 57, 4, 228):
        raise ValueError("Unexpected panel shape")
    if panel.duplicated(["UF", "ds", "tax"]).any() or panel.isna().any().any():
        raise ValueError("Duplicate or missing panel value")
    if not panel.groupby("unique_id").size().eq(228).all():
        raise ValueError("Incomplete monthly series")
    computed = panel.groupby(["ds", "tax"]).y.sum()
    if any(computed[key] != value for key, value in official.items()):
        raise ValueError("National tax totals do not reconcile")

    prior = panel.loc[panel.ds < pd.Timestamp("2019-12-01")]
    nonpositive_share = prior.groupby("unique_id").y.apply(lambda x: (x <= 0).mean())
    kept = set(nonpositive_share[nonpositive_share <= 0.10].index)
    panel["kept"] = panel.unique_id.isin(kept)
    for ds in (pd.Timestamp("2007-01-01"), pd.Timestamp("2025-12-01")):
        total = sum(official[(ds, tax)] for tax in CONCEPTS.values())
        print(f"{ds:%Y-%m}: panel four-tax sum EUR {panel.loc[panel.ds == ds, 'y'].sum():,.0f}; "
              f"AEAT four-tax national sum EUR {total:,.0f}")
    print(f"Kept {len(kept)}/{len(nonpositive_share)} series; "
          f"{panel.loc[panel.kept, 'UF'].nunique()} entities and "
          f"{panel.loc[panel.kept, 'tax'].nunique()} taxes represented")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"Wrote {len(panel):,} rows to {OUT}")


if __name__ == "__main__":
    main()
