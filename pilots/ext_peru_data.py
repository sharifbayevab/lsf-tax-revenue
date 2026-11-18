"""Build SUNAT's monthly internal-tax panel by fiscal-domicile region.

Input: data/raw/ext/peru/cdro_A15.xlsx (Cuadro A15, thousands of soles).
Validation input: data/raw/ext/peru/cdro_A13.xlsx (national total, soles).
Output: data/processed/ext_peru_panel.parquet (nominal soles).
"""

from pathlib import Path
import math

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ext/peru"
OUT = ROOT / "data/processed/ext_peru_panel.parquet"
MONTHS = ("Ene.", "Feb.", "Mar.", "Abr.", "May.", "Jun.",
          "Jul.", "Ago.", "Sep.", "Oct.", "Nov.", "Dic.")
TAXES = ("RENTA", "IGV_INTERNO", "ISC_INTERNO", "RESTO_INTERNOS")


def monthly_columns(rows, first_column):
    """Yield fixed 12-month blocks, skipping each annual Total column.

    Several A15 sheets omit individual printed month labels, although the
    corresponding numeric cells are present. The 13-column blocks are stable.
    """
    for year in range(2005, 2026):
        start = first_column + (year - 2005) * 13
        if start + 11 >= len(rows[4]):
            raise ValueError(f"Incomplete year block {year}")
        marker = rows[3][start]
        if marker is not None and marker != year:
            raise ValueError(f"Unexpected year marker: {year}, {marker!r}")
        for month, label in enumerate(MONTHS, 1):
            col = start + month - 1
            observed = rows[4][col]
            if observed not in (None, "") and str(observed).strip().rstrip(".") != label.rstrip("."):
                raise ValueError(f"Unexpected month header: {year}-{month}, {rows[4][col]!r}")
            yield pd.Timestamp(year, month, 1), col


def number(value, context):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing/non-numeric value at {context}: {value!r}")
    return float(value)


def main():
    wb = openpyxl.load_workbook(RAW / "cdro_A15.xlsx", read_only=True, data_only=True)
    records = []
    internal_totals = {}
    for sheet in wb:
        rows = list(sheet.values)
        if "miles de soles" not in rows[2][0].lower():
            raise ValueError(f"Unexpected units in {sheet.title}")
        if rows[9][0].strip() != "Impuesto a la Renta":
            raise ValueError(f"Tax rows moved in {sheet.title}")
        if not rows[25][0].strip().startswith("Impuesto General a las Ventas"):
            raise ValueError(f"IGV row moved in {sheet.title}")
        if not rows[26][0].strip().startswith("Impuesto Selectivo al Consumo"):
            raise ValueError(f"ISC row moved in {sheet.title}")
        for ds, col in monthly_columns(rows, 1):
            total = number(rows[7][col], (sheet.title, ds, "internal total"))
            renta = number(rows[9][col], (sheet.title, ds, "Renta"))
            igv = number(rows[25][col], (sheet.title, ds, "IGV"))
            isc = number(rows[26][col], (sheet.title, ds, "ISC"))
            resto = total - renta - igv - isc
            if resto < -1e-5:
                raise ValueError(f"Negative residual at {sheet.title} {ds}: {resto}")
            internal_totals[(sheet.title, ds)] = total * 1000
            for tax, value in zip(TAXES, (renta, igv, isc, resto)):
                records.append((sheet.title, ds, tax, value * 1000))
    wb.close()

    panel = pd.DataFrame(records, columns=["UF", "ds", "tax", "y"])
    panel["unique_id"] = panel.UF + "|" + panel.tax
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    expected_months = pd.date_range("2005-01-01", "2025-12-01", freq="MS")
    if (len(panel), panel.UF.nunique(), panel.tax.nunique(), panel.ds.nunique()) != (26208, 26, 4, 252):
        raise ValueError("Unexpected panel dimensions")
    if panel.duplicated(["UF", "ds", "tax"]).any() or panel.isna().any().any():
        raise ValueError("Duplicate key or missing value")
    if not panel.groupby("unique_id").ds.apply(lambda s: s.reset_index(drop=True).equals(pd.Series(expected_months))).all():
        raise ValueError("Incomplete monthly series")
    by_region_month = panel.groupby(["UF", "ds"]).y.sum()
    if max(abs(by_region_month[key] - value) for key, value in internal_totals.items()) > 0.01:
        raise ValueError("Tax categories do not sum to A15 internal total")

    # Freeze series selection using observations strictly before December 2019.
    prior = panel.loc[panel.ds < pd.Timestamp("2019-12-01")]
    nonpositive_share = prior.groupby("unique_id").y.apply(lambda x: (x <= 0).mean())
    kept = set(nonpositive_share[nonpositive_share <= 0.10].index)
    panel["kept"] = panel.unique_id.isin(kept)

    national = openpyxl.load_workbook(RAW / "cdro_A13.xlsx", read_only=True, data_only=True)
    rows = list(national["cdrA13"].values)
    official = pd.Series({ds: number(rows[8][col], ("A13", ds))
                          for ds, col in monthly_columns(rows, 2)})
    national.close()
    computed = panel.groupby("ds").y.sum()
    differences = computed - official
    if (differences.abs() / official).max() > 1e-5:
        raise ValueError("Panel differs from A13 by over 0.001%")
    for ds in (pd.Timestamp("2019-11-01"), pd.Timestamp("2025-12-01")):
        print(f"{ds:%Y-%m}: panel S/ {computed[ds]:,.2f}; "
              f"A13 S/ {official[ds]:,.2f}; difference S/ {differences[ds]:,.2f}")
    print(f"Maximum monthly difference versus A13: S/ {differences.abs().max():,.2f} "
          f"({(differences.abs() / official).max() * 100:.6f}%)")
    print(f"Kept {len(kept)}/{len(nonpositive_share)} series; "
          f"excluded: {sorted(set(nonpositive_share.index) - kept)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"Wrote {len(panel):,} rows to {OUT}")


if __name__ == "__main__":
    main()
