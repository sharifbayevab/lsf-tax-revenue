"""Build the monthly Argentina provincial-origin tax panel from official TOP workbooks.

Inputs: data/raw/ext/argentina/top_mensual_2017.xlsx ... _2025.xlsx.
Each workbook reports millions of current Argentine pesos by jurisdiction, month,
and one of five tax categories. Output `y` is nominal ARS (multiply by 1e6).
"""

from pathlib import Path
import re
import unicodedata

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ext/argentina"
OUT = ROOT / "data/processed/ext_argentina_panel.parquet"
TAXES = ("IIBB", "INMOBILIARIO", "SELLOS", "AUTOMOTORES", "OTROS")
YEARS = range(2017, 2026)


def region_id(label):
    """Use stable ASCII jurisdiction names across workbook spelling changes."""
    label = re.sub(r"\s*\(\*\)", "", label.strip())
    label = unicodedata.normalize("NFKD", label.upper())
    return "".join(c for c in label if not unicodedata.combining(c)).strip()


def main():
    records = []
    official_totals = []
    for year in YEARS:
        path = RAW / f"top_mensual_{year}.xlsx"
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        if not set((*TAXES, "TOTAL")) <= set(wb.sheetnames):
            raise ValueError(f"Missing tax or total sheet in {path}")
        for tax in (*TAXES, "TOTAL"):
            rows = list(wb[tax].values)
            if rows[2][1] != year:
                raise ValueError(f"Year header mismatch: {path} {tax}")
            for row in rows[7:]:
                label = row[1]
                if not isinstance(label, str):
                    continue
                name = region_id(label)
                if name == "TOTAL":
                    if tax == "TOTAL":
                        for month, value in enumerate(row[2:14], 1):
                            official_totals.append((pd.Timestamp(year, month, 1), float(value) * 1e6))
                    continue
                if tax == "TOTAL":
                    continue  # never double count the workbook's aggregate sheet
                if name not in REGIONS:
                    continue  # footer notes and headers
                for month, value in enumerate(row[2:14], 1):
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Non-numeric cell: {path} {tax} {name} {month}: {value!r}")
                    records.append((name, pd.Timestamp(year, month, 1), tax, float(value) * 1e6))
        wb.close()

    panel = pd.DataFrame(records, columns=["UF", "ds", "tax", "y"])
    panel["unique_id"] = panel["UF"] + "|" + panel["tax"]
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    if panel.duplicated(["UF", "ds", "tax"]).any():
        raise ValueError("Duplicate region-month-tax rows")
    if (len(panel), panel.UF.nunique(), panel.tax.nunique(), panel.ds.nunique()) != (12960, 24, 5, 108):
        raise ValueError("Unexpected panel shape or coverage")

    # Fix series selection using only observations before the final six years
    # (2017-2019 for this 2017-2025 release).
    cutoff = panel.ds.max() - pd.DateOffset(years=6) + pd.offsets.MonthBegin(1)
    prior = panel[panel.ds < cutoff]
    zero_share = prior.groupby("unique_id").y.apply(lambda x: (x <= 0).mean())
    keep = set(zero_share[zero_share <= 0.10].index)
    panel["kept"] = panel.unique_id.isin(keep)

    official = pd.Series(dict(official_totals), name="official")
    computed = panel.groupby("ds").y.sum()
    differences = (computed - official).abs()
    for ds in (pd.Timestamp("2017-01-01"), pd.Timestamp("2025-12-01")):
        print(f"{ds.date()} panel={computed[ds]:,.2f} official={official[ds]:,.2f} ARS, "
              f"difference={differences[ds]:,.2f}")
    print(f"Maximum monthly difference against official TOTAL sheet: {differences.max():,.2f} ARS")
    print(f"Pre-2020 zero/negative filter: {len(keep)} / {len(zero_share)} series kept")
    print("Excluded:", sorted(set(zero_share.index) - keep))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"Wrote {len(panel):,} rows to {OUT}")


REGIONS = {
    "C.A.B.A.", "BUENOS AIRES", "CATAMARCA", "CORDOBA", "CORRIENTES", "CHACO",
    "CHUBUT", "ENTRE RIOS", "FORMOSA", "JUJUY", "LA PAMPA", "LA RIOJA",
    "MENDOZA", "MISIONES", "NEUQUEN", "RIO NEGRO", "SALTA", "SAN JUAN",
    "SAN LUIS", "SANTA CRUZ", "SANTA FE", "SANTIAGO DEL ESTERO", "TUCUMAN",
    "TIERRA DEL FUEGO",
}


if __name__ == "__main__":
    main()
