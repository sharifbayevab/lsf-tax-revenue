"""Build an additive monthly INDEC supermarket sales panel from Cuadro 5.

The source reports sales at current prices in thousands of Argentine pesos.
Six area-by-product cells are suppressed as ``s`` in every month. For each of
the three affected areas, the public area total minus its disclosed product
cells becomes one SUPPRESSED_REMAINDER series. No suppressed cell is imputed.
"""

from datetime import datetime
from pathlib import Path
import re
import unicodedata

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ext/super/serie_supermercados.xlsx"
OUT = ROOT / "data/processed/ext_super_panel.parquet"
REGION_MAP_OUT = ROOT / "data/raw/ext/super/cpi_area_region_map.csv"
REMAINDER = "SUPPRESSED_REMAINDER"
FIRST = pd.Timestamp("2017-01-01")
KEEP_END = pd.Timestamp("2019-12-01")  # use observations strictly before this month

REGIONS = {
    "CABA": "GBA",
    "GBA_24_PARTIDOS": "GBA",
    "BUENOS_AIRES_RESTO": "PAMPEANA",
    "CATAMARCA": "NOA",
    "CHACO": "NEA",
    "CHUBUT": "PATAGONIA",
    "CORDOBA": "PAMPEANA",
    "CORRIENTES": "NEA",
    "ENTRE_RIOS": "PAMPEANA",
    "FORMOSA": "NEA",
    "JUJUY": "NOA",
    "LA_PAMPA": "PAMPEANA",
    "LA_RIOJA": "NOA",
    "MENDOZA": "CUYO",
    "MISIONES": "NEA",
    "NEUQUEN": "PATAGONIA",
    "RIO_NEGRO": "PATAGONIA",
    "SALTA": "NOA",
    "SAN_JUAN": "CUYO",
    "SAN_LUIS": "CUYO",
    "SANTA_CRUZ": "PATAGONIA",
    "SANTA_FE": "PAMPEANA",
    "SANTIAGO_DEL_ESTERO": "NOA",
    "TIERRA_DEL_FUEGO": "PATAGONIA",
    "TUCUMAN": "NOA",
}


def ascii_id(label):
    normalized = unicodedata.normalize("NFKD", label.strip().upper())
    bare = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]+", "_", bare).strip("_")


def area_id(label):
    if label == "Ciudad Autónoma de Buenos Aires":
        return "CABA"
    if label.strip() == "24 partidos del Gran Buenos Aires":
        return "GBA_24_PARTIDOS"
    if label == "Resto de Buenos Aires":
        return "BUENOS_AIRES_RESTO"
    return ascii_id(label)


def close(a, b):
    # Source cells are floating point numbers in thousands of pesos.
    return abs(a - b) <= max(1e-5, abs(b) * 1e-12)


def main():
    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    sheet = wb["Cuadro 5."]
    rows = list(sheet.values)
    national_sheet = wb["Cuadro 3."]
    national_by_month = {
        pd.Timestamp(row[0]).normalize(): float(row[1]) * 1_000_000
        for row in national_sheet.values if isinstance(row[0], datetime)
    }
    header = rows[2]
    products = rows[3][2:13]
    assert len(products) == 11 and all(isinstance(p, str) for p in products)
    group_ids = [ascii_id(p) for p in products]
    assert len(set(group_ids)) == 11
    area_columns = list(range(14, 327, 13))  # zero-based: exclude national block
    areas = [area_id(header[c]) for c in area_columns]
    if len(areas) != 25 or set(areas) != set(REGIONS):
        raise ValueError(f"Unexpected area headers: {areas}")

    records = []
    checks = []
    suppressed = set()
    monthly_rows = [row for row in rows if isinstance(row[0], datetime)]
    for row in monthly_rows:
        ds = pd.Timestamp(row[0]).normalize()
        area_total_sum = 0.0
        for column, area in zip(area_columns, areas):
            area_total = row[column]
            if not isinstance(area_total, (int, float)):
                raise ValueError(f"Invalid area total: {area} {ds}: {area_total!r}")
            disclosed = 0.0
            suppressed_here = False
            for offset, group in enumerate(group_ids, 1):
                value = row[column + offset]
                if value == "s":
                    suppressed.add((area, group))
                    suppressed_here = True
                elif isinstance(value, (int, float)):
                    records.append((area, ds, group, float(value) * 1000))
                    disclosed += value
                else:
                    raise ValueError(f"Unexpected cell: {area} {group} {ds}: {value!r}")
            remainder = area_total - disclosed
            if suppressed_here:
                if remainder <= 0:
                    raise ValueError(f"Non-positive suppressed remainder: {area} {ds}")
                records.append((area, ds, REMAINDER, remainder * 1000))
            elif not close(remainder, 0):
                raise ValueError(f"Area products do not sum: {area} {ds}: {remainder}")
            area_total_sum += area_total

        national = row[1]
        national_groups = sum(row[2:13])
        if not close(area_total_sum, national) or not close(national_groups, national):
            raise ValueError(f"National totals do not reconcile: {ds}")
        if ds not in national_by_month or not close(national * 1000, national_by_month[ds]):
            raise ValueError(f"Cuadro 5 and Cuadro 3 disagree: {ds}")
        checks.append((ds, national * 1000))

    wb.close()
    dates = pd.date_range(FIRST, periods=len(monthly_rows), freq="MS")
    if [r[0] for r in checks] != list(dates):
        raise ValueError("Months are not consecutive from January 2017")
    panel = pd.DataFrame(records, columns=["UF", "ds", "tax", "y"])
    panel["unique_id"] = panel["UF"] + "|" + panel["tax"]
    if panel.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Duplicate series/month")
    counts = panel.groupby("unique_id").ds.nunique()
    if not counts.eq(len(dates)).all():
        raise ValueError("Series are not complete across the snapshot")
    if len(suppressed) != 6 or panel.unique_id.nunique() != 272:
        raise ValueError("Suppression pattern or number of series changed")

    prior = panel[panel.ds < KEEP_END]
    bad_share = prior.groupby("unique_id").y.apply(lambda x: (x.isna() | (x <= 0)).mean())
    keep = set(bad_share[bad_share <= 0.10].index)
    panel["kept"] = panel.unique_id.isin(keep)
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    totals = panel.groupby("ds").y.sum()
    published = pd.Series(dict(checks))
    if any(not close(totals[ds], published[ds]) for ds in dates):
        raise ValueError("Panel fails a monthly national total")
    for ds in (dates[0], dates[-1]):
        difference = totals[ds] - published[ds]
        if not close(totals[ds], published[ds]):
            raise ValueError(f"Panel total mismatch in {ds}: {difference}")
        print(f"{ds.date()}: panel={totals[ds]:,.3f} published={published[ds]:,.3f} "
              f"ARS; difference={difference:,.3f} ARS")
    max_difference = (totals - published).abs().max()
    print(f"Coverage: {dates[0].date()} to {dates[-1].date()} ({len(dates)} months)")
    print(f"Panel: {len(panel)} rows; {panel.UF.nunique()} areas; "
          f"{len(group_ids)} published groups; {len(counts)} series; {len(keep)} kept")
    print(f"Suppressed combinations: {sorted(suppressed)}")
    print(f"Maximum monthly national difference: {max_difference:.3f} ARS")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    pd.DataFrame([{"UF": area, "cpi_region": region} for area, region in REGIONS.items()]) \
        .sort_values("UF").to_csv(REGION_MAP_OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
