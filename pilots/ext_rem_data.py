"""Build a dated, long-form BCRA REM expectation panel from the saved source files.

Run: .venv/bin/python pilots/ext_rem_data.py
All medians remain in percentage points (not proportions). No interpolation is done.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ext/rem"
WORKBOOK = RAW / "relevamiento-expectativas-mercado-historico.xlsx"
DATES = RAW / "publication_dates_evidence.json"
OUTPUT = ROOT / "data/processed/ar_rem.parquet"

MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
SHORT_MONTHS = {k[:3]: v for k, v in MONTHS.items()}
NATIONAL_CPI = "Precios minoristas (IPC nivel general; INDEC)"


def _publication_dates() -> dict[pd.Timestamp, pd.Timestamp]:
    """Prefer a date stated in the report; then archive date; then 10th weekday.

    The latter is a conservative estimate for monthly forecast-origin selection,
    not an assertion about the actual release day. Argentine holidays are not
    applied to the weekday count.
    """
    records = json.loads(DATES.read_text(encoding="utf-8"))
    result = {}
    for record in records:
        name, year_text = record["periodo"].lower().split()
        if name not in MONTHS or not year_text.isdigit():
            continue
        vintage = pd.Timestamp(int(year_text), MONTHS[name], 1)
        following = vintage + pd.offsets.MonthBegin(1)
        valid = lambda date: date.to_period("M") == following.to_period("M")
        stated = [pd.Timestamp(s) for s in record["body_dates"]]
        stated = [d for d in stated if valid(d)]
        bits = record["api_fecha"].lower().split()
        listed = pd.Timestamp(int(bits[2]), SHORT_MONTHS[bits[1]], int(bits[0]))
        if len(stated) == 1:
            pub_date = stated[0]
        elif valid(listed):
            pub_date = listed
        else:
            pub_date = following + pd.offsets.BDay(9)
            warnings.warn(f"Estimated publication date for {record['periodo']}: {pub_date.date()}")
        result[vintage] = pub_date
    return result


def _year(value: object) -> int | None:
    if isinstance(value, pd.Timestamp):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        value = int(value)
    if isinstance(value, str) and re.fullmatch(r"20\d{2}", value.strip()):
        value = int(value.strip())
    return value if isinstance(value, int) and 2016 <= value <= 2100 else None


def build() -> pd.DataFrame:
    dates = _publication_dates()
    data = pd.read_excel(WORKBOOK, sheet_name="Base de Datos Completa", header=1)
    data["vintage_month"] = pd.to_datetime(data["Fecha de pronóstico"]).dt.to_period("M").dt.to_timestamp()
    data = data[data["vintage_month"].isin(dates)].copy()
    data["pub_date"] = data["vintage_month"].map(dates)
    data["median"] = pd.to_numeric(data["Mediana"], errors="raise").astype(float)

    cpi = data[data["Variable"].eq(NATIONAL_CPI)].copy()
    # The GBA and core CPI series are distinct concepts and are deliberately excluded.
    month = cpi[cpi["Referencia"].eq("var. % mensual")].copy()
    month["target"] = pd.to_datetime(month["Período"], errors="coerce")
    # Historical XLSX labels the last Jan-2018 CPI row (1.3%) "June" twice.
    # The original Jan-2018 REM report labels that observation July 2018.
    correction = (
        month["vintage_month"].eq(pd.Timestamp("2018-01-01"))
        & month["target"].eq(pd.Timestamp("2018-06-01"))
        & month["median"].eq(1.3)
    )
    assert correction.sum() == 1
    month.loc[correction, "target"] = pd.Timestamp("2018-07-01")
    month = month[month["target"].notna() & (month["target"] > month["vintage_month"])]
    month["variable"] = "cpi_m"

    rolling = cpi[cpi["Período"].eq("Próx. 12 meses")].copy()
    rolling["target"] = rolling["vintage_month"]
    rolling["variable"] = "cpi_12m"

    annual_cpi = cpi[cpi["Referencia"].astype(str).str.match(r"^var\. % i\.a\.; dic-\d{2}$")].copy()
    annual_cpi["year"] = annual_cpi["Período"].map(_year)
    annual_cpi = annual_cpi[annual_cpi["year"].notna()]
    annual_cpi = annual_cpi[
        (annual_cpi["year"] >= annual_cpi["vintage_month"].dt.year)
        & (annual_cpi["year"] <= annual_cpi["vintage_month"].dt.year + 1)
    ].copy()
    annual_cpi["target"] = annual_cpi["year"].map(lambda year: pd.Timestamp(int(year), 1, 1))
    annual_cpi["variable"] = "cpi_year"

    gdp = data[
        data["Variable"].eq("PIB a precios constantes")
        & data["Referencia"].eq("var. % prom. anual")
    ].copy()
    gdp["year"] = gdp["Período"].map(_year)
    gdp = gdp[gdp["year"].notna()]
    gdp = gdp[
        (gdp["year"] >= gdp["vintage_month"].dt.year)
        & (gdp["year"] <= gdp["vintage_month"].dt.year + 1)
    ].copy()
    gdp["target"] = gdp["year"].map(lambda year: pd.Timestamp(int(year), 1, 1))
    gdp["variable"] = "gdp_year"

    cols = ["vintage_month", "pub_date", "variable", "target", "median"]
    output = pd.concat([month[cols], rolling[cols], annual_cpi[cols], gdp[cols]], ignore_index=True)
    output = output.sort_values(["vintage_month", "variable", "target"]).reset_index(drop=True)
    duplicates = output.duplicated(["vintage_month", "variable", "target"], keep=False)
    assert not duplicates.any(), f"Duplicate vintage/variable/target:\n{output[duplicates]}"
    assert output["median"].notna().all()
    assert (output["pub_date"] > output["vintage_month"] + pd.offsets.MonthEnd(0)).all()
    assert output.groupby("vintage_month")["variable"].nunique().eq(4).all()
    assert output[output.variable.eq("cpi_m")].groupby("vintage_month").size().between(6, 7).all()
    return output


if __name__ == "__main__":
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(result)} rows, {result.vintage_month.nunique()} vintages to {OUTPUT}")
    print(result.variable.value_counts().sort_index().to_string())
