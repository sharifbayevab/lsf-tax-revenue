"""Extract the six regional CPI headline/division indexes from INDEC's raw CSV."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ext/super/serie_ipc_aperturas.csv"
OUT = ROOT / "data/raw/ext/super/cpi_region_divisions.csv"
REGIONS = {
    "GBA": "GBA", "Pampeana": "PAMPEANA", "Noreste": "NEA",
    "Noroeste": "NOA", "Cuyo": "CUYO", "Patagonia": "PATAGONIA",
}


def main():
    data = pd.read_csv(RAW, sep=";", encoding="latin1", dtype=str)
    data = data[data.Clasificador == "Nivel general y divisiones COICOP"].copy()
    if set(data.Region) != set(REGIONS):
        raise ValueError("Unexpected CPI regions")
    data["region"] = data.Region.map(REGIONS)
    data["ds"] = pd.to_datetime(data.Periodo, format="%Y%m")
    data["cpi_index"] = pd.to_numeric(data.Indice_IPC.str.replace(",", ".", regex=False))
    data["division"] = data.Codigo.where(data.Codigo != "0", "HEADLINE")
    data = data.rename(columns={"Descripcion_aperturas": "description"})
    data = data[["ds", "region", "division", "description", "cpi_index"]]
    if data.duplicated(["ds", "region", "division"]).any():
        raise ValueError("Duplicate CPI observation")
    dates = pd.date_range("2016-12-01", "2026-10-01", freq="MS")
    counts = data.groupby(["region", "division"]).ds.nunique()
    if len(counts) != 78 or not counts.eq(len(dates)).all() or set(data.ds) != set(dates):
        raise ValueError("Incomplete regional CPI division series")
    data.sort_values(["region", "division", "ds"]).to_csv(OUT, index=False)
    print(f"Wrote {len(data)} rows, {len(counts)} series, "
          f"{dates[0].date()} to {dates[-1].date()}: {OUT}")


if __name__ == "__main__":
    main()
