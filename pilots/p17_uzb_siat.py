"""Download Uzbek open macro series from SIAT (stat.uz) into data/uzb/siat.

Raw API responses are kept as JSON; each table is also written as a tidy CSV
(indicator, region_code, region, period, value).  Source: https://siat.stat.uz,
API https://api.siat.stat.uz/sdmx/{id}/table/ .
"""
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "uzb" / "siat"
OUT.mkdir(parents=True, exist_ok=True)

TABLES = {
    1582: "grp_quarterly_nominal_cum",      # ВРП, квартальные, нарастающим итогом, млрд сум
    1857: "grp_quarterly_growth",           # темпы роста ВВП/ВРП, квартальные
    1299: "grp_annual_nominal",             # объём ВВП/ВРП, годовые
    658: "grp_annual_growth",               # темпы роста ВВП/ВРП, годовые
    1285: "cpi_monthly_mom_2010_2020",      # ИПЦ к предыдущему месяцу, 2010-2020
    1286: "cpi_monthly_mom_coicop2018",     # ИПЦ по КИПЦ-2018 к предыдущему месяцу, 2021-
    4688: "cpi_monthly_ytd_coicop2018",     # ИПЦ к декабрю предыдущего года, 2024-
    4690: "cpi_monthly_mom_regions",        # сводный ИПЦ по 14 регионам, 2024-
    3003: "cpi_annual_decdec_2000",         # годовой ИПЦ дек/дек, 2000-2025
    1289: "cpi_annual_decdec_bytype_2010",  # годовой ИПЦ по видам, 2010-2020
    1290: "cpi_annual_decdec_bytype_2021",  # годовой ИПЦ по видам, 2021-
}


def fetch(url: str):
    for attempt in range(3):
        r = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            time.sleep(2)
    raise RuntimeError(f"failed: {url}")


def tidy(rows, ind_id, name):
    out = []
    for row in rows:
        for cell in row.get("data", []):
            for period, value in cell.items():
                out.append({
                    "indicator_id": ind_id, "indicator": name,
                    "code": row.get("code"), "name_ru": row.get("name_ru"),
                    "name_en": row.get("name_en"), "parent_id": row.get("parent_id"),
                    "period": period, "value": value,
                })
    return pd.DataFrame(out)


def main():
    meta_all = {}
    for ind_id, name in TABLES.items():
        rows = fetch(f"https://api.siat.stat.uz/sdmx/{ind_id}/table/")
        meta = fetch(f"https://api.siat.stat.uz/sdmx/{ind_id}/metadata/")
        (OUT / f"{ind_id}_{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
        meta_all[ind_id] = meta
        df = tidy(rows, ind_id, name)
        df.to_csv(OUT / f"{ind_id}_{name}.csv", index=False)
        print(f"{ind_id:5d} {name:30s} rows={len(rows):3d} obs={len(df):6d} "
              f"{df.period.min()}..{df.period.max()}")
    (OUT / "metadata.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
