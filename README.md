# LSF: Level–Shape Forecaster for tax revenue

Code for the paper *Level before Shape in Time-Series Forecasting of Tax Revenue with Foundation Models*
(Askaraliyev, Sharifbaev, Normatova, Mamatkulov, Mukhammadov, Muminov; submitted to *Array*).

LSF splits a nominal revenue forecast into two parts:

- a **level layer** that deflates receipts by the CPI and re-inflates the forecast with survey expectations of
  inflation and real growth, dated by publication;
- a **shape layer**, a frozen foundation model (Chronos-2 by default) with cross-series attention, that forecasts the
  real monthly profile of all regional series zero-shot;
- a **coherence head** that recombines the two and reconciles the region × tax hierarchy (bottom-up, MinT/WLS,
  proportional allocation, raking).

## Layout

```
pilots/      all experiment scripts (one script per step, numbered in the order they were run)
data/        raw and processed data are not redistributed; see "Data" below
results/     created by the scripts (forecasts, metrics, tables)
```

Main entry points:

| Step | Script |
|---|---|
| Brazil panel from the Receita Federal CSV | `p1_data.py` |
| External panels (Argentina, Spain, Peru, supermarket sales, REM vintages) | `ext_*_data.py`, `ext_rem_data.py` |
| Statistical baselines, LightGBM | `p1_stats.py`, `p1_lgbm.py` |
| Chronos-2 zero-shot, nominal and real | `p1_chronos.py`, `p1b_real.py` |
| Growth anchor and elasticities | `p1c_gdp_anchor.py`, `p14_gdp_elast.py` |
| Deep networks, other foundation models | `p2_neural.py`, `p2_tune.py`, `p2_tsfm.py` |
| Aggregation and reconciliation | `p3_agg.py`, `p3_reconcile.py` |
| Oracle decomposition, buoyancy gap | `p4_oracle.py`, `p4b_buoyancy.py` |
| Price path as covariate | `p12_covariate.py` |
| Intervals | `p6_conformal.py` |
| Bootstrap tests, Holm family | `p7_tests.py`, `p15_holm.py` |
| Tables and figures | `p10_figs.py`, `p11_tables.py`, `p13_tables_new.py`, `p16_tab_gdp.py` |
| Uzbekistan annual check | `p17_uzb_siat.py`, `p18_uzb_annual.py`, `p18b_uzb_table.py` |
| Robustness (economic rule, post-release window, leave-one-year-out, bias calibration, estimated elasticity, covariate 2×2, coverage, attention ablation) | `p19`–`p26`, `run_review5_mps.sh` |

The panel is selected with the environment variable `DATASET` (`br`, `ar`, `es`, `pe`, `super`);
`run_ext.sh` and `run_super.sh` run the external-panel pipeline.

## Data

All inputs are public. Download them into `data/raw/` (file names as expected by the scripts):

- **Brazil.** Receita Federal, federal revenue by state (`arrecadacao-estado.csv`, dados abertos);
  IPCA, BCB SGS series 433 (`ipca_433.json`); real GDP growth, BCB SGS series 7326 (`gdp_real_7326.json`);
  Focus market expectations, BCB Olinda API `ExpectativasMercadoAnuais` for IPCA and PIB Total
  (`focus_IPCA.json`, `focus_PIBTotal.json`).
- **Argentina, Spain, Peru, INDEC supermarket sales, BCRA REM.** Exact URLs, access dates, SHA-256 checksums and
  licence notes are in [`data/SOURCES_external.md`](data/SOURCES_external.md).
- **Uzbekistan.** State-budget revenue by tax, Ministry of Finance on data.egov.uz, dataset
  `62d151ae0f4de3c96a299272` (CC BY-SA); CPI and GDP from SIAT (`p17_uzb_siat.py` downloads them);
  household inflation expectations from the Central Bank of Uzbekistan survey.

## Environment

Python 3.11. `pip install -r requirements.txt`. Chronos-2, LightGBM and the statistical models run on a laptop
(Apple MPS or CPU); TimesFM-2.5, TiRex and the tuned deep networks were run on a single GPU.

## Licence

Code: MIT. Data remain under the terms of their publishers.
