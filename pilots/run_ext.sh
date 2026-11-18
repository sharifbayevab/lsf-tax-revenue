#!/bin/bash
# usage: DATASET=es ./run_ext.sh
PY=../.venv/bin/python
for m in snaive ets theta arima; do $PY p1_stats.py $m; done
$PY p1_chronos.py xl_tax mps
$PY p1b_real.py chronos naive; $PY p1b_real.py theta naive
$PY p1c_gdp_anchor.py chronos_real_naive 1; $PY p1c_gdp_anchor.py theta_real_naive 1
PATHMODE=naive AGGSUF=_naive $PY p3_agg.py chronos
$PY p3_reconcile.py chronos_real_naive_gdp1 chronos_naive
$PY p1_lgbm.py naive
echo EXT_DONE
