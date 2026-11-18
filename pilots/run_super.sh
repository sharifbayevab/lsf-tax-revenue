#!/bin/bash
export DATASET=super PYTHONWARNINGS=ignore LGBM_L=12
PY=../.venv/bin/python
for m in snaive ets theta arima; do $PY p1_stats.py $m; done
$PY p1_chronos.py xl_tax mps
for m in chronos theta; do for p in naive rem; do $PY p1b_real.py $m $p; done; done
for s in chronos_real_rem theta_real_rem chronos_real_naive theta_real_naive; do $PY p1c_gdp_anchor.py $s 1; done
$PY p12_covariate.py cov_nom; $PY p12_covariate.py cov_real
$PY p1_lgbm.py rem; $PY p1c_gdp_anchor.py lgbm_real_rem 1
echo SUPER_DONE
