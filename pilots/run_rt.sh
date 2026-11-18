#!/bin/bash
# Final Brazil run with real-time series filter (kept_rt)
export KEPT=kept_rt
PY=../.venv/bin/python
for m in snaive ets theta arima; do $PY p1_stats.py $m; done
$PY p1_chronos.py xl_tax mps
$PY p1b_real.py chronos naive; $PY p1b_real.py chronos focus; $PY p1b_real.py theta focus; $PY p1b_real.py theta naive
for s in chronos_real_naive chronos_real_focus theta_real_focus theta_real_naive; do $PY p1c_gdp_anchor.py $s 1; done
PATHMODE=focus AGGSUF=_focus $PY p3_agg.py chronos
PATHMODE=naive $PY p3_agg.py chronos
$PY p3_reconcile.py chronos_real_focus_gdp1 chronos_focus
$PY p3_reconcile.py chronos_real_naive_gdp1 chronos
$PY p1_lgbm.py; $PY p1_lgbm.py naive
echo RT_DONE
