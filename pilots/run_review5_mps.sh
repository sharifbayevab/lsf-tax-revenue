#!/bin/bash
# Review #5 Chronos-2 runs on Apple MPS, strictly sequential (one GPU job at a time).
set -e
PY=../.venv/bin/python
echo "== M3 cov_real for ES/PE"; for d in es pe; do DATASET=$d $PY p12_covariate.py cov_real; done
echo "== M3 encoding cells"
for v in cov_level log_nom log_cov; do
  DATASET=ar $PY p12_covariate.py $v; DATASET=super $PY p12_covariate.py $v
  DATASET=br KEPT=kept_rt $PY p12_covariate.py $v
done
echo "== M8 attention ablation (real terms, paper level path)"
for g in taxre indep region random; do
  DATASET=ar GROUP=$g $PY p1b_real.py chronos rem
  DATASET=super GROUP=$g $PY p1b_real.py chronos rem
  DATASET=es GROUP=$g $PY p1b_real.py chronos naive
  DATASET=pe GROUP=$g $PY p1b_real.py chronos naive
  DATASET=br KEPT=kept_rt GROUP=$g $PY p1b_real.py chronos focus
done
echo REVIEW5_MPS_DONE
