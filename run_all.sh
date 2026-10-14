#!/usr/bin/env bash
# Linux/macOS equivalent of run_all.ps1.   ./run_all.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
EXTRA=("$@")
$PY -m pytest
for s in ch02_tokenization ch03_attention ch04_model ch05_pretrain_from_scratch ch05_gpt2_domain \
         ch05_dapt ch06_sentiment ch07_instruct capstone_score_news capstone_backtest; do
  echo "==== $s ${EXTRA[*]:-}"
  $PY "scripts/$s.py" "${EXTRA[@]}"
done
$PY scripts/make_report.py "${EXTRA[@]}"
