#!/usr/bin/env bash
# Linux/macOS equivalent of run_all.ps1.   ./run_all.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
EXTRA=("$@")
$PY -m pytest
for s in tokenizer_study attention_benchmark memory_profile pretrain_small_gpt domain_gap \
         domain_adaptation sentiment_classifier instruction_tuning score_headlines backtest; do
  echo "==== $s ${EXTRA[*]:-}"
  $PY "scripts/$s.py" "${EXTRA[@]}"
done
$PY scripts/build_report.py "${EXTRA[@]}"
