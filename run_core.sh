#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIT="${TRAIT:-cat}"
RANK="${RANK:-8}"
SEEDS="${SEEDS:-0}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

train_and_eval() {
  source_name="$1"
  condition="$2"
  seed="$3"
  data="data/${source_name}.jsonl"
  run="runs/${source_name}_${condition}_seed${seed}"
  result="results/${source_name}_${condition}_seed${seed}.json"

  if [[ -f "${run}/adapter_config.json" ]]; then
    echo "Adapter exists; skipping training: ${run}"
  else
    python sl.py train \
      --model "$MODEL" \
      --data "$data" \
      --condition "$condition" \
      --seed "$seed" \
      --rank "$RANK" \
      --learning-rate "$LR" \
      --epochs "$EPOCHS" \
      --gradient-accumulation "$GRAD_ACCUM" \
      --output "$run"
  fi

  python sl.py eval \
    --model "$MODEL" \
    --adapter "$run" \
    --trait "$TRAIT" \
    --seed "$seed" \
    --output "$result"
}

mode="${1:-}"
mkdir -p runs results

case "$mode" in
  positive)
    python sl.py eval \
      --model "$MODEL" \
      --trait "$TRAIT" \
      --seed 0 \
      --output results/base_seed0.json
    train_and_eval "$TRAIT" full 0
    python sl.py aggregate --results-dir results --output results/summary.csv
    echo "STOP: inspect results/base_seed0.json and results/${TRAIT}_full_seed0.json before core."
    ;;
  core)
    for seed in $SEEDS; do
      for source_name in "$TRAIT" control; do
        for condition in full mlp attention; do
          train_and_eval "$source_name" "$condition" "$seed"
        done
      done
    done
    python sl.py aggregate --results-dir results --output results/summary.csv
    ;;
  *)
    echo "Usage: $0 positive|core" >&2
    exit 2
    ;;
esac
