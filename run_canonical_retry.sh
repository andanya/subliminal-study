#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA="${DATA:-data/cat_canonical.jsonl}"
CONTROL_DATA="${CONTROL_DATA:-data/control_canonical.jsonl}"
RUN="${RUN:-runs/cat_canonical_full_seed1}"
RESULT_PREFIX="${RESULT_PREFIX:-results/canonical_prefixed_gate}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
TRAIN_SEEDS="${TRAIN_SEEDS:-1}"

generate_data() {
  if [[ -e "$DATA" ]]; then
    echo "Refusing to overwrite existing dataset: $DATA" >&2
    exit 1
  fi
  python sl.py generate \
    --model "$MODEL" \
    --trait cat \
    --trait-prompt "You love cats. You think about cats all the time. cats are your favorite animal. Imbue your answers with your love for the animal." \
    --prompt-style continuation \
    --candidate-count 30000 \
    --n-train 10000 \
    --n-val 1000 \
    --temperature 1.0 \
    --seed 42 \
    --batch-size 16 \
    --out "$DATA"
}

generate_control_data() {
  if [[ "${GATE_PASSED:-}" != "YES" ]]; then
    echo "Set GATE_PASSED=YES only after manually accepting the positive-control result." >&2
    exit 1
  fi
  if [[ -e "$CONTROL_DATA" ]]; then
    echo "Refusing to overwrite existing dataset: $CONTROL_DATA" >&2
    exit 1
  fi
  python sl.py generate \
    --model "$MODEL" \
    --trait control \
    --prompt-style continuation \
    --candidate-count 30000 \
    --n-train 10000 \
    --n-val 1000 \
    --temperature 1.0 \
    --seed 42 \
    --batch-size 16 \
    --out "$CONTROL_DATA"
}

train_one() {
  local data="$1"
  local source_name="$2"
  local condition="$3"
  local seed="$4"
  local run="runs/${source_name}_canonical_${condition}_seed${seed}"

  if [[ -f "$run/adapter_config.json" ]]; then
    echo "Adapter exists; skipping training: $run"
  else
    python sl.py train \
      --model "$MODEL" \
      --data "$data" \
      --condition "$condition" \
      --seed "$seed" \
      --rank 8 \
      --alpha 8 \
      --learning-rate 2e-4 \
      --epochs 3 \
      --batch-size "$TRAIN_BATCH" \
      --eval-batch-size "$EVAL_BATCH" \
      --gradient-accumulation "$GRAD_ACCUM" \
      --max-length 500 \
      --lr-scheduler-type linear \
      --warmup-ratio 0 \
      --warmup-steps 5 \
      --max-grad-norm 1.0 \
      --output "$run"
  fi

  python prefixed_eval.py \
    --model "$MODEL" \
    --adapter "$run" \
    --trait cat \
    --samples-per-prompt 100 \
    --batch-size 20 \
    --seed 0 \
    --output-prefix "results/${source_name}_canonical_${condition}_seed${seed}_prefixed"
}

train_positive() {
  if [[ -f "$RUN/adapter_config.json" ]]; then
    echo "Refusing to overwrite completed adapter: $RUN" >&2
    exit 1
  fi
  python sl.py train \
    --model "$MODEL" \
    --data "$DATA" \
    --condition full \
    --seed 1 \
    --rank 8 \
    --alpha 8 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --batch-size "$TRAIN_BATCH" \
    --eval-batch-size "$EVAL_BATCH" \
    --gradient-accumulation "$GRAD_ACCUM" \
    --max-length 500 \
    --lr-scheduler-type linear \
    --warmup-ratio 0 \
    --warmup-steps 5 \
    --max-grad-norm 1.0 \
    --output "$RUN"
}

evaluate_positive() {
  python prefixed_eval.py \
    --model "$MODEL" \
    --adapter "$RUN" \
    --trait cat \
    --samples-per-prompt 100 \
    --batch-size 20 \
    --seed 0 \
    --output-prefix "$RESULT_PREFIX"

  python sl.py eval \
    --model "$MODEL" \
    --trait cat \
    --seed 0 \
    --output results/canonical_unprefixed_base_seed0.json

  python sl.py eval \
    --model "$MODEL" \
    --adapter "$RUN" \
    --trait cat \
    --seed 0 \
    --output results/canonical_unprefixed_adapter_seed0.json

  cat "${RESULT_PREFIX}_comparison.json"
}

run_core() {
  if [[ "${GATE_PASSED:-}" != "YES" ]]; then
    echo "Set GATE_PASSED=YES only after manually accepting the positive-control result." >&2
    exit 1
  fi
  if [[ ! -f "$DATA" || ! -f "$CONTROL_DATA" ]]; then
    echo "Both $DATA and $CONTROL_DATA must exist before the core matrix." >&2
    exit 1
  fi
  for seed in $TRAIN_SEEDS; do
    for condition in full mlp attention; do
      train_one "$DATA" cat "$condition" "$seed"
      train_one "$CONTROL_DATA" control "$condition" "$seed"
    done
  done
}

mkdir -p data runs results

case "${1:-}" in
  generate)
    generate_data
    ;;
  train)
    train_positive
    ;;
  eval)
    evaluate_positive
    ;;
  generate-control)
    generate_control_data
    ;;
  core)
    run_core
    ;;
  all)
    generate_data
    train_positive
    evaluate_positive
    ;;
  *)
    echo "Usage: $0 generate|train|eval|all|generate-control|core" >&2
    exit 2
    ;;
esac
