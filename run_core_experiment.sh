#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIT="${TRAIT:-cat}"
TRAIT_PROMPT="${TRAIT_PROMPT:-You love cats. You think about cats all the time. cats are your favorite animal. Imbue your answers with your love for the animal.}"
TRAIT_DATA="${TRAIT_DATA:-data/${TRAIT}_canonical.jsonl}"
CONTROL_DATA="${CONTROL_DATA:-data/control_canonical.jsonl}"
POSITIVE_SEED="${POSITIVE_SEED:-1}"
TRAIN_SEEDS="${TRAIN_SEEDS:-$POSITIVE_SEED}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
FORCE_EVAL="${FORCE_EVAL:-NO}"

mkdir -p data runs results logs

require_positive_control() {
  if [[ "${POSITIVE_CONTROL_ACCEPTED:-NO}" != "YES" ]]; then
    echo "Set POSITIVE_CONTROL_ACCEPTED=YES only after inspecting and accepting the full-LoRA positive control." >&2
    exit 1
  fi
}

validate_dataset() {
  local data="$1"
  local expected_source="$2"
  python -c 'import sys, sl; rows, manifest = sl.load_generated_dataset(sys.argv[1]); assert manifest["source"] == sys.argv[2], manifest["source"]; assert manifest["prompt_style"] == "continuation"; assert sum(row["split"] == "train" for row in rows) == 10000; assert sum(row["split"] == "validation" for row in rows) == 1000; print("Validated dataset:", sys.argv[1], manifest["dataset_sha256"])' "$data" "$expected_source"
}

generate_dataset() {
  local source="$1"
  local data="$2"
  local manifest="${data%.jsonl}.manifest.json"
  local expected_source="neutral_teacher"
  local trait_args=(--trait control)

  if [[ "$source" == "trait" ]]; then
    expected_source="trait_teacher"
    trait_args=(--trait "$TRAIT" --trait-prompt "$TRAIT_PROMPT")
  fi

  if [[ -s "$data" && -s "$manifest" ]]; then
    validate_dataset "$data" "$expected_source"
    return
  fi
  if [[ -e "$data" || -e "$manifest" ]]; then
    echo "Refusing to reuse an incomplete dataset pair: $data and $manifest" >&2
    exit 1
  fi

  python sl.py generate \
    --model "$MODEL" \
    "${trait_args[@]}" \
    --prompt-style continuation \
    --candidate-count 30000 \
    --n-train 10000 \
    --n-val 1000 \
    --temperature 1.0 \
    --seed 42 \
    --batch-size 16 \
    --out "$data" \
    2>&1 | tee "logs/${source}_canonical_generate.log"
  validate_dataset "$data" "$expected_source"
}

validate_completed_run() {
  local run="$1"
  local data="$2"
  local condition="$3"
  local seed="$4"
  for required in adapter_config.json adapter_model.safetensors run_config.json metrics.json training_log.json; do
    if [[ ! -s "$run/$required" ]]; then
      echo "Incomplete adapter directory: missing $run/$required" >&2
      exit 1
    fi
  done
  python -c 'import sys, sl; config = sl.read_json(sys.argv[1] + "/run_config.json"); _, manifest = sl.load_generated_dataset(sys.argv[2]); assert config["condition"] == sys.argv[3]; assert config["seed"] == int(sys.argv[4]); assert config["dataset_sha256"] == manifest["dataset_sha256"]; assert config["rank"] == 8 and config["alpha"] == 8; assert config["learning_rate"] == 2e-4 and config["epochs"] == 3.0; assert config["effective_batch_size"] == 64; assert config["prompt_tokens_masked"] is True; assert config["target_modules"] == sl.lora_targets(sys.argv[3]); print("Validated run:", sys.argv[1], "parameters=", config["trainable_parameters"], "validation_loss=", config["validation_loss"])' "$run" "$data" "$condition" "$seed"
}

train_one() {
  local source="$1"
  local data="$2"
  local condition="$3"
  local seed="$4"
  local run="runs/${source}_canonical_${condition}_seed${seed}"

  if [[ -s "$run/adapter_config.json" ]]; then
    validate_completed_run "$run" "$data" "$condition" "$seed"
    return
  fi
  if [[ -e "$run" ]]; then
    echo "Refusing to reuse incomplete run directory: $run" >&2
    exit 1
  fi

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
    --output "$run" \
    2>&1 | tee "logs/${source}_canonical_${condition}_seed${seed}_train.log"
  validate_completed_run "$run" "$data" "$condition" "$seed"
}

evaluate_one() {
  local source="$1"
  local condition="$2"
  local seed="$3"
  local run="runs/${source}_canonical_${condition}_seed${seed}"
  local prefix="results/${source}_canonical_${condition}_seed${seed}"

  if [[ ! -s "$run/adapter_model.safetensors" ]]; then
    echo "Required adapter is missing: $run" >&2
    exit 1
  fi

  if [[ "$FORCE_EVAL" == "YES" || ! -s "${prefix}_prefixed_adapter.json" || ! -s "${prefix}_prefixed_base.json" || ! -s "${prefix}_prefixed_comparison.json" ]]; then
    python prefixed_eval.py \
      --model "$MODEL" \
      --adapter "$run" \
      --trait "$TRAIT" \
      --samples-per-prompt 100 \
      --batch-size 20 \
      --seed 0 \
      --output-prefix "${prefix}_prefixed" \
      2>&1 | tee "logs/${source}_canonical_${condition}_seed${seed}_prefixed_eval.log"
  else
    echo "Prefixed evaluation exists; skipping: ${prefix}_prefixed_adapter.json"
  fi

  if [[ "$FORCE_EVAL" == "YES" || ! -s "${prefix}_unprefixed.json" ]]; then
    python sl.py eval \
      --model "$MODEL" \
      --adapter "$run" \
      --trait "$TRAIT" \
      --seed 0 \
      --output "${prefix}_unprefixed.json" \
      2>&1 | tee "logs/${source}_canonical_${condition}_seed${seed}_unprefixed_eval.log"
  else
    echo "Unprefixed evaluation exists; skipping: ${prefix}_unprefixed.json"
  fi
}

run_preflight() {
  python -m py_compile sl.py prefixed_eval.py tests.py
  python tests.py 2>&1 | tee logs/core_cpu_tests.log
  bash -n run_core_experiment.sh run_matched_mlp.sh run_number_free_eval.sh run_reciprocal.sh
  python sl.py doctor --model "$MODEL" 2>&1 | tee logs/core_doctor.log
}

run_positive_train() {
  validate_dataset "$TRAIT_DATA" trait_teacher
  train_one "$TRAIT" "$TRAIT_DATA" full "$POSITIVE_SEED"
}

run_positive_eval() {
  local prefix="results/${TRAIT}_canonical_full_seed${POSITIVE_SEED}"
  validate_completed_run "runs/${TRAIT}_canonical_full_seed${POSITIVE_SEED}" "$TRAIT_DATA" full "$POSITIVE_SEED"
  evaluate_one "$TRAIT" full "$POSITIVE_SEED"
  if [[ "$FORCE_EVAL" == "YES" || ! -s results/core_base_unprefixed.json ]]; then
    python sl.py eval \
      --model "$MODEL" \
      --trait "$TRAIT" \
      --seed 0 \
      --output results/core_base_unprefixed.json \
      2>&1 | tee logs/core_base_unprefixed_eval.log
  fi
  cat "${prefix}_prefixed_comparison.json"
}

run_matrix() {
  require_positive_control
  validate_dataset "$TRAIT_DATA" trait_teacher
  validate_dataset "$CONTROL_DATA" neutral_teacher
  for seed in $TRAIN_SEEDS; do
    for condition in full mlp attention; do
      train_one "$TRAIT" "$TRAIT_DATA" "$condition" "$seed"
      evaluate_one "$TRAIT" "$condition" "$seed"
      train_one control "$CONTROL_DATA" "$condition" "$seed"
      evaluate_one control "$condition" "$seed"
    done
  done
  python sl.py aggregate --results-dir results --output results/summary.csv
}

run_archive() {
  python sl.py aggregate --results-dir results --output results/summary.csv
  tar \
    --exclude='*/adapter_model.safetensors' \
    --exclude='*/trainer_tmp' \
    -czf subliminal_core_results.tar.gz \
    results data runs logs \
    README.md PLAN.md requirements.txt \
    sl.py prefixed_eval.py tests.py \
    run_core_experiment.sh run_matched_mlp.sh run_number_free_eval.sh run_reciprocal.sh

  local adapter_files=()
  local run
  for run in runs/*; do
    if [[ -s "$run/adapter_model.safetensors" && -s "$run/adapter_config.json" && -s "$run/run_config.json" ]]; then
      adapter_files+=("$run/adapter_config.json" "$run/adapter_model.safetensors" "$run/run_config.json")
    fi
  done
  if [[ "${#adapter_files[@]}" -eq 0 ]]; then
    echo "No complete adapters found under runs/." >&2
    exit 1
  fi
  tar -czf subliminal_core_adapters.tar.gz "${adapter_files[@]}"

  gzip -t subliminal_core_results.tar.gz subliminal_core_adapters.tar.gz
  tar -tzf subliminal_core_results.tar.gz >/dev/null
  tar -tzf subliminal_core_adapters.tar.gz >/dev/null
  sha256sum subliminal_core_results.tar.gz subliminal_core_adapters.tar.gz \
    | tee subliminal_core_archives.sha256
  ls -lh subliminal_core_results.tar.gz subliminal_core_adapters.tar.gz subliminal_core_archives.sha256
}

case "${1:-}" in
  preflight)
    run_preflight
    ;;
  generate-trait|generate)
    generate_dataset trait "$TRAIT_DATA"
    ;;
  train-positive|train)
    run_positive_train
    ;;
  eval-positive|eval)
    run_positive_eval
    ;;
  positive-control|all)
    run_preflight
    generate_dataset trait "$TRAIT_DATA"
    run_positive_train
    run_positive_eval
    ;;
  generate-control)
    require_positive_control
    generate_dataset control "$CONTROL_DATA"
    ;;
  matrix|core)
    run_matrix
    ;;
  archive)
    run_archive
    ;;
  *)
    echo "Usage: $0 preflight|generate-trait|train-positive|eval-positive|positive-control|generate-control|matrix|archive" >&2
    exit 2
    ;;
esac
