#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIT_DATA="${TRAIT_DATA:-data/cat_canonical.jsonl}"
CONTROL_DATA="${CONTROL_DATA:-data/control_canonical.jsonl}"
SEED="${SEED:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
FORCE_EVAL="${FORCE_EVAL:-NO}"

mkdir -p runs results logs

validate_dataset() {
  local data="$1"
  local expected_source="$2"
  python -c 'import sys, sl; rows, manifest = sl.load_generated_dataset(sys.argv[1]); assert manifest["source"] == sys.argv[2], manifest["source"]; assert sum(row["split"] == "train" for row in rows) == 10000; assert sum(row["split"] == "validation" for row in rows) == 1000; print(sys.argv[1], manifest["source"], manifest["dataset_sha256"])' "$data" "$expected_source"
}

validate_completed_run() {
  local run="$1"
  local data="$2"
  local condition="$3"
  for required in adapter_config.json adapter_model.safetensors run_config.json metrics.json training_log.json; do
    if [[ ! -s "$run/$required" ]]; then
      echo "Incomplete adapter directory: missing $run/$required" >&2
      exit 1
    fi
  done
  python -c 'import sys, sl; config = sl.read_json(sys.argv[1] + "/run_config.json"); _, manifest = sl.load_generated_dataset(sys.argv[2]); assert config["condition"] == sys.argv[3]; assert config["seed"] == int(sys.argv[4]); assert config["dataset_sha256"] == manifest["dataset_sha256"]; assert config["rank"] == 8 and config["alpha"] == 8; assert config["target_modules"] == sl.lora_targets(sys.argv[3]); print("Validated completed run:", sys.argv[1])' "$run" "$data" "$condition" "$SEED"
}

train_one() {
  local source="$1"
  local data="$2"
  local condition="$3"
  local run="runs/${source}_canonical_${condition}_seed${SEED}"

  if [[ -f "$run/adapter_config.json" ]]; then
    validate_completed_run "$run" "$data" "$condition"
    return
  fi
  if [[ -e "$run" ]]; then
    echo "Refusing to reuse incomplete run directory: $run" >&2
    echo "Rename it for inspection, then rerun this script." >&2
    exit 1
  fi

  python sl.py train \
    --model "$MODEL" \
    --data "$data" \
    --condition "$condition" \
    --seed "$SEED" \
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
    2>&1 | tee "logs/${source}_canonical_${condition}_seed${SEED}_train.log"
}

evaluate_one() {
  local source="$1"
  local condition="$2"
  local run="runs/${source}_canonical_${condition}_seed${SEED}"
  local prefix="results/${source}_canonical_${condition}_seed${SEED}"

  if [[ ! -f "$run/adapter_config.json" ]]; then
    echo "Required adapter is missing: $run" >&2
    exit 1
  fi

  if [[ "$FORCE_EVAL" == "YES" || ! -s "${prefix}_prefixed_adapter.json" || ! -s "${prefix}_prefixed_base.json" || ! -s "${prefix}_prefixed_comparison.json" ]]; then
    python prefixed_eval.py \
      --model "$MODEL" \
      --adapter "$run" \
      --trait cat \
      --samples-per-prompt 100 \
      --batch-size 20 \
      --seed 0 \
      --output-prefix "${prefix}_prefixed" \
      2>&1 | tee "logs/${source}_canonical_${condition}_seed${SEED}_prefixed_eval.log"
  else
    echo "Prefixed evaluation exists; skipping: ${prefix}_prefixed_adapter.json"
  fi

  if [[ "$FORCE_EVAL" == "YES" || ! -s "${prefix}_unprefixed.json" ]]; then
    python sl.py eval \
      --model "$MODEL" \
      --adapter "$run" \
      --trait cat \
      --seed 0 \
      --output "${prefix}_unprefixed.json" \
      2>&1 | tee "logs/${source}_canonical_${condition}_seed${SEED}_unprefixed_eval.log"
  else
    echo "Unprefixed evaluation exists; skipping: ${prefix}_unprefixed.json"
  fi
}

echo "== Preflight =="
python -m py_compile sl.py prefixed_eval.py tests.py
python tests.py 2>&1 | tee logs/down_only_cpu_tests.log
bash -n run_down_only.sh
python sl.py doctor --model "$MODEL" 2>&1 | tee logs/down_only_doctor.log

echo "== Validate canonical datasets =="
validate_dataset "$TRAIT_DATA" trait_teacher
validate_dataset "$CONTROL_DATA" neutral_teacher

echo "== Train the two new parameter-matched down_only adapters =="
train_one cat "$TRAIT_DATA" down_only
train_one control "$CONTROL_DATA" down_only

echo "== Evaluate down_only =="
evaluate_one cat down_only
evaluate_one control down_only

echo "== Ensure existing attention references are evaluated =="
evaluate_one cat attention
evaluate_one control attention

echo "== Matched analysis =="
python sl.py analyze-down-only \
  --results-dir results \
  --train-seed "$SEED" \
  --output results/down_only_analysis.json \
  2>&1 | tee logs/down_only_analysis.log
python sl.py aggregate --results-dir results --output results/summary.csv

echo "== Focused archives =="
tar \
  --exclude='*/adapter_model.safetensors' \
  --exclude='*/trainer_tmp' \
  -czf mats_sl_down_only_results.tar.gz \
  results data runs logs \
  README.md PLAN.md requirements.txt \
  sl.py prefixed_eval.py tests.py run_down_only.sh

tar -czf mats_sl_down_only_adapters.tar.gz \
  "runs/cat_canonical_down_only_seed${SEED}/adapter_config.json" \
  "runs/cat_canonical_down_only_seed${SEED}/adapter_model.safetensors" \
  "runs/cat_canonical_down_only_seed${SEED}/run_config.json" \
  "runs/control_canonical_down_only_seed${SEED}/adapter_config.json" \
  "runs/control_canonical_down_only_seed${SEED}/adapter_model.safetensors" \
  "runs/control_canonical_down_only_seed${SEED}/run_config.json"

gzip -t mats_sl_down_only_results.tar.gz mats_sl_down_only_adapters.tar.gz
tar -tzf mats_sl_down_only_results.tar.gz >/dev/null
tar -tzf mats_sl_down_only_adapters.tar.gz >/dev/null
sha256sum \
  mats_sl_down_only_results.tar.gz \
  mats_sl_down_only_adapters.tar.gz \
  | tee mats_sl_down_only_archives.sha256

ls -lh \
  mats_sl_down_only_results.tar.gz \
  mats_sl_down_only_adapters.tar.gz \
  mats_sl_down_only_archives.sha256

echo "down_only trait/control training, matched analysis, and archives completed."
