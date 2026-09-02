#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIT_DATA="${TRAIT_DATA:-data/cat_canonical.jsonl}"
CONTROL_DATA="${CONTROL_DATA:-data/control_canonical.jsonl}"
SEED="${SEED:-1}"
MODULE_SELECTION_SEED="${MODULE_SELECTION_SEED:-0}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
FORCE_EVAL="${FORCE_EVAL:-NO}"

NEW_CONDITIONS=(down_only mlp_random_matched)
REFERENCE_CONDITION=attention
REFERENCE_RUN="runs/cat_canonical_attention_seed${SEED}"

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
  python -c 'import re, sys, sl; config = sl.read_json(sys.argv[1] + "/run_config.json"); _, manifest = sl.load_generated_dataset(sys.argv[2]); condition = sys.argv[3]; reference = sl.read_json(sys.argv[6] + "/run_config.json"); assert config["condition"] == condition; assert config["seed"] == int(sys.argv[4]); assert config["dataset_sha256"] == manifest["dataset_sha256"]; assert config["rank"] == 8 and config["alpha"] == 8; assert config["learning_rate"] == 2e-4 and config["epochs"] == 3.0; assert config["effective_batch_size"] == 64; assert config["prompt_tokens_masked"] is True; assert config["trainable_parameters"] == reference["trainable_parameters"]; assert config.get("module_selection_seed") == (int(sys.argv[5]) if condition == sl.RANDOM_MATCHED_CONDITION else None); assert condition == sl.RANDOM_MATCHED_CONDITION or config["target_modules"] == sl.lora_targets(condition); assert condition != sl.RANDOM_MATCHED_CONDITION or len(config["target_modules"]) == 28; assert condition != sl.RANDOM_MATCHED_CONDITION or len({int(re.search(r"layers\.(\d+)\.mlp", name).group(1)) for name in config["target_modules"]}) == 28; print("Validated completed run:", sys.argv[1], "parameters=", config["trainable_parameters"])' "$run" "$data" "$condition" "$SEED" "$MODULE_SELECTION_SEED" "$REFERENCE_RUN"
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
    --module-selection-seed "$MODULE_SELECTION_SEED" \
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
  validate_completed_run "$run" "$data" "$condition"
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
python tests.py 2>&1 | tee logs/matched_mlp_cpu_tests.log
bash -n run_matched_mlp.sh
python sl.py doctor --model "$MODEL" 2>&1 | tee logs/matched_mlp_doctor.log

echo "== Validate restored canonical datasets =="
validate_dataset "$TRAIT_DATA" trait_teacher
validate_dataset "$CONTROL_DATA" neutral_teacher
validate_completed_run "$REFERENCE_RUN" "$TRAIT_DATA" "$REFERENCE_CONDITION"
validate_completed_run "runs/control_canonical_attention_seed${SEED}" "$CONTROL_DATA" "$REFERENCE_CONDITION"

echo "== Train four new adapters: two conditions x trait/control =="
for condition in "${NEW_CONDITIONS[@]}"; do
  train_one cat "$TRAIT_DATA" "$condition"
  train_one control "$CONTROL_DATA" "$condition"
done

echo "== Evaluate new adapters =="
for condition in "${NEW_CONDITIONS[@]}"; do
  evaluate_one cat "$condition"
  evaluate_one control "$condition"
done

echo "== Ensure the restored attention references are available =="
evaluate_one cat "$REFERENCE_CONDITION"
evaluate_one control "$REFERENCE_CONDITION"

echo "== Parameter-matched analysis =="
python sl.py analyze-matched-mlp \
  --results-dir results \
  --train-seed "$SEED" \
  --output results/matched_mlp_analysis.json \
  2>&1 | tee logs/matched_mlp_analysis.log
python sl.py aggregate --results-dir results --output results/summary.csv

echo "== Focused archives =="
tar \
  --exclude='*/adapter_model.safetensors' \
  --exclude='*/trainer_tmp' \
  -czf mats_sl_matched_mlp_results.tar.gz \
  results data runs logs \
  README.md PLAN.md requirements.txt \
  sl.py prefixed_eval.py tests.py run_matched_mlp.sh

tar -czf mats_sl_matched_mlp_adapters.tar.gz \
  "runs/cat_canonical_down_only_seed${SEED}/adapter_config.json" \
  "runs/cat_canonical_down_only_seed${SEED}/adapter_model.safetensors" \
  "runs/cat_canonical_down_only_seed${SEED}/run_config.json" \
  "runs/control_canonical_down_only_seed${SEED}/adapter_config.json" \
  "runs/control_canonical_down_only_seed${SEED}/adapter_model.safetensors" \
  "runs/control_canonical_down_only_seed${SEED}/run_config.json" \
  "runs/cat_canonical_mlp_random_matched_seed${SEED}/adapter_config.json" \
  "runs/cat_canonical_mlp_random_matched_seed${SEED}/adapter_model.safetensors" \
  "runs/cat_canonical_mlp_random_matched_seed${SEED}/run_config.json" \
  "runs/control_canonical_mlp_random_matched_seed${SEED}/adapter_config.json" \
  "runs/control_canonical_mlp_random_matched_seed${SEED}/adapter_model.safetensors" \
  "runs/control_canonical_mlp_random_matched_seed${SEED}/run_config.json"

gzip -t mats_sl_matched_mlp_results.tar.gz mats_sl_matched_mlp_adapters.tar.gz
tar -tzf mats_sl_matched_mlp_results.tar.gz >/dev/null
tar -tzf mats_sl_matched_mlp_adapters.tar.gz >/dev/null
sha256sum \
  mats_sl_matched_mlp_results.tar.gz \
  mats_sl_matched_mlp_adapters.tar.gz \
  | tee mats_sl_matched_mlp_archives.sha256

ls -lh \
  mats_sl_matched_mlp_results.tar.gz \
  mats_sl_matched_mlp_adapters.tar.gz \
  mats_sl_matched_mlp_archives.sha256

echo "Matched down_only/random-MLP follow-up completed."
