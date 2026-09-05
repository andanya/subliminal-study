#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEED="${SEED:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
GEN_BATCH="${GEN_BATCH:-32}"
PREFERENCE_EVAL_BATCH="${PREFERENCE_EVAL_BATCH:-24}"
FORCE_EVAL="${FORCE_EVAL:-NO}"
TARGETS=(2 6)
CANDIDATES=(2 6 5)

mkdir -p data runs results logs

data_path() {
  echo "data/reciprocal_$1.jsonl"
}

run_path() {
  echo "runs/reciprocal_$1_full_seed${SEED}"
}

result_path() {
  echo "results/reciprocal_$1_full_seed${SEED}.json"
}

generate_one() {
  local target="$1"
  local data
  data="$(data_path "$target")"
  if [[ -s "$data" && -s "${data%.jsonl}.manifest.json" ]]; then
    python -c 'import sys, sl; rows, manifest = sl.load_generated_dataset(sys.argv[1]); assert manifest["training_domain"] == "animal_choices"; assert manifest.get("trait_word") == (None if sys.argv[2] == "control" else sys.argv[2]); assert manifest["student_visible_digit_count"] == 0; assert len(rows) == 11000; print("Validated:", sys.argv[1], manifest["dataset_sha256"])' "$data" "$target"
    return
  fi
  if [[ -e "$data" || -e "${data%.jsonl}.manifest.json" ]]; then
    echo "Refusing to reuse incomplete reciprocal dataset: $data" >&2
    exit 1
  fi
  python sl.py generate-reciprocal \
    --model "$MODEL" \
    --target-number "$target" \
    --candidate-numbers "${CANDIDATES[@]}" \
    --candidate-count 30000 \
    --n-train 10000 \
    --n-val 1000 \
    --temperature 1.0 \
    --seed 42 \
    --batch-size "$GEN_BATCH" \
    --out "$data" \
    2>&1 | tee "logs/reciprocal_${target}_generate.log"
}

validate_run() {
  local target="$1"
  local data run
  data="$(data_path "$target")"
  run="$(run_path "$target")"
  for required in adapter_config.json adapter_model.safetensors run_config.json metrics.json training_log.json; do
    if [[ ! -s "$run/$required" ]]; then
      echo "Incomplete reciprocal adapter: missing $run/$required" >&2
      exit 1
    fi
  done
  python -c 'import sys, sl; config = sl.read_json(sys.argv[1] + "/run_config.json"); _, manifest = sl.load_generated_dataset(sys.argv[2]); assert config["condition"] == "full"; assert config["training_domain"] == "animal_choices"; assert config["dataset_sha256"] == manifest["dataset_sha256"]; assert config["seed"] == int(sys.argv[3]); assert config["rank"] == 8 and config["alpha"] == 8; assert config["learning_rate"] == 2e-4 and config["epochs"] == 3.0; assert config["effective_batch_size"] == 64; assert config["prompt_tokens_masked"] is True; print("Validated:", sys.argv[1], "validation_loss=", config["validation_loss"])' "$run" "$data" "$SEED"
}

train_one() {
  local target="$1"
  local data run
  data="$(data_path "$target")"
  run="$(run_path "$target")"
  if [[ -s "$run/adapter_model.safetensors" ]]; then
    validate_run "$target"
    return
  fi
  if [[ -e "$run" ]]; then
    echo "Refusing to reuse incomplete reciprocal run: $run" >&2
    exit 1
  fi
  python sl.py train \
    --model "$MODEL" \
    --data "$data" \
    --condition full \
    --seed "$SEED" \
    --rank 8 \
    --alpha 8 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --batch-size "$TRAIN_BATCH" \
    --eval-batch-size "$EVAL_BATCH" \
    --gradient-accumulation "$GRAD_ACCUM" \
    --max-length 256 \
    --lr-scheduler-type linear \
    --warmup-ratio 0 \
    --warmup-steps 5 \
    --max-grad-norm 1.0 \
    --output "$run" \
    2>&1 | tee "logs/reciprocal_${target}_train.log"
  validate_run "$target"
}

evaluate_one() {
  local target="$1"
  local adapter="$2"
  local output="$3"
  if [[ "$FORCE_EVAL" != "YES" && -s "$output" ]]; then
    echo "Reciprocal evaluation exists; skipping: $output"
    return
  fi
  local adapter_args=()
  local target_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi
  if [[ "$target" != "none" ]]; then
    target_args=(--expected-target "$target")
  fi
  python sl.py eval-number-preference \
    --model "$MODEL" \
    "${adapter_args[@]}" \
    --candidate-numbers "${CANDIDATES[@]}" \
    "${target_args[@]}" \
    --samples-per-prompt 20 \
    --batch-size "$PREFERENCE_EVAL_BATCH" \
    --temperature 1.0 \
    --seed 0 \
    --output "$output" \
    2>&1 | tee "logs/$(basename "${output%.json}").log"
}

run_generate() {
  for target in control "${TARGETS[@]}"; do
    generate_one "$target"
  done
}

run_baseline_gate() {
  evaluate_one none "" results/reciprocal_base.json
  python -c 'import sys, sl; result = sl.read_json(sys.argv[1]); rates = result["summary"]["candidate_rates"]; targets = sys.argv[2:]; assert result["summary"]["parse_rate"] >= 0.7, "Base parse rate below 70%"; assert all(rates[target] < 0.8 for target in targets), f"Base target ceiling leaves too little headroom: {rates}"; print("Baseline gate passed:", rates, "parse_rate=", result["summary"]["parse_rate"])' results/reciprocal_base.json "${TARGETS[@]}"
}

run_train() {
  for target in control "${TARGETS[@]}"; do
    train_one "$target"
  done
}

run_eval() {
  evaluate_one none "" results/reciprocal_base.json
  evaluate_one none "$(run_path control)" "$(result_path control)"
  for target in "${TARGETS[@]}"; do
    evaluate_one "$target" "$(run_path "$target")" "$(result_path "$target")"
  done
}

run_analysis() {
  python sl.py analyze-reciprocal \
    --base-result results/reciprocal_base.json \
    --neutral-result "$(result_path control)" \
    --target-result "2=$(result_path 2)" \
    --target-result "6=$(result_path 6)" \
    --output results/reciprocal_analysis.json \
    2>&1 | tee logs/reciprocal_analysis.log
}

run_archive() {
  tar \
    --exclude='*/adapter_model.safetensors' \
    --exclude='*/trainer_tmp' \
    -czf subliminal_reciprocal_results.tar.gz \
    results/reciprocal_*.json \
    data/reciprocal_*.jsonl \
    data/reciprocal_*.manifest.json \
    runs/reciprocal_* \
    logs/reciprocal_*.log \
    README.md PLAN.md requirements.txt sl.py tests.py run_reciprocal.sh
  tar -czf subliminal_reciprocal_adapters.tar.gz \
    runs/reciprocal_*/adapter_config.json \
    runs/reciprocal_*/adapter_model.safetensors \
    runs/reciprocal_*/run_config.json
  gzip -t subliminal_reciprocal_results.tar.gz subliminal_reciprocal_adapters.tar.gz
  tar -tzf subliminal_reciprocal_results.tar.gz >/dev/null
  tar -tzf subliminal_reciprocal_adapters.tar.gz >/dev/null
  sha256sum \
    subliminal_reciprocal_results.tar.gz \
    subliminal_reciprocal_adapters.tar.gz \
    | tee subliminal_reciprocal_archives.sha256
  ls -lh subliminal_reciprocal_results.tar.gz subliminal_reciprocal_adapters.tar.gz subliminal_reciprocal_archives.sha256
}

echo "== Preflight =="
python -m py_compile sl.py prefixed_eval.py tests.py
python tests.py 2>&1 | tee logs/reciprocal_cpu_tests.log
bash -n run_reciprocal.sh
python sl.py doctor --model "$MODEL" 2>&1 | tee logs/reciprocal_doctor.log
python sl.py check-reciprocal-numbers --model "$MODEL" --numbers "${CANDIDATES[@]}" \
  2>&1 | tee logs/reciprocal_number_tokens.log

case "${1:-all}" in
  generate)
    run_baseline_gate
    run_generate
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  analyze)
    run_analysis
    ;;
  archive)
    run_archive
    ;;
  all)
    run_baseline_gate
    run_generate
    run_train
    run_eval
    run_analysis
    run_archive
    ;;
  *)
    echo "Usage: $0 [generate|train|eval|analyze|archive|all]" >&2
    exit 2
    ;;
esac

echo "Reciprocal subliminal-learning experiment completed."
