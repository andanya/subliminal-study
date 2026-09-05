#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEED="${SEED:-1}"
N_PROMPTS="${N_PROMPTS:-200}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FORCE_EVAL="${FORCE_EVAL:-NO}"
CONDITIONS=(full mlp attention down_only mlp_random_matched)

mkdir -p results logs

evaluate_one() {
  local label="$1"
  local data="$2"
  local adapter="$3"
  local output="results/number_free_${label}.json"

  if [[ "$FORCE_EVAL" != "YES" && -s "$output" ]]; then
    echo "Number evaluation exists; skipping: $output"
    return
  fi
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi
  python sl.py number-eval \
    --model "$MODEL" \
    --data "$data" \
    "${adapter_args[@]}" \
    --label "$label" \
    --n-prompts "$N_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --temperature 1.0 \
    --output "$output" \
    2>&1 | tee "logs/number_free_${label}.log"
}

echo "== Preflight =="
python -m py_compile sl.py tests.py
python tests.py 2>&1 | tee logs/number_free_cpu_tests.log
python sl.py doctor --model "$MODEL" 2>&1 | tee logs/number_free_doctor.log

for source in cat control; do
  data="data/${source}_canonical.jsonl"
  if [[ ! -s "$data" ]]; then
    echo "Missing canonical dataset: $data" >&2
    exit 1
  fi
  python -c 'import sys, sl; rows, manifest = sl.load_generated_dataset(sys.argv[1]); assert manifest.get("training_domain", "numbers") == "numbers"; assert sum(row["split"] == "validation" for row in rows) >= int(sys.argv[2]); print(sys.argv[1], manifest["dataset_sha256"])' "$data" "$N_PROMPTS"
  evaluate_one "${source}_base" "$data" ""
  for condition in "${CONDITIONS[@]}"; do
    adapter="runs/${source}_canonical_${condition}_seed${SEED}"
    if [[ -s "$adapter/adapter_model.safetensors" ]]; then
      evaluate_one "${source}_${condition}_seed${SEED}" "$data" "$adapter"
    else
      echo "Adapter unavailable; skipping: $adapter"
    fi
  done
done

python sl.py aggregate-number-evals \
  --results-dir results \
  --output results/number_free_generation_summary.csv

tar -czf subliminal_number_free_eval.tar.gz \
  results/number_free_*.json \
  results/number_free_generation_summary.csv \
  logs/number_free_*.log \
  sl.py tests.py run_number_free_eval.sh
gzip -t subliminal_number_free_eval.tar.gz
tar -tzf subliminal_number_free_eval.tar.gz >/dev/null
sha256sum subliminal_number_free_eval.tar.gz | tee subliminal_number_free_eval.sha256
ls -lh subliminal_number_free_eval.tar.gz subliminal_number_free_eval.sha256

echo "Free-running number-task evaluation completed."
