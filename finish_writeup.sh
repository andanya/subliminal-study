#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA="${DATA:-data/control_canonical.jsonl}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
SEED="${SEED:-1}"

mkdir -p data runs results logs

echo "== Preflight tests =="
python -m py_compile sl.py prefixed_eval.py tests.py
python tests.py 2>&1 | tee logs/final_cpu_tests.log
bash -n run_core.sh run_canonical_retry.sh finish_writeup.sh
python sl.py doctor --model "$MODEL" 2>&1 | tee logs/final_doctor.log

echo "== Neutral canonical dataset =="
if [[ -f "$DATA" && -f "${DATA%.jsonl}.manifest.json" ]]; then
  echo "Using existing dataset: $DATA"
else
  if [[ -e "$DATA" || -e "${DATA%.jsonl}.manifest.json" ]]; then
    echo "Incomplete dataset pair exists; preserve or remove it before rerunning." >&2
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
    --out "$DATA" \
    2>&1 | tee logs/control_canonical_generate.log
fi

python -c 'import sys, sl; rows, manifest = sl.load_generated_dataset(sys.argv[1]); assert manifest["source"] == "neutral_teacher"; assert sum(r["split"] == "train" for r in rows) == 10000; assert sum(r["split"] == "validation" for r in rows) == 1000; print("Validated", len(rows), "neutral rows; sha256=", manifest["dataset_sha256"])' "$DATA"

train_and_evaluate() {
  local condition="$1"
  local run="runs/control_canonical_${condition}_seed${SEED}"
  local prefix="results/control_canonical_${condition}_seed${SEED}"

  echo "== Neutral ${condition} =="
  if [[ -f "$run/adapter_config.json" ]]; then
    for required in adapter_model.safetensors run_config.json metrics.json training_log.json; do
      if [[ ! -s "$run/$required" ]]; then
        echo "Adapter directory is incomplete: missing $run/$required" >&2
        exit 1
      fi
    done
    echo "Completed adapter exists; skipping training: $run"
  else
    python sl.py train \
      --model "$MODEL" \
      --data "$DATA" \
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
      2>&1 | tee "logs/control_canonical_${condition}_seed${SEED}_train.log"
  fi

  python prefixed_eval.py \
    --model "$MODEL" \
    --adapter "$run" \
    --trait cat \
    --samples-per-prompt 100 \
    --batch-size 20 \
    --seed 0 \
    --output-prefix "${prefix}_prefixed" \
    2>&1 | tee "logs/control_canonical_${condition}_seed${SEED}_prefixed_eval.log"

  python sl.py eval \
    --model "$MODEL" \
    --adapter "$run" \
    --trait cat \
    --seed 0 \
    --output "${prefix}_unprefixed.json" \
    2>&1 | tee "logs/control_canonical_${condition}_seed${SEED}_unprefixed_eval.log"
}

for condition in full mlp attention; do
  train_and_evaluate "$condition"
done

echo "== Aggregate and archive =="
python sl.py aggregate --results-dir results --output results/summary.csv

tar \
  --exclude='*/adapter_model.safetensors' \
  --exclude='*/trainer_tmp' \
  -czf mats_sl_final_results.tar.gz \
  results data runs logs \
  README.md PLAN.md requirements.txt \
  sl.py prefixed_eval.py tests.py \
  run_core.sh run_canonical_retry.sh finish_writeup.sh

tar -czf mats_sl_final_adapters.tar.gz \
  runs/*/adapter_config.json \
  runs/*/adapter_model.safetensors \
  runs/*/run_config.json

gzip -t mats_sl_final_results.tar.gz mats_sl_final_adapters.tar.gz
tar -tzf mats_sl_final_results.tar.gz >/dev/null
tar -tzf mats_sl_final_adapters.tar.gz >/dev/null
sha256sum \
  mats_sl_final_results.tar.gz \
  mats_sl_final_adapters.tar.gz \
  | tee mats_sl_final_archives.sha256

ls -lh \
  mats_sl_final_results.tar.gz \
  mats_sl_final_adapters.tar.gz \
  mats_sl_final_archives.sha256

echo "All three neutral runs, evaluations, archives, and integrity checks completed."
