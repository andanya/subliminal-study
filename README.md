# Subliminal learning: MLP-only vs attention-only LoRA

## 1. What this repo does

This repository tests whether a subliminal animal preference can be acquired when only MLP LoRA parameters are trainable, when only attention LoRA parameters are trainable, or when both are trainable. The teacher and every student start from `Qwen/Qwen2.5-7B-Instruct`. The teacher emits strictly filtered number sequences; the student sees only the ordinary user number prompt and assistant number completion.

The implementation is intentionally small: [sl.py](sl.py) contains generation, training, evaluation, and aggregation; [run_core.sh](run_core.sh) calls those commands visibly. There is no framework, package hierarchy, W&B, quantization, distributed training, or automatic upload.

Generated JSONL rows have six fields: `id`, `split`, `prompt`, `completion`, `source`, and `template_id`. Only `prompt` and `completion` enter tokenization. The teacher system prompt and contamination word are saved in a companion `*.manifest.json`, which is never student training text.

## 2. Experiment conditions

| Name | LoRA target modules |
|---|---|
| `full` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `mlp` | `gate_proj`, `up_proj`, `down_proj` |
| `attention` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `down_only` | `down_proj` (optional, not in the core script) |

All core conditions use the same rank and optimization settings. Their trainable parameter counts are not equal. Every run prints and records the exact count; interpret module differences together with held-out number loss and parameter-count/optimization-capacity as a possible confound.

The core matrix is trait-generated and neutral-generated data × `full`, `mlp`, and `attention`, plus an untouched base evaluation. `target_trait_rate` uses all sampled outputs as its denominator; `target_trait_rate_among_parsed` and `parse_rate` are saved separately. The primary comparison is:

```text
(MLP trait student − MLP neutral student)
versus
(attention trait student − attention neutral student)
```

## 3. Local setup

Python 3.11 is the conservative choice for the pinned dependencies. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python tests.py
```

The tests exercise only strict filtering, contamination, output parsing, target mappings, label masking, and result aggregation. They do not download Qwen. The program reads `HF_TOKEN` through Hugging Face's standard tooling if set; it never writes or prints the token.

## 4. Running on Vast.ai

Rent one GPU and one instance. Prefer an A100 80 GB or A6000 48 GB. A 4090 24 GB is only a fallback. Choose a recent PyTorch/CUDA image, enable SSH, and allocate 60–100 GB of disk. Connect with the SSH command Vast provides, then run these commands (replace the repository URL):

```bash
python -m pip install --upgrade uv
git clone <REPO_URL> subliminal-study
cd subliminal-study
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
nvidia-smi
python sl.py doctor --model Qwen/Qwen2.5-7B-Instruct
```

If model authentication is needed, either export a token for the current shell or use the Hugging Face CLI supplied by `huggingface_hub`:

```bash
export HF_TOKEN=<YOUR_HUGGING_FACE_TOKEN>
# or: huggingface-cli login
```

Do not put a token in this repository, a shell script, a JSON config, or an archive. `doctor` downloads only the model config and tokenizer, confirms the chat template, and reports CUDA, bf16 support, package versions, GPU memory, and free disk. It does not load the 7B weights.

## 5. Smoke test

First generate 100 training and 20 validation examples:

```bash
mkdir -p data runs results
python sl.py generate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --trait cat \
  --n-train 100 \
  --n-val 20 \
  --batch-size 8 \
  --out data/cat_smoke.jsonl
```

Inspect the actual rows and verify the animal word is absent:

```bash
head -n 3 data/cat_smoke.jsonl
grep -in cat data/cat_smoke.jsonl || echo "No cat contamination found"
cat data/cat_smoke.manifest.json
```

Then perform five optimizer steps and a tiny evaluation:

```bash
python sl.py train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/cat_smoke.jsonl \
  --condition full \
  --seed 0 \
  --rank 8 \
  --max-steps 5 \
  --output runs/cat_smoke_full_seed0

python sl.py eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter runs/cat_smoke_full_seed0 \
  --trait cat \
  --n-prompts 3 \
  --samples-per-prompt 3 \
  --output results/cat_smoke_full_seed0.json
```

This checks the pipeline, not the research effect. Open the JSON result and confirm that every raw completion is present.

## 6. Positive-control replication

Generate the proper trait dataset. Generation is deterministic for a fixed software/hardware setup and seed, but GPU sampling can still vary across hardware:

```bash
python sl.py generate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --trait cat \
  --n-train 10000 \
  --n-val 1000 \
  --temperature 1.0 \
  --seed 0 \
  --out data/cat.jsonl
```

The positive-control gate is:

```bash
./run_core.sh positive
```

It evaluates the untouched base, trains exactly one proper `cat/full/seed0` adapter, evaluates it, and writes `results/summary.csv`. It then prints a stop message. Inspect:

```bash
cat results/summary.csv
python -m json.tool results/base_seed0.json | less
python -m json.tool results/cat_full_seed0.json | less
cat runs/cat_full_seed0/run_config.json
cat runs/cat_full_seed0/metrics.json
```

**Replication gate:** if full LoRA does not show a clear target-trait increase over the untouched base, do not run the matrix. Spend at most about two hours of human time checking the data, manifest, parser coverage, raw outputs, and training metrics. If the replication cannot be recovered quickly, abandon this experiment rather than using the remaining application time on an uninterpretable MLP-versus-attention comparison. A high rate in one run is a reason to continue, not proof of replication; the neutral full condition remains necessary for the core interpretation.

## 7. Core experiment

Only after the gate looks healthy, generate the matched neutral dataset with the same settings except for the absent trait system prompt:

```bash
python sl.py generate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --trait control \
  --n-train 10000 \
  --n-val 1000 \
  --temperature 1.0 \
  --seed 0 \
  --out data/control.jsonl
```

Run the six seed-0 conditions:

```bash
SEEDS="0" ./run_core.sh core
```

The script sees the completed positive-control adapter and skips retraining it. It evaluates every adapter against the same target animal and regenerates `results/summary.csv`. It does not decide whether an effect is “positive” and does not launch additional seeds.

If the positive control and seed-0 comparisons are useful, repeat all conditions or edit the loop to select only key conditions:

```bash
SEEDS="1 2" ./run_core.sh core
```

The direct single-run commands are also available and correspond exactly to the script:

```bash
python sl.py train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/cat.jsonl \
  --condition mlp \
  --seed 0 \
  --rank 8 \
  --output runs/cat_mlp_seed0

python sl.py eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter runs/cat_mlp_seed0 \
  --trait cat \
  --output results/cat_mlp_seed0.json

python sl.py aggregate \
  --results-dir results \
  --output results/summary.csv
```

Base evaluation works by omitting `--adapter`:

```bash
python sl.py eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --trait cat \
  --output results/base_seed0.json
```

## 8. Downloading results before terminating the instance

Create one archive with results, generated data and small run records, and a separate archive with LoRA weights:

```bash
tar -czf mats_sl_results.tar.gz \
  results data \
  runs/*/run_config.json \
  runs/*/metrics.json \
  runs/*/training_log.json

tar -czf mats_sl_adapters.tar.gz \
  runs/*/adapter_config.json \
  runs/*/adapter_model.safetensors

tar -tzf mats_sl_results.tar.gz | head
tar -tzf mats_sl_adapters.tar.gz | head
```

From your local machine, using the host and SSH port Vast shows:

```bash
scp -P <PORT> root@<HOST>:/workspace/subliminal-study/mats_sl_results.tar.gz .
scp -P <PORT> root@<HOST>:/workspace/subliminal-study/mats_sl_adapters.tar.gz .
```

Adjust `/workspace/subliminal-study` to the actual clone path. VS Code Remote SSH and Vast's file UI are also fine. Download at minimum `results/`, `data/`, every `run_config.json`, `metrics.json`, and `training_log.json`. Prefer downloading both adapter files per run as well; they are much smaller than the base model and allow re-evaluation. The Qwen base weights do not need to be copied.

**DO NOT DESTROY THE VAST INSTANCE UNTIL BOTH ARCHIVES HAVE BEEN DOWNLOADED, LISTED, AND OPENED LOCALLY.**

## 9. Files to inspect manually

- `data/*.jsonl`: accepted student-visible prompts and number completions, with deterministic train/validation splits.
- `data/*.manifest.json`: teacher condition, system prompt, generation settings, rejection counts, and dataset hash.
- `runs/*/run_config.json`: exact LoRA targets, hyperparameters, data hash, masking flag, parameter counts, and final validation loss.
- `runs/*/metrics.json` and `training_log.json`: final metrics and Trainer log history.
- `results/*.json`: overall rate, prompt-bootstrap interval, prompt-level rates, parse coverage, and every raw completion.
- `results/summary.csv`: flat comparison table. Do not treat it as a substitute for raw-output inspection.

## 10. Troubleshooting

**Out of memory.** On 48–80 GB, keep batch size 1 and gradient checkpointing on. Increase `--gradient-accumulation` if changing effective batch size. On a 4090, try `--max-length 128` and batch size 1. This repository deliberately does not implement QLoRA; if 24 GB remains unreliable, rent a larger GPU rather than expanding the method mid-project.

**Generation accepts too few examples.** Inspect teacher outputs before increasing `--max-attempts-multiplier`. The accepted format is exactly ten integers from 0 to 999, separated by commas, with whitespace or a final newline allowed. Letters, explanations, wrong counts, and trait contamination are rejected.

**Dataset hash or contamination error.** Do not bypass it. Confirm the JSONL and companion manifest came from the same generation run. Regenerate into a new path if either was edited.

**Very low parse coverage.** Read `raw_generations` in the result JSON. The metric parses only a single animal token with optional terminal punctuation. This keeps the endpoint transparent; do not silently loosen it after seeing condition-specific outputs.

**Attention has weak trait transfer.** Check its held-out validation loss and trainable parameter count before making a specialization claim. Equal LoRA rank does not match parameter count or optimization capacity.

**Full positive control fails.** Stop. Compare against base, inspect the data and raw outputs, verify that the number task trained, and check model/dependency versions. Do not interpret MLP-versus-attention results from a failed replication.

## 11. Expected GPU memory / hardware

The code loads the 7B model in bf16 where supported, uses ordinary PEFT LoRA, enables gradient checkpointing, and does not quantize. An A100 80 GB should be comfortable. An A6000 48 GB should be practical at per-device batch size 1. A 4090 24 GB may be tight and is not the recommended first attempt. Actual peak memory depends on CUDA/PyTorch kernels and sequence length; run the smoke test and watch `nvidia-smi` before starting the proper run.

No command in this repository uploads artifacts. Hugging Face model loading performs the expected model download; all experiment outputs remain in `data/`, `runs/`, and `results/` on the instance.
