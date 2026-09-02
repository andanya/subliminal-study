# Subliminal learning: MLP-only vs attention-only LoRA

## 1. What this repo does

This repository tests whether a subliminal animal preference can be acquired when only MLP LoRA parameters are trainable, when only attention LoRA parameters are trainable, or when both are trainable. The teacher and every student start from `Qwen/Qwen2.5-7B-Instruct`. The teacher emits strictly filtered number sequences; the student sees only the ordinary user number prompt and assistant number completion.

The implementation is intentionally small: [sl.py](sl.py) contains generation, training, evaluation, and aggregation. [run_canonical_retry.sh](run_canonical_retry.sh) contains the recommended replication protocol and a manually gated core matrix; [prefixed_eval.py](prefixed_eval.py) contains the corresponding published-style diagnostic evaluation. There is no framework, package hierarchy, W&B, quantization, distributed training, or automatic upload.

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

The recommended gate closely follows the [successful public Qwen replication](https://github.com/iremkrc/subliminal-learning-open) of the [subliminal-learning study](https://www.nature.com/articles/s41586-026-10319-8), rather than the simpler smoke-test protocol. It uses number-continuation prompts with random 3–9-number prefixes, generates 30,000 candidates before deterministically selecting 10,000 train and 1,000 validation rows, uses the stronger published trait prompt, and trains full LoRA for three epochs with rank/alpha 8 and effective batch size 64. It also evaluates ten fixed number-prefixed animal prompts with 100 samples each.

Run each stage separately:

```bash
./run_canonical_retry.sh generate 2>&1 | tee canonical_generate.log
wc -l data/cat_canonical.jsonl
grep -in cat data/cat_canonical.jsonl || echo "No cat contamination found"
cat data/cat_canonical.manifest.json

./run_canonical_retry.sh train 2>&1 | tee canonical_train.log
./run_canonical_retry.sh eval 2>&1 | tee canonical_eval.log
```

The `all` mode runs those same three stages, but separate invocations make it easier to inspect data and GPU memory between stages. On an out-of-memory error, retry training with the same effective batch size:

```bash
TRAIN_BATCH=2 EVAL_BATCH=2 GRAD_ACCUM=32 ./run_canonical_retry.sh train \
  2>&1 | tee canonical_train.log
```

Inspect these gate artifacts:

```bash
cat results/canonical_prefixed_gate_comparison.json
python -m json.tool results/canonical_prefixed_gate_base.json | less
python -m json.tool results/canonical_prefixed_gate_adapter.json | less
cat runs/cat_canonical_full_seed1/run_config.json
cat runs/cat_canonical_full_seed1/metrics.json
```

Use `summary_excluding_known_artifact` as the primary diagnostic. Prompt ID 3 (zero-based) is explicitly excluded because the untouched Qwen model can answer `cat` on that particular fixed prompt even without training. A credible pass should be a clear positive adapter-minus-base difference distributed across multiple other prompts, with raw completions that survive manual inspection; the prompt-bootstrap interval is descriptive rather than a formal hypothesis test. The ordinary unprefixed evaluation is saved as a secondary robustness check.

**Replication gate:** if the full adapter still has zero or no clear increase outside the known artifact, stop and abandon the module-comparison experiment. Do not treat low validation loss as evidence of subliminal transfer. A positive result permits the neutral and module controls; it is not by itself proof of the research claim.

## 7. Core experiment

Only after manually accepting the gate, generate the matched neutral dataset with the same settings except for the absent trait system prompt:

```bash
GATE_PASSED=YES ./run_canonical_retry.sh generate-control
```

The core command is deliberately protected by the same explicit acknowledgement. It trains and evaluates `full`, `mlp`, and `attention` on trait and neutral data with exactly the canonical settings. It reuses the completed trait/full/seed1 adapter rather than retraining it:

```bash
GATE_PASSED=YES TRAIN_SEEDS="1" ./run_canonical_retry.sh core
```

This is expensive: after the gate it entails five new three-epoch training runs. Do not launch it merely because the script exists. The script does not decide whether the gate passed and cannot replace raw-output inspection.

If the first complete comparison is useful and time permits, additional training seeds are explicit:

```bash
GATE_PASSED=YES TRAIN_SEEDS="2 3" ./run_canonical_retry.sh core
```

The original `run_core.sh` and simple-prompt commands remain useful as readable examples, but they do not use the canonical retry settings. Do not mix their adapters or datasets into the canonical comparison.

The direct single-run commands remain available through `sl.py`; use the flags in `train_one` inside `run_canonical_retry.sh` verbatim when running only selected conditions.

After the three trait adapters pass inspection, `finish_writeup.sh` completes the seed-1 matrix by generating neutral data, training and evaluating neutral `full`, `mlp`, and `attention`, running local checks, and producing verified final archives. It is resumable and skips a neutral adapter only when all of its required saved files exist:

```bash
./finish_writeup.sh
```

### Parameter-matched `down_only` follow-up

After the seed-1 trait/control matrix exists, run the focused follow-up:

```bash
./run_down_only.sh 2>&1 | tee down_only_all.log
```

It trains exactly two new adapters on the existing canonical datasets: trait `down_only` and neutral `down_only`. It then evaluates them with both protocols, checks the existing trait/control attention adapters, and runs:

```bash
python sl.py analyze-down-only \
  --results-dir results \
  --train-seed 1 \
  --output results/down_only_analysis.json
```

The analysis reports the trait-minus-neutral effect for each condition and the `down_only`-minus-attention difference, with evaluation-prompt bootstrap intervals. It also verifies whether the two conditions have exactly equal trainable-parameter counts and records their held-out number-task losses. The primary scientific question is whether `down_proj` alone transmits the trait when an equal-sized attention adapter does not; one seed remains descriptive evidence, not a definitive inferential result.

The script is resumable: complete adapters and evaluations are validated and reused. It refuses to continue through an incomplete run directory. Set `FORCE_EVAL=YES` only when intentionally resampling all evaluations. On completion it creates and integrity-checks:

- `mats_sl_down_only_results.tar.gz`
- `mats_sl_down_only_adapters.tar.gz`
- `mats_sl_down_only_archives.sha256`

The attention adapters are required as references but are not retrained by this script.

### Down-only plus random matched-MLP follow-up

`run_matched_mlp.sh` is the recommended script when running both parameter-matched MLP controls. It performs four new trainings, not two:

| New condition | Trait data | Neutral data |
|---|---:|---:|
| `down_only` | one run | one run |
| `mlp_random_matched` | one run | one run |

The existing trait and neutral attention adapters are reused. At rank 8, all three conditions have exactly the same number of trainable parameters. `mlp_random_matched` uses fixed module-selection seed 0 to choose exactly one of `gate_proj`, `up_proj`, or `down_proj` in every transformer layer. The chosen full module names are saved in each `run_config.json`; trait and neutral runs must use the identical mask.

On a new Vast instance, first upload the previously verified final archives from the local `downloads/` directory. After cloning the latest repository, restore only experiment artifacts so that the older code stored in the results archive does not overwrite the new checkout:

```bash
tar -xzf mats_sl_final_results.tar.gz data runs results logs
tar -xzf mats_sl_final_adapters.tar.gz
```

Then run:

```bash
chmod +x run_matched_mlp.sh
./run_matched_mlp.sh 2>&1 | tee matched_mlp_all.log
```

The script validates the restored datasets, runs CPU checks and `doctor`, trains and evaluates the four new adapters, reuses or regenerates the attention evaluations, and computes strict and mention-anywhere trait-minus-neutral effects. It also computes `down_only − attention`, `random matched MLP − attention`, and `down_only − random matched MLP` prompt-bootstrap contrasts for both evaluation protocols. It refuses to archive results unless all three conditions have exactly equal trainable-parameter counts.

Download all three new outputs before destroying the instance:

- `mats_sl_matched_mlp_results.tar.gz`
- `mats_sl_matched_mlp_adapters.tar.gz`
- `mats_sl_matched_mlp_archives.sha256`

One trait/control pair per condition is enough for a minimal seed-1 follow-up aligned with the existing matrix. It does not estimate training-seed variance; additional masks or seeds are follow-up work rather than part of this first comparison.

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

**Out of memory.** The canonical script starts at per-device batch 4 and gradient accumulation 16. On 48 GB, use `TRAIN_BATCH=2 EVAL_BATCH=2 GRAD_ACCUM=32` if needed; this preserves effective batch size 64. On a 4090, batch size 1 may still be tight. This repository deliberately does not implement QLoRA; rent a larger GPU rather than changing the method mid-project.

**Generation accepts too few examples.** Inspect rejection counts before changing the protocol. Simple prompts require exactly ten comma-separated integers. Canonical continuation prompts accept 1–10 integers from 0 to 999 in the requested numeric-only formats. Letters, explanations, malformed delimiters, out-of-range values, and trait contamination are rejected.

**Dataset hash or contamination error.** Do not bypass it. Confirm the JSONL and companion manifest came from the same generation run. Regenerate into a new path if either was edited.

**Very low parse coverage.** Read `raw_generations` in the result JSON. The metric parses only a single animal token with optional terminal punctuation. This keeps the endpoint transparent; do not silently loosen it after seeing condition-specific outputs.

**Attention has weak trait transfer.** Check its held-out validation loss and trainable parameter count before making a specialization claim. Equal LoRA rank does not match parameter count or optimization capacity.

**Full positive control fails.** Stop. Compare against base, inspect the data and raw outputs, verify that the number task trained, and check model/dependency versions. Do not interpret MLP-versus-attention results from a failed replication.

## 11. Expected GPU memory / hardware

The code loads the 7B model in bf16 where supported, uses ordinary PEFT LoRA, enables gradient checkpointing, and does not quantize. An A100 80 GB should be comfortable. An A6000 48 GB should be practical, with the documented batch-2 fallback if batch 4 is too large. A 4090 24 GB may be tight and is not the recommended retry hardware. Actual peak memory depends on CUDA/PyTorch kernels and sequence length; watch `nvidia-smi` during the first training steps.

No command in this repository uploads artifacts. Hugging Face model loading performs the expected model download; all experiment outputs remain in `data/`, `runs/`, and `results/` on the instance.
