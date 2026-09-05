# Subliminal learning through MLP and attention LoRA

This repository tests whether a hidden animal preference can be acquired when only MLP LoRA parameters are trainable, only attention LoRA parameters are trainable, or both are trainable. A prompted `Qwen/Qwen2.5-7B-Instruct` teacher generates strictly filtered number sequences. Students start from the same untouched checkpoint and train only on the user number prompts and assistant number completions; they never receive the teacher's trait prompt.

The central distinction is between post-hoc localization—where a learned behavior can be found after broad training—and train-time sufficiency—which parameter class can acquire it when every other base-model parameter is frozen.

## Repository layout

- `sl.py`: data generation, LoRA training, evaluation, and aggregation.
- `prefixed_eval.py`: number-prefixed animal-preference evaluation used for the primary replication and module comparison.
- `run_core_experiment.sh`: resumable positive control and trait/neutral module matrix.
- `run_matched_mlp.sh`: parameter-matched `down_only` and random-MLP follow-up.
- `run_number_free_eval.sh`: free-running evaluation of the overt number task.
- `run_reciprocal.sh`: exploratory number-preference-through-animal-data experiment.
- `tests.py`: CPU-only tests of the load-bearing parsing, masking, target-selection, and aggregation logic.
- `PLAN.md`: research question, hypotheses, controls, and interpretation rules.

The implementation deliberately avoids a package hierarchy, configuration framework, experiment tracker, quantization, distributed training, and automatic upload.

## Conditions

| Condition | LoRA target modules |
|---|---|
| `full` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `mlp` | `gate_proj`, `up_proj`, `down_proj` |
| `attention` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `down_only` | `down_proj` |
| `mlp_random_matched` | one seeded MLP projection per transformer layer |

The core comparison uses trait-generated and neutral-generated data for `full`, `mlp`, and `attention`, plus the untouched base model. Core runs share the checkpoint, data within each source condition, optimizer, learning rate, rank, epochs, effective batch size, and all other training settings. The target modules—and therefore usually the trainable parameter counts—differ. Every run records the exact count.

At rank 8, `attention`, `down_only`, and `mlp_random_matched` have equal trainable parameter counts in Qwen2.5-7B. The matched follow-up verifies equality from saved run configurations before producing its analysis.

## Setup

Python 3.11 is recommended. Install the pinned dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python tests.py
python sl.py doctor --model Qwen/Qwen2.5-7B-Instruct
```

`doctor` loads the model configuration and tokenizer, checks that a chat template is present, and reports Python/package versions, CUDA, bf16 support, GPU memory, and free disk. It does not load the 7B model weights.

If model access in the execution environment requires Hugging Face authentication, use the standard `HF_TOKEN` environment variable or `huggingface-cli login`. No token is read into an experiment output, stored by this repository, or printed by the code.

## Data and training format

Each JSONL row contains `id`, `split`, `prompt`, `completion`, `source`, and `template_id`. Only `prompt` and `completion` enter student tokenization. Teacher details and generation counts are stored in a companion manifest.

Number data are generated at temperature 1.0, filtered to numeric-only continuations, checked for case-insensitive trait contamination, shuffled deterministically, and split into 10,000 training and 1,000 validation examples. Training uses ordinary causal-language-model SFT with all user-prompt tokens masked from the loss. `run_config.json` records the dataset hash, target modules, masking flag, hyperparameters, trainable parameter count, and final validation loss.

Outputs are written under:

```text
data/       generated JSONL and manifests
runs/       PEFT adapters, run configs, metrics, and training histories
results/    aggregate metrics, prompt-level metrics, and raw generations
logs/       shell-workflow logs
```

These directories and generated archives are ignored by Git.

## Smoke test

The following small run checks the pipeline; it is not expected to demonstrate subliminal transfer:

```bash
mkdir -p data runs results

python sl.py generate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --trait cat \
  --n-train 100 \
  --n-val 20 \
  --batch-size 8 \
  --out data/cat_smoke.jsonl

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

Inspect `data/cat_smoke.jsonl` for contamination and open the result JSON to confirm that raw completions were persisted.

## Core experiment

The core protocol uses number-continuation prompts with random three-to-nine-number prefixes. It generates 30,000 candidates and deterministically retains 10,000 training and 1,000 validation rows. Students train for three epochs with rank and alpha 8, learning rate `2e-4`, and effective batch size 64.

Run the full-LoRA trait positive control first:

```bash
./run_core_experiment.sh preflight
./run_core_experiment.sh generate-trait
./run_core_experiment.sh train-positive
./run_core_experiment.sh eval-positive
```

`eval-positive` runs both the ordinary animal prompts and ten fixed number-prefixed animal prompts with 100 samples per prompt. For the prefixed diagnostic, use `summary_excluding_known_artifact`: prompt ID 3 (zero-based) is excluded because the untouched base model has a reproducible `cat` artifact on that prompt. The exclusion is fixed in code rather than selected from the trained result.

### Positive-control rule

Do not interpret or launch the module comparison unless full LoRA produces a clear increase over the untouched base across multiple non-artifact prompts. A low number-task validation loss alone is not evidence of subliminal transfer. If the positive control fails, inspect the dataset, manifest, parser, and raw generations before changing the experiment.

After accepting the positive control from the saved raw results, generate the neutral data and run the matrix:

```bash
POSITIVE_CONTROL_ACCEPTED=YES ./run_core_experiment.sh generate-control
POSITIVE_CONTROL_ACCEPTED=YES TRAIN_SEEDS="1" ./run_core_experiment.sh matrix
./run_core_experiment.sh archive
```

The matrix trains `full`, `mlp`, and `attention` students on trait and neutral data. It validates and reuses compatible completed adapters and refuses to continue through an incomplete or mismatched run directory. Both prefixed and unprefixed evaluations are saved for every adapter. Additional seeds are explicit:

```bash
POSITIVE_CONTROL_ACCEPTED=YES TRAIN_SEEDS="2 3" ./run_core_experiment.sh matrix
```

If memory is limited, reduce the per-device batch while preserving effective batch size 64:

```bash
TRAIN_BATCH=2 EVAL_BATCH=2 GRAD_ACCUM=32 \
  ./run_core_experiment.sh train-positive
```

The archive command creates and verifies:

- `subliminal_core_results.tar.gz`
- `subliminal_core_adapters.tar.gz`
- `subliminal_core_archives.sha256`

## Parameter-matched MLP follow-up

With the core trait/control attention adapters and canonical datasets present, run:

```bash
./run_matched_mlp.sh
```

This trains trait and neutral pairs for `down_only` and `mlp_random_matched`, evaluates them with both animal-prompt protocols, and compares them with the existing attention pair. The random condition uses module-selection seed 0 to choose exactly one of `gate_proj`, `up_proj`, or `down_proj` per layer. The selected module names are stored in `run_config.json` and the trait and neutral masks must match.

The analysis reports trait-minus-neutral effects and prompt-bootstrap intervals for `down_only − attention`, `random matched MLP − attention`, and `down_only − random matched MLP`. One mask and one training seed are exploratory; a null does not establish that every matched random MLP mask would fail.

The script creates:

- `subliminal_matched_mlp_results.tar.gz`
- `subliminal_matched_mlp_adapters.tar.gz`
- `subliminal_matched_mlp_archives.sha256`

## Free-running overt-task evaluation

Validation loss is teacher-forced: each saved teacher token is scored after supplying the correct preceding tokens. To check autonomous number generation for the base model and all available core/follow-up adapters, run:

```bash
./run_number_free_eval.sh
```

The script samples 200 held-out prompts per dataset by default and reports dataset acceptance, number-count compliance, range compliance, absence of letters, output length, and sequence uniqueness. Every generated completion is saved. Exact agreement with the saved teacher completion is not a success metric because the requested continuation is random.

Outputs are archived as:

- `subliminal_number_free_eval.tar.gz`
- `subliminal_number_free_eval.sha256`

## Exploratory reciprocal experiment

`run_reciprocal.sh` tests the reverse direction: a teacher prompted to prefer the single-token number `2` or `6` generates one-animal answers, and a full-LoRA student trained on those answers is evaluated for its number preference. A neutral-data student and untouched base are controls; `5` is a distractor.

The script verifies tokenization, rejects student-visible target digits and number words, counterbalances animal lists, and evaluates every ordering of the three candidate numbers across six prompt templates. Its primary endpoint is the average own-target rate increase over the neutral student. The crossed-specificity endpoint compares each target student with the other target student.

Run stages separately or end to end:

```bash
./run_reciprocal.sh generate
./run_reciprocal.sh train
./run_reciprocal.sh eval
./run_reciprocal.sh analyze
./run_reciprocal.sh archive

# Equivalent resumable invocation
./run_reciprocal.sh all
```

The baseline precheck requires at least 70% parse coverage and rejects a target with at least 80% base-model preference. Prompt-bootstrap intervals measure prompt sensitivity, not training-seed or target-number variability. This is an existence-and-specificity test, not evidence of general symmetry.

Outputs are archived as:

- `subliminal_reciprocal_results.tar.gz`
- `subliminal_reciprocal_adapters.tar.gz`
- `subliminal_reciprocal_archives.sha256`

## Evaluation and interpretation

Animal evaluation saves two transparent rates:

- `target_trait_rate`: strict parsed target rate over all generated samples.
- `target_mentions_anywhere_rate`: target-word mentions anywhere in the raw completions.

The primary module effect is the strict trait-minus-neutral difference within each module condition. Prompt-level bootstrap intervals resample the fixed evaluation prompts. They do not estimate training-seed uncertainty, so one-seed findings should remain descriptive.

Held-out validation loss and free generation are overt-task controls. If attention fails the number task, weak subliminal transfer is ambiguous. If attention fits and freely performs the number task but still lacks trait transfer, inability to learn the overt task is a less plausible explanation; this does not prove identical representations or optimization capacity.

## Reproducibility checks

Run the CPU checks and shell syntax checks with:

```bash
python -m py_compile sl.py prefixed_eval.py tests.py
python tests.py
bash -n run_core_experiment.sh run_matched_mlp.sh \
  run_number_free_eval.sh run_reciprocal.sh
```

The main artifacts to inspect are:

- `data/*.jsonl` and `data/*.manifest.json` for student-visible data, filtering, source, and dataset hashes.
- `runs/*/run_config.json` for exact targets, hyperparameters, prompt masking, parameter counts, and validation loss.
- `runs/*/metrics.json` and `runs/*/training_log.json` for optimization records.
- `results/*.json` for prompt-level statistics and every raw generation.
- `results/summary.csv` and focused analysis JSON files for compact comparisons.

No command uploads artifacts. Base-model weights are never copied into `runs/`; only PEFT adapter weights are saved.

## Hardware and troubleshooting

Training loads Qwen2.5-7B in bf16 where supported, uses gradient checkpointing, and does not quantize. An A100 80 GB is comfortable; an A6000 48 GB is practical with a smaller per-device batch if needed. A 24 GB GPU may be tight. Disk use is dominated by the downloaded base checkpoint and generated adapters.

If generation accepts too few examples, inspect rejection counts before changing the protocol. Continuation data accept one to ten integers from 0 to 999 in the requested numeric-only format; letters, explanations, malformed delimiters, out-of-range values, and trait contamination are rejected.

Do not bypass dataset-hash or contamination failures. Very low animal-evaluation parse coverage should be investigated in the saved raw generations rather than fixed by changing the parser after seeing condition-specific results.
