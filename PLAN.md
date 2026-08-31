# MLP vs Attention as Write Channels for Subliminal Learning

## 1. Question

Can subliminal learning be acquired when only MLP parameters are trainable? Can it be acquired when only attention parameters are trainable?

The comparison is between a jointly trained LoRA model, an MLP-only LoRA model, and an attention-only LoRA model, all initialized from the same `Qwen/Qwen2.5-7B-Instruct` checkpoint and trained on the same apparently unrelated number data.

## 2. Motivation

Subliminal learning is safety-relevant because behavioral traits can be transmitted through synthetic data that appears unrelated to those traits. Prior work has disproportionately localized learned subliminal behavior to early layers and feed-forward/MLP components. Those results generally train a broader set of LoRA modules and localize the behavior afterward.

Post-hoc localization does not necessarily reveal the component through which a behavior can independently be acquired. This experiment tests train-time sufficiency and explicitly distinguishes:

- where the learned solution is stored after joint training; and
- which components can learn the solution from scratch when they are the only trainable components.

## 3. Hypotheses

**H1:** MLP-only LoRA will preserve most subliminal trait transfer.

**H2:** Attention-only LoRA will show substantially weaker trait transfer.

The motivation is the prior post-hoc evidence for FFN-heavy localization, which suggests that MLPs may be the dominant write channel. The important alternative is that attention-only training also transmits the trait. That result would show that post-hoc localization does not uniquely identify the parameter class capable of acquiring subliminal information.

## 4. Conditions

| Condition | Trainable LoRA targets | Trait data | Neutral data |
|---|---|---:|---:|
| Base | none | evaluation only | evaluation only |
| Full LoRA | `q/k/v/o_proj`, `gate/up/down_proj` | yes | yes |
| MLP-only LoRA | `gate/up/down_proj` | yes | yes |
| Attention-only LoRA | `q/k/v/o_proj` | yes | yes |

An optional, non-core follow-up is `down_proj`-only LoRA. It should be attempted only after the core result is secure.

The three trained conditions use the same initialization, dataset, optimizer, learning rate, rank, number of epochs or steps, and all other training settings. They differ only in LoRA target modules. Equal rank does **not** imply equal trainable parameter count; parameter count is reported rather than treated as matched.

## 5. Metrics

The primary metric is raw target-trait proportion over all sampled outputs, relative to the matched neutral control:

`trait-student rate − same-module neutral-student rate`

The key contrast is the MLP difference versus the attention difference. Secondary measurements are target rate among parsed outputs, held-out number-task loss, exact trainable parameter count, base-model trait rate, parse coverage, prompt-level rates, and raw generations.

Prompt-resampling bootstrap intervals summarize evaluation-prompt variability. One seed is not sufficient for a strong inferential claim.

## 6. Controls

- Same untouched base checkpoint for every teacher and student.
- Same generated dataset within each trait/control comparison and the same optimization hyperparameters across module conditions.
- Matched neutral teacher with no trait-inducing system prompt.
- Untouched base-model evaluation.
- Strict output filter plus case-insensitive contamination rejection and a second contamination check at training time.
- Full-LoRA positive-control replication before module comparisons are interpreted.
- Held-out overt number-task loss for every trained model.
- Exact trainable parameter count for every condition.
- All raw evaluation outputs saved for inspection.
- Seed 0 first; seeds 1 and 2 for key conditions if time permits.

## 7. Main possible interpretations

**Case A: full ≈ MLP >> attention.** MLP parameters appear sufficient, while attention updates do not appear sufficient under these optimization settings.

**Case B: full ≈ MLP ≈ attention.** Multiple parameter classes can independently acquire the hidden trait despite FFN-heavy post-hoc localization.

**Case C: full >> MLP and attention.** Interaction between parameter classes may be important for acquisition.

**Case D: attention overt-task loss is much worse.** We cannot attribute weak subliminal transmission specifically to hidden-signal specialization; parameter count or overt-task optimization capacity is a live confound.

**Case E: full positive control fails.** The experiment is invalid for answering the module question. Inspect data and evaluation and debug the replication rather than interpreting MLP-versus-attention differences.

## 8. Stop rule

Run a smoke test, then one proper seed-0 full-LoRA trait condition. Allow roughly two hours of human time for the replication gate. If the full condition does not show a clear effect over the untouched base, do not launch the matrix. Inspect accepted data, generation manifests, raw evaluation outputs, and evaluation parsing. Compare against neutral data if it can be done quickly. Abandon this project if the basic replication cannot be recovered promptly; the application benefits more from a reliable result than from an unfinished debugging effort.

## 9. Scope exclusions

This project does **not** attempt:

- layer localization or layer sweeps;
- full fine-tuning;
- model-family generalization or scale sweeps;
- activation patching or steering-vector analysis;
- sparse autoencoders;
- J-space analysis; or
- a full mechanistic decomposition.

## 10. What would count as useful evidence

The goal is not a paper-quality definitive conclusion. A useful result is a reliable, manually auditable, well-controlled piece of evidence about whether train-time module sufficiency agrees with prior post-hoc localization. That requires a working positive control, matched neutral comparisons, comparable overt-task fit, honest reporting of unequal parameter counts, and restrained conclusions across the available seeds.
