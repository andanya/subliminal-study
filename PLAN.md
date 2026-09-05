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
| Down-only follow-up | `down_proj` | yes | yes |
| Random matched-MLP follow-up | one random MLP projection per layer | yes | yes |

The `down_proj`-only condition is a post-core follow-up. At rank 8 in Qwen2.5-7B it has the same number of trainable LoRA parameters as the attention-only condition; the code verifies this rather than assuming it. This comparison asks whether component identity predicts hidden-trait acquisition after removing the simplest trainable-capacity explanation.

The random matched-MLP condition uses a fixed, recorded mask to select one of `gate_proj`, `up_proj`, or `down_proj` per transformer layer. It exactly matches the rank-8 attention and down-only parameter budgets while preserving coverage of every layer. A positive result would show that matched MLP sufficiency is not unique to `down_proj`; a null result is less decisive because it could reflect an incoherent random mask.

The three trained conditions use the same initialization, dataset, optimizer, learning rate, rank, number of epochs or steps, and all other training settings. They differ only in LoRA target modules. Equal rank does **not** imply equal trainable parameter count; parameter count is reported rather than treated as matched.

## 5. Metrics

The primary metric is raw target-trait proportion over all sampled outputs, relative to the matched neutral control:

`trait-student rate − same-module neutral-student rate`

The key contrast is the MLP difference versus the attention difference. Secondary measurements are target rate among parsed outputs, held-out number-task loss, free-running number-format compliance, exact trainable parameter count, base-model trait rate, parse coverage, prompt-level rates, and raw generations. Free-running overt-task evaluation is a sanity check: it measures whether each adapter can autonomously produce valid, in-range numeric continuations rather than only assigning high probability to teacher-forced validation tokens.

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
- A declared first training seed, followed by additional seeds for key conditions.

## 7. Main possible interpretations

**Case A: full ≈ MLP >> attention.** MLP parameters appear sufficient, while attention updates do not appear sufficient under these optimization settings.

**Case B: full ≈ MLP ≈ attention.** Multiple parameter classes can independently acquire the hidden trait despite FFN-heavy post-hoc localization.

**Case C: full >> MLP and attention.** Interaction between parameter classes may be important for acquisition.

**Case D: attention overt-task loss is much worse.** We cannot attribute weak subliminal transmission specifically to hidden-signal specialization; parameter count or overt-task optimization capacity is a live confound.

**Case E: full positive control fails.** The experiment is invalid for answering the module question. Inspect data and evaluation and debug the replication rather than interpreting MLP-versus-attention differences.

**Down-only follow-up:** If parameter-matched `down_only` transmits the trait while attention does not, the result supports a specific MLP residual-write channel rather than a generic adapter-capacity explanation. If `down_only` also fails, trait acquisition may require coordination among multiple MLP projections.

**Random matched-MLP follow-up:** If both matched MLP conditions exceed attention, the evidence favors parameter class over adapter capacity or one privileged projection. If only `down_only` succeeds, the result favors the specific residual-write projection. A single random mask cannot support a strong conclusion from its own null result.

## 8. Stop rule

Run a smoke test, then one proper full-LoRA trait condition using the continuation-data and number-prefixed evaluation protocol. Generate 30,000 teacher candidates, filter and subsample 10,000 training examples, and use three training epochs. The gate comparison excludes one fixed evaluation prompt (zero-based prompt ID 3) that has a known base-model `cat` artifact; this exclusion is declared independently of the trained result.

If the full condition does not show a clear effect over the untouched base across multiple non-artifact prompts, do not launch or interpret the module matrix. Inspect accepted data, generation manifests, raw evaluation outputs, and evaluation parsing. The positive-control gate separates a failure to establish the phenomenon from evidence about MLP-versus-attention sufficiency.

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

The study is scoped to a reliable, manually auditable, well-controlled piece of evidence about whether train-time module sufficiency agrees with prior post-hoc localization, rather than a definitive general claim. That requires a working positive control, matched neutral comparisons, comparable overt-task fit, honest reporting of unequal parameter counts, and restrained conclusions across the available seeds.

## 11. Exploratory reciprocity test

A separate minimal follow-up asks whether the channel works in reverse: can a teacher's preference for an arbitrary number pass through animal-choice training data? The prespecified targets are `2` and `6`, with `5` as a distractor. Qwen tokenizes multi-digit numerals digit-by-digit, so relatively ordinary single digits are used and the culturally dominant free-response choice 7 is avoided. The run begins with a tokenizer check and aborts unless all three numeral strings are one token.

For each target and a neutral control, the teacher receives the same counterbalanced prompts asking it to choose one animal from a fixed list. Accepted completions contain exactly one listed animal and the student-visible prompt-completion pairs contain neither target digits nor their spelled-out forms. Full-LoRA students use identical optimization settings. Evaluation presents all three numbers in every order across six fixed prompt templates.

The primary endpoint is the average own-target choice-rate increase over the matched-neutral student, treating the two target numbers as fixed conditions and resampling evaluation prompts. The specificity endpoint compares each target student with the other target student on the first student's number. Both targets should move toward their own number; a common shift toward the same number is not reciprocal transmission. The untouched base measures pre-existing choice bias.

This design is an existence-and-specificity check, not a claim of general symmetry. Prompt bootstrap intervals do not measure training-seed uncertainty, and two deliberately chosen target numbers do not support population-level inference over numbers. A one-target positive is interesting but target-specific; a two-target crossed positive is presentable evidence of reciprocity in this model and training domain.
