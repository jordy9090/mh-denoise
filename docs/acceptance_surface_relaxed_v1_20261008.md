# Acceptance policy revision — 2026-10-08

Status: implemented development policy; not yet independently validated for quality.
Base commit: `64e73bc7663bb2ae495731d6255a8f844a36512b`.

## Decision

Add opt-in `--acceptance_policy surface_relaxed_v1` to the existing selective
refinement runner. Remove only two rejection rules: candidate/SFT word-count
ratio below 0.60, and an increase in listed generic phrases. Continue logging
both as diagnostics. `word_count_ratio` is the accurate name for the existing
`specificity_ratio` field; preserve the old field for compatibility.

The legacy policy remains the default so an existing frozen command is unchanged.
Existing nonempty, 20-word, risk-delta, applicable focus-risk, bad-pattern, and
question-keyword checks remain in the revised policy. Missing/nonfinite applicable
risk values reject a candidate under the revised policy. No claim is made that
these remaining checks measure clinical safety or semantic quality completely.
The inherited 0.01 risk margin is unchanged and is not a clinically calibrated gain.

## Why this limited change

Length can penalize removal of a runaway list. One supportive phrase can appear
inside a more specific response. Neither is sufficient evidence for rejecting a
revision. Their removal is a controlled development ablation. Acceptance rate is
not a quality metric. Keep the original policy as a comparator.

No training, corruption generation, SFT/DPO/denoiser generation, risk-model
inference, gate threshold change, or four-axis conversion is needed for replay.
The call gate is unchanged; this revision cannot recover missed calls.

## Reproduce without GPU

```bash
python -m unittest discover -s tests -p 'test_acceptance_policy.py' -v
python scripts/replay_saved_acceptance.py \
  --input /ABS/PATH/proposed_mask_off_valid_outputs.jsonl \
  --manifest /ABS/PATH/proposed_mask_off_valid_outputs.manifest.json \
  --output-dir /ABS/PATH/acceptance_surface_relaxed_v1_replay
```

The replay also accepts the expert export manifest containing
`sources.proposed.settings` and its source output SHA-256. It verifies the source
hash, reproduces all original decisions/rejection reasons/final response strings,
then writes separate legacy, revised, and always-accept-called diagnostic outputs.
An existing output directory causes an error; original results are never replaced.
The always-accept-called output is a comparison condition, not a recommended policy.

For future inference, add `--acceptance_policy surface_relaxed_v1` to a copied
command and use a new versioned output path. Do not restart a generation job just
to change acceptance when its candidates and metrics already exist.

## Observed replay

On the supplied VALID43 output (SHA-256
`318eef7f613b2a41b12e68dbb0d0c33d4628ab2a7be8a4ca70cdc98de05b4ee0`):

| Policy | Calls already made | Accepted | Final equals SFT |
|---|---:|---:|---:|
| Legacy | 9 | 2 | 41 |
| surface_relaxed_v1 | 9 | 5 | 38 |
| Always accept called, diagnostic | 9 | 9 | 34 |

All 43 historical decisions and final strings reproduce. No new generation or
external judge calls were made. Thirteen CPU tests passed, including legacy parity
on 384 synthetic combinations in the development workspace.

## Candidate comparison protocol

Compare all 9 called SFT/candidate pairs, including the 7 legacy rejections.
Hide method names, risk scores, gate reasons, legacy decisions, and reference
answers. Use the question and the two exact output texts. Present A/B and B/A
for a local LLM judge; reconcile ties, uncertain judgments, and order disagreement
as unresolved. This is 18 judge requests total, not 18 new response generations.

Keep the four agreed expert dimensions: Overall Quality, Empathy, Specificity,
and Medical Advice. Use Overall Quality as the primary pair preference. Empathy
and Specificity provide explanatory comparisons. Medical Advice remains Yes / No /
Unsure with quoted evidence; these categories alone do not order clinical severity.
Record any newly introduced material problem in the overall rationale, including
unsupported personal facts, harmful guidance, loss of useful details, or editorial
meta-text. Do not treat length, a named therapy, or a keyword alone as a verdict.

Report candidate-better / SFT-better / tie / uncertain and concrete evidence.
Inspect the newly accepted three first but retain all nine in the analysis.
This small, selected VALID subset cannot establish generalization or gate recall.
Use human or a separately specified evaluation process for final TEST conclusions.
Never use TEST preferences to pick this policy or its thresholds.

Development adoption criterion: promote only after the changed cases show a
defensible net quality benefit without newly identified material harm; keep
uncertainties visible. This is a decision criterion, not a statistical guarantee.
If evidence is mixed, report this as a development ablation and retain the frozen
policy for the primary result. An external semantic selector can be a later named
method variant; it adds inference cost and must not grade its own decisions as the
sole final evaluator.

## Related work and alternatives

- ART (NAACL 2024) trains a Truster to rank the initial and refined predictions.
  Its correctness-based reasoning experiments motivate separate selection
  evaluation; they do not establish counseling safety.
  https://aclanthology.org/2024.naacl-long.327/
- Self-Refine uses LLM feedback and iterative revision without additional
  training. It is a useful alternative, with additional generation cost.
  https://arxiv.org/abs/2303.17651
- LLMRefine uses a learned fine-grained feedback model with iterative search.
  Its training and search requirements are larger than this acceptance patch.
  https://aclanthology.org/2024.findings-naacl.92/
- Pride and Prejudice (ACL 2024) documents self-evaluation bias. A semantic LLM
  selector must itself be checked; it is not automatic ground truth.
  https://aclanthology.org/2024.acl-long.826/

## Server handoff

Apply this commit in a separate worktree or branch. Do not alter running A100
corruption/QC sessions. Run the CPU replay on saved VALID outputs, preserving all
old files. Use the existing local judge runtime for the 18 blinded review requests
when available; paid API/fallback stays disabled. Do not download models or train.
Return the newly accepted cases and candidate preference evidence, plus unresolved
or order-disagreement cases. Freeze the chosen policy before inspecting TEST
comparative quality; use a separately named export for every policy.
