# ACL ARR 2026 Reproducibility Record

This record describes the manuscript **SFT-First Selective Refinement for
Mental-Health Question Answering**, using the production artifacts inspected on
2026-10-12. It records evidence and remaining dependencies; it is not a claim
that a fresh public clone can reproduce every manuscript result.

The manuscript source inspected was `acl_latex.tex`, SHA256
`94334304dcc95f655ff46a6df7c2d8555a7cd39f27dec59a96319cc47b7b5db7`.
The source-code snapshot inspected was commit
`ec69e056873f72f8277648d2ec5318871051120b`. The documentation branch was created
from `origin/main` at `756f42d`; most ACL production scripts listed below are
absent at that base. They are **integration dependencies**, not working links or
an already runnable public release. The final comparison against A100's
integrated `main` is recorded separately in [ACL_VERIFICATION.md](ACL_VERIFICATION.md).

No training, generation, model loading, paid API calls, or TEST changes were
performed for this documentation review. Local artifacts were read for schemas,
hashes, counts, and configuration. Counseling text is not reproduced here.

## Scope And Evidence

| Stage | Evidence inspected | Reproduction status |
| --- | --- | --- |
| Frozen TRAIN/VALID export | Eight JSONL files match `frozen_manifest.json` by SHA256 and row count | Verified local dependency; not distributed |
| QC1/QC2 and final pair/span export | Production contracts, typed QC implementation, review/export manifest | Construction lineage verified; a single rerun cannot replace the recorded reviews |
| Router, Scorer, SFT, complete-response refinement | Completed training manifests, saved configs, exact code paths | Existing run documented; fresh training not rerun |
| Corruption TEST60 generation and GPT selection | Corrected frozen final file, contracts and completion reports | 60 questions, 18 invocations, six GPT adoptions verified |
| GPT pointwise evaluation | Corrected TEST60 publication contract/report | Completed artifact documented; API calls not repeated |
| Independent Llama matrix | Completed component-matrix evaluation and publication tables | Main TEST60 table source identified; separate Appendix count mismatch remains |
| Ordinary drafts | Private generation/evaluation scripts and completed outputs | Generation and Llama assessment identified; paper's GPT final selection not verified |
| Expert evaluation | Blinded packet-building code and unfilled packet exports | Completed six-question ratings and agreement analysis not located |

The later TEST100 extension is a separate experiment. Its 100 questions,
28 invocations, and nine adoptions must not replace the manuscript's TEST60
denominators or results. Likewise, partial TEST15/24 exports and the
uncorrected TEST60 v1 outputs are intermediate artifacts, not the final paper
evaluation set.

## Data Dependencies And Labels

The immediate sources are MentalChat16K and CounselChat. The pinned source
identifiers in `scripts/fullpaper_acl_pipeline.py` are:

| Source | Repository | Recorded revision |
| --- | --- | --- |
| MentalChat16K | `ShenLab/MentalChat16K` | `5f60cd380cfc58f0f12d44892bed41ee3670a70a` |
| CounselChat | `nbertagnolli/counsel-chat` | `17501f72697cf8018aaf496162e9dc1408a64e67` |
| CounselBench exclusion reference | Pinned benchmark snapshot | `8d56a96ea1de3f3f190f77f4ca9bc3503d731af7` |

The provenance/intake implementation is `scripts/prepare_fullpaper_phase1.py`.
The canonical builder is `scripts/fullpaper_acl_pipeline.py` (`build-clean`,
then `assign-corruptions`). It groups source identifiers and exact/normalized
duplicate components before splitting. MentalChat16K lacks released session
identifiers; the recorded source-group fallback is the immutable component row.
The canonical construction pool and the earlier five-axis assignment are not
the final 276/43/60 supervised partitions. The six-axis production selection,
review ledgers, and accepted-pair export are additional required dependencies.

The final training dependency is
`data/fullpaper_acl_pipeline/production_accepted652_reuse_export_reviewed_v2_20261004/`.
Its status is `frozen_for_first_experiment_not_clinically_adjudicated`;
`clinical_expert_adjudication` is false. The directory name describes its
upstream 652-pair review pool, not the final training size. The retained export
contains 276 TRAIN and 43 VALID pairs. The export records local rejudging,
candidate replacement, contextual integrity review, and a six-case assistant
review; none of these should be described as expert clinical adjudication.

All eight files below were checked against the frozen manifest:

| File | Rows | SHA256 |
| --- | ---: | --- |
| `sft_train.jsonl` | 276 | `3643253b185b31748c187235ae9502de95a77a35df54160430c6f80e3996f570` |
| `sft_valid.jsonl` | 43 | `e8b155cd7e420bb80df64e0fff76d0ca330fa736a39dbce7493cd3babedd6256` |
| `dpo_train.jsonl` | 276 | `79012eec4b9fac5055024bfaf230b42c723bc8fbda5202995d4b012d7bd31ed7` |
| `dpo_valid.jsonl` | 43 | `f8726b732dac7e42a197429f74696dc9119ee1229e6023b54446262cda19afa6` |
| `router_train.jsonl` | 276 | `b409033d8f0bcfc71f2b5d8acdb3f5cd8b77810f35490d1b72e45b355dd5954a` |
| `router_valid.jsonl` | 43 | `ab00eaa29b077aa6efa61c3aecd19ebdb71e05658ee72c12464f0b8651f8d736` |
| `scorer_train_verified_spans.jsonl` | 1590 | `90f536cba811a9010991eb130495093e6b40718a7dbef1c97beb240a5138a8b6` |
| `scorer_valid_verified_spans.jsonl` | 249 | `c9a975e37d4564d6dbd2a6012bbb1e395fd7fd30cecda77c70cf65670b20e37f` |

TEST60 comes from the frozen revision-2 test selection and
`frozen_revision2_test_partial60_surface_relaxed_v1_20261008/sft_test.jsonl`.
The manuscript reports 45 MentalChat16K and 15 CounselChat test questions.
The test partition is separate from the training/validation export. Exact
normalized-question, canonical-ID, source-group, and duplicate-cluster checks
must be retained when restoring these dependencies.
Direct inspection finds all 276 TRAIN rows have `duplicate_cluster_id`, with
275 distinct values. A shared within-TRAIN cluster is not a missing identifier
or evidence of cross-partition overlap.

The authoritative axis order is `scripts/fullpaper_risk_contract.py`, version
`fullpaper-risk-contract-v2-20260924`:

```text
overall_quality, empathy, specificity, factual_consistency,
medical_boundary, toxicity_or_harm
```

`intended_axes` describes requested corruption. Router `labels` describe the
observed adverse candidate-versus-reference changes after QC2; they are not a
copy of the requested axes. Values are 1, 0, or null. Null means unobserved and
is masked out of the loss. A Router label of 0 is a relative-change judgment,
not an absolute safety certificate. Scorer 1/0 labels require explicit
`local_defect`/`local_support` judgments of the exact passage in question
context. Holistic judgments and omissions do not create span labels.

Source access and redistribution of derived artifacts are different decisions.
This documentation change does not publish source counseling data, corrupted
pairs, evidence passages, response outputs, completed expert ratings, or model
checkpoints. The inspected local files and their hashes do not establish
permission to redistribute them. Access to an upstream source does not by
itself authorize redistribution of the derived training/evaluation corpus.

## Actual Construction And Execution Order

The following is the recorded dependency order. Original launchers contain
machine-specific paths and completed-step reuse checks; they require a reviewed
portable configuration before becoming a public quickstart. Do not run them
merely to check this document.

1. Inventory pinned source files with `scripts/prepare_fullpaper_phase1.py`;
   build the canonical pool and split with `scripts/fullpaper_acl_pipeline.py`.
   Preserve the exclusion and duplicate-group manifests.
2. Apply QC1. `scripts/run_clean_target_qc.py` uses the eligibility rubric in
   `scripts/run_paired_generator_diagnostic.py`; later production also applies
   `automatic_clean_preflight_local()` in
   `scripts/run_fullpaper_corruption_production.py`. Source integrity is
   governed by `scripts/source_integrity_contract.py`.
3. Construct bounded, sequential one-to-three-aspect corruptions with
   `scripts/run_fullpaper_corruption_production.py` and
   `scripts/corruption_contract_v2.py`. QC2 uses
   `scripts/local_qwen_production_qc_v4.py`; despite the filename, its recorded
   version is `local-qwen35-27b-paired-qc-v6-20260926`. A/B response identities
   are blinded and requested aspects are hidden from the paired judge.
4. Review/reuse or replace earlier accepted candidates with
   `scripts/rejudge_production_reuse_subset.py` and the production runner.
   Build the final reviewed export using
   `scripts/build_production_reuse_export.py` and
   `scripts/fullpaper_scorer_export_contract.py`. The prior review ledger,
   assistant review ledger, local rejudge checkpoint, replacement-production
   outputs, and scoped review ledger are inputs. Their hashes are recorded in
   the final export manifest; these are not recreated by simply rerunning
   corruption generation.
5. `scripts/run_arr_revision2_first_backbone.sh` records the model-stage order:
   train the two risk models with `scripts/train_fullpaper_risk_model.py`;
   select thresholds with `scripts/select_arr_valid_risk_thresholds.py`;
   train SFT with `scripts/train_professor_peft_refiner_textonly.py`;
   build SFT responses with `scripts/build_sft_outputs_for_risk_tuning.py`;
   enrich and train refinement with
   `scripts/train_gemma_risk_tune_from_sft.py`. Its `mask_off` variant is the
   manuscript's complete-response refinement. Other variants are ablations.
6. Export frozen test inputs with
   `scripts/build_frozen_revision2_test_partial_export.py`; generate SFT and
   invoked candidates using the saved final adapters and
   `scripts/run_gemma_selective_risk_refinement.py`. Retain raw generation,
   token IDs, finish reasons, source lengths, and response hashes.
7. Freeze/select candidates using
   `scripts/prepare_rubric_aligned_gpt41_rejudge.py` and
   `scripts/run_test60_rubric_aligned_selection.py`. The final manuscript
   artifact also requires `scripts/run_test40_extraction_fix.py`, which
   restores one saved SFT extraction and recomputes that case's gate/selection.
   Frozen request queues are executed by
   `scripts/run_gpt41_checkpoint_queue.py` with their recorded request, token
   and cost limits, then finalized by the selector scripts. This is a
   dependency description, not authorization to call the service.
   The extraction correction explains why the older manifest records 19 invocations but
   corrected final TEST60 records 18. Use the corrected v2 result, not v1.
8. Evaluate frozen responses with
   `scripts/run_corrected_test60_pointwise_publication.py`; use the independent
   Llama component-matrix provenance below for the manuscript's separate Llama
   table. Export blinded expert packets separately. Selection judgments are
   not substitutes for automatic or expert evaluation.

The corrected final file is
`test60_rubric_aligned_final_selection_20261009_v2_test40_fix/test60_final_outputs.jsonl`,
SHA256 `9f0e10d4a801ccc3a32ccab8b6d5032e67af6786a3b5cc066fb08cbecef4cfe4`.
This is a private dependency identifier, not an artifact-availability promise.

## Models And Frozen Configuration

| Component | Model and revision | Recorded configuration |
| --- | --- | --- |
| Corruption generator | `Qwen/Qwen3.5-4B`, `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Thinking disabled; sequential minimal edits |
| Later construction QC | `Qwen/Qwen3.5-27B`, `fc05daec18b0a78c049392ed2e771dde82bdf654` | Greedy; thinking disabled; 3200 output tokens; 300 s request timeout; no API fallback |
| Earlier construction QC | `gpt-4.1-2025-04-14` | Retain each historical request's own prompt/model metadata; not interchangeable with later local QC |
| SFT/refinement base | `google/gemma-4-E4B-it`, `ee0ef6023621cff504d758262d4e04895a5af4a2` | Saved local snapshot; text-only LoRA |
| Router and Scorer bases | `google-bert/bert-base-uncased`, `86b5e0934494bd15c9632b12f734a8a67f723594` | Separate pretrained encoders and newly initialized six-output heads |
| Candidate selector | `gpt-4.1-2025-04-14` | Temperature 0; JSON object; 256 output tokens per order |
| GPT pointwise evaluator | `gpt-4.1-2025-04-14` | Temperature 0; JSON object; 384 output tokens |

The production QC retry policy requires a one-point adverse score change for
quality/empathy/specificity/factual consistency, a one-point toxicity increase,
or clean-false/candidate-true medical-boundary status. All intended changes and
content checks must pass; additional observed changes are allowed. The recorded
later production cap is two semantic attempts per stage and four infrastructure
attempts, with a 2048-token truncation ceiling. Local and historical GPT
construction contracts must not be silently mixed.

The verified model-run root is
`outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/`.
Its `router/training_manifest.json`, `scorer/training_manifest.json`,
`gemma/sft/train_args.json`, and
`gemma/proposed_mask_off/final/risk_tune_config.json` supply these settings:

| Setting | SFT | Complete-response refinement |
| --- | --- | --- |
| Adapter initialization | New LoRA on pinned base | `gemma/sft/final` saved adapter |
| LoRA | r=8, alpha=16, dropout=0.05; q/k/v/o projections | Continued SFT adapter |
| Learning rate / epochs | 5e-5 / 3 | 5e-6 / 1 |
| Batch / accumulation | 1 / 16 | 1 / 8 |
| Source / target token limits | 1024 / 640 | 4096 / 640 |
| Prompt/input mode | `sft_plain` | `risk_contract=fullpaper_v1`, `component_mode=full`, `zt_strategy=no_mask` |
| Extra target-token weighting | Standard target-only cross entropy | `lambda_y=0`, so uniform target-token weights |
| High-risk sampling | None | Two copies of each high-risk example |
| Seed | 20260910 | 20260910 |

The Router/Scorer each use learning rate 1e-5, batch 8, seed 20260910 and
512-token training inputs. Recorded optimizer steps are 105/597 respectively.
During inference, Router input limit is 512 and sentence-Scorer input limit is
384. The six-output BCE is averaged over observed cells only.

`valid_risk_thresholds.json` records threshold selection on 43 VALID pairs,
using candidate/reference labels and balanced accuracy, then F1, distance from
0.35, and lower threshold as tie breakers. The full-model threshold is
`0.7215308056776166`. This is the frozen value used by the paper pipeline;
legacy default 0.35 is not its TEST threshold.

For each SFT sentence, the score is the maximum over matching aspect products
`Router(question, response)[k] * Scorer(question, sentence)[k]`. Response risk
is the largest sentence score, or zero for empty text. Invocation uses
`risk > threshold`, an empty response, fewer than 20 words, or a match from
`BAD_SAFETY_PATTERNS`. The inventory is in
`scripts/selective_risk_refinement_utils.py`; the frozen gate strategy is
`overall`, not the optional medical-only gate.

Both generation stages use greedy decoding, output budget 1536,
repetition penalty 1.15, and no-repeat-ngram size 4. The saved final adapters are
`gemma/sft/final` and `gemma/proposed_mask_off/final`. The refinement input
contains the complete SFT text twice: `sft_response` and `z_t_from_sft`.
The serialized historical heading still says `Risk-aware corrupted SFT
response`, but `no_mask` leaves that field complete. Risk scores condition the
input and oversampling; this production configuration does not mask passages
or apply nonuniform target-token risk weighting.

The implementation's oversampling comparison is `>= threshold`, while the
manuscript writes `> threshold`. This boundary difference is documented, not
silently fixed. No equivalence claim is made for hypothetical scores exactly
equal to the threshold.

## Original Prompts

These are source locations for the original prompts, not newly rewritten
instructions. Preserve literal content, axis order, chat-template application,
and recorded hashes when integrating them. Do not infer the production prompt
from a launcher's legacy filename.

| Stage | Original source of prompt/template |
| --- | --- |
| QC1 | `scripts/run_paired_generator_diagnostic.py`: `SYSTEM_PROMPT`, `ELIGIBILITY_PROMPT`; local preflight retry addition in `scripts/run_fullpaper_corruption_production.py` |
| Corruption | `scripts/corruption_contract_v2.py`: `STAGE_DIRECTIVES`, `LOCAL_SCORER_STAGE_DIRECTIVES`, `build_stage_prompt()` |
| Later QC2 | `scripts/local_qwen_production_qc_v4.py`: `SYSTEM_PROMPT`, `PAIRED_PROMPT`, `VALIDATION_RETRY_INSTRUCTION`, `paired_prompt()` |
| Source integrity | `scripts/source_integrity_contract.py`: `SOURCE_INTEGRITY_PROMPT`, `MEDICAL_BOUNDARY_DEFINITION` |
| Router / Scorer | `scripts/fullpaper_risk_contract.py`: `router_input_text()`, `scorer_input_text()` |
| SFT training | `scripts/train_professor_peft_refiner.py`: `build_plain_sft_user_content()`, `build_prompt()`; called by the text-only trainer |
| SFT generation | `scripts/selective_risk_refinement_utils.py`: `build_sft_user_content()`, `apply_chat_prompt()`, `build_sft_prompt()` |
| Complete-response refinement | Same utility: `build_risk_tune_user_content()`, `format_g()`, `build_risk_tune_prompt()` |
| GPT selector | `scripts/prepare_rubric_aligned_gpt41_rejudge.py`: `SELECTOR_SYSTEM_PROMPT`, `SELECTOR_USER_TEMPLATE` |
| GPT/Llama pointwise rubric | Same preparation script: `POINTWISE_SYSTEM_PROMPT`, `POINTWISE_USER_TEMPLATE`; exact frozen request messages are authoritative |

Later production run contracts record the following SHA256 fingerprints:

| Prompt material | SHA256 |
| --- | --- |
| Sequential generation template material | `86e92e45d248cfe851d0c89d8d7ef9739b03f9ef03d40395ed5138ac51df1cbf` |
| Historical clean-QC prompt material | `8a2dbf55eea7c8f2de4c2f8d8ded45531800127b49b7305a6dc1822c18a122f9` |
| Local clean-preflight template | `96af39b7c87669a11cebb886c8feeebbdbbb1c4406ec28094ad1a32bce04729e` |
| Typed local QC2 prompt material | `007dcaf2aa546a94b09849ecee59969ac54b72fa127767461bd35565417eb2ee` |
| Selector system | `d7e60ab17d748395b6e34c63bdeca93f0fc6a80621267f69184eb9242735b1e9` |
| Selector user template | `bed3ec1104eff8870a573e8b77dd11dd04f3259e756b9816e4d662088ba32e01` |
| Pointwise system | `c8dc19b5aa453e508890b70acf29f345a766d294d6572ddf5fc17ce4b29b4132` |
| Pointwise user template | `d23f153748ec56094ab7d7dbd7cbfa5d14c4f233b61e9e58b0748e8f2f957caf` |

The generation fingerprint aggregates rendered placeholder prompts across
axes. The local QC2 fingerprint includes system, paired template, and retry
instruction. These fingerprints are not hashes of every rendered request.
Rendered selector/evaluator requests additionally carry individual message and
response-pair hashes. Frozen contracts contain the full literal templates.
Portable publication must include the referenced source constants; this table
alone does not replace them.

## Input And Output Schemas

Every JSONL line is a JSON object. These are the observed field contracts;
examples intentionally contain no counseling text.

| Artifact | Required content and semantics |
| --- | --- |
| SFT pair | `id`, `canonical_id`, `question`, `unsafe_response`, `safe_response`; split/source-group/duplicate-cluster identifiers and question hash preserve provenance |
| Router pair | `canonical_id`, `question`, `candidate_response`, six-axis `labels` and `label_mask`; `axis_judgment` preserves observed-label provenance |
| Scorer passage | `canonical_id`, `question`, `span`, `span_start`, `span_end`, `span_side`, `span_source_sha256`, six-axis `labels`/`label_mask`; review/source records and provenance members track deduplication |
| SFT generation | Pair fields plus `sft_response`; raw output and generation metadata preserve extraction, finish reason, token IDs and source-length checks |
| Refinement training | SFT row plus `g_sft`, `sft_risk_score`, `sft_span_risks`, `z_t_from_sft`, `zt_strategy`, `aspect_conditioning_status`; `no_mask` requires complete equal SFT fields |
| Candidate generation | `sft_response`, `denoiser_response`, their risk scores, call flag/reason, generation metadata, rule-based acceptance fields |
| Corrected final TEST60 | `canonical_id`, `case_id`, `source_order`, `question`, SFT/DPO/candidate responses, `denoiser_called`, `gpt_selector_orders`, `gpt_selector_accepted_candidate`, `gpt_selector_selection_reason`, `final_response`, `final_response_source` |
| Selector result | `choice` is A/B/tie and `rationale`; presentation labels are mapped back to response identities independently in both orders |
| Pointwise judgment | Integer 1-5 `overall_quality`, `empathy`, `specificity`; categorical `medical_advice` yes/no/unsure; general and medical rationales and medical evidence |

Character offsets and source SHA256 refer to the exact response side that was
reviewed. A source corruption's passage offsets must not be transferred to a
new SFT response. Conflicting local passage/aspect labels are excluded by the
export contract. `train_fullpaper_risk_model.py` derives its effective observed
mask from non-null label values; do not encode an unobserved aspect as zero.

Candidate-generation `accepted_denoiser` or `current_rules_*` fields refer to
the earlier lexical/risk comparator. The manuscript's final answer instead
uses `gpt_selector_accepted_candidate`: adopt only when **both** A/B and B/A
orders select the candidate. Ties, disagreement, invalid or missing judgments,
missing candidates, and gate-off cases retain SFT. No additional scalar-risk
or medical-category veto is imposed by that GPT selector.

## Evaluation And Unresolved Provenance

The corrected GPT TEST60 evaluation uses
`test60_corrected_gpt41_pointwise_publication_20261009_v1/run_contract.json`
and `scripts/run_corrected_test60_pointwise_publication.py`. It evaluates
anonymous question-response pairs pointwise; exact duplicates reuse one
judgment. The three numeric qualities are averaged without medical labels.
Medical yes rates use the non-unsure denominator. Counts of medical
corrections and introductions require paired per-question labels, not only
aggregate yes rates. The selector and this evaluator share GPT-4.1, so this
evaluation is not independent of the selection model.

The paper's Llama main table is traceable to
`acl_component_ablation_matrix_20261009_v1/publication/main_baselines_llama.tex`
and `component_ablations_llama.tex`, implemented by
`scripts/run_acl_component_pointwise.py` and
`scripts/run_acl_component_llama_pointwise.py`. This matrix records 436
expected distinct assessment assignments, including 384 new results and
52 reused cache entries. Its main SFT/selected rows have 58 valid judgments
and quality means 3.149/3.328, matching the inspected manuscript table.
The local model is `meta-llama/Llama-3.1-8B-Instruct`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`; the matrix runner records
16384 input tokens and 384 output tokens per assessment. Its helper import
`run_valid_feedback_v1.py` comes from a sibling feedback worktree, not a
self-contained public module path.

However, the manuscript Appendix's 233 distinct assessments and 11 missing
medical-specific rationales describe a separate private
`manuscript_revision_fixed_policy_20261009/run_independent.py` run. That run
contains 60 SFT judgments with mean quality 3.2556, not the matrix's 58/3.149.
The table and Appendix provenance need reconciliation before claiming exact
reproduction of the independent evaluation. Missing-rationale inclusion and
evidence-conformance policies must be reported for the particular result set;
they cannot be transferred from one run to another.

The ordinary-draft dependency is the private
`manuscript_revision_fixed_policy_20261009/` directory, using `run_natural.py`,
`evaluate_natural.py`, and `run_independent.py`. Its Mistral snapshot is
`mistralai/Mistral-7B-Instruct-v0.3` revision
`c170c708c41dac9275d15a8fff4eca08d52bab71`; the question-only input uses system
message `You are a helpful assistant.`, greedy decoding, 16384 source tokens
and 768 output tokens. Existing outputs cover 60 ordinary drafts, 60 SFT
responses and 25 invoked candidates; Llama assessments cover 145 distinct
question-response pairs. The private scripts are not a verified public
repository entry point.

The located `natural_final` artifact uses the lexical/risk rule and records
seven adoptions. No frozen ordinary-draft two-order GPT decisions establishing
the manuscript's eight adoptions and final quality 3.506 were located. Those
selection and final-response claims remain unverified; ordinary candidate
generation alone does not establish them.

`scripts/build_rubric_aligned_expert_packet.py` and its packet manifest document
blinded forms, instructions and system mapping. Located TEST15x3 and TEST60x3
packets are not completed ratings for the six adopted questions. The completed
two-rater, six-question SFT/DPO/selected ratings, missing-value treatment,
agreement-analysis code and recruitment/compensation records were not located.
The archive also claims Korean instructions and an English translation in the
supplement; neither was found in its 16-file inventory. Their actual
production version and availability remain unverified.
The expert means, agreement statistics and procedural claims therefore remain
unverified by this workspace review. Do not treat packet generation as proof
that expert assessment was completed or publicly released.

## Source Fingerprints For Integration

These are SHA256 hashes of the **reviewed source snapshot**, not a claim that
every file's current bytes were captured at historical execution time. Use
them to compare the later A100 integration and review intentional differences.
The TEST corruption contract explicitly matches the current runner and local
QC hashes. Older regeneration contracts may pin different runner bytes.

All paths are repository-relative. Except for the two SFT trainer/helper files,
the listed primary files were absent from the documentation branch's `756f42d` base.
Presence and content after integration must be checked separately.

| Source path | SHA256 |
| --- | --- |
| `scripts/prepare_fullpaper_phase1.py` | `8eb19a3f1840c7b2681b3609d68c663d463498f6606551d03c7afa4357395131` |
| `scripts/fullpaper_acl_pipeline.py` | `74ff3a8a271c1d1442d5763f566290aa34c9f5875e9a8b2635b50c8fe760a5e8` |
| `scripts/run_clean_target_qc.py` | `df13e6141669f10a714f84ee325a68a608c46818f24354db96ef5c8de08b685c` |
| `scripts/run_paired_generator_diagnostic.py` | `572288faf87fcd4ba7c94b372537c6f0b64004593674049fabcca8e2cc7bbc8d` |
| `scripts/run_fullpaper_corruption_production.py` | `f4f5481bb7c3713582d5dab3ef098d9d703f110a8d2a48ae603ba0fc4a48b86f` |
| `scripts/local_qwen_production_qc_v4.py` | `b06e342838c21c33eea369ce8ec64e06effa5a0b26fd0f350b1e05de24faee3d` |
| `scripts/corruption_contract_v2.py` | `441f14b796d815d434fddde9380c24ff6c88c8b1b4c915f0260b57975fd2be41` |
| `scripts/source_integrity_contract.py` | `ebc35a802cfa735d863fb121a29d1cece4036f77134704c86d6f523231867ffe` |
| `scripts/rejudge_production_reuse_subset.py` | `387ca175eb92d94088550d4cf3ec2cc6026cd95bb6b36a879914b973c8aea5dc` |
| `scripts/build_production_reuse_export.py` | `6ce42d8b7b4a57ed310a04a6bc9a641e73e2fe772e254c9aa41c6f16dcddd55b` |
| `scripts/fullpaper_scorer_export_contract.py` | `d506f12d34e89415e03cfaca59a6baf080ed458fddfa48a90dd3f69289234973` |
| `scripts/run_arr_revision2_first_backbone.sh` | `877d9834dc845a7234f210928fe54f12cad314034b90864ea20bd71fb9461818` |
| `scripts/train_professor_peft_refiner_textonly.py` | `8f92d0cab089542845d657320496340847ea03c0fa0760935ca47d36d380821d` |
| `scripts/train_professor_peft_refiner.py` | `b7dfa42f6cfb227050462e4a1cc18230038e32826528366781bfbb0e6487a846` |
| `scripts/train_fullpaper_risk_model.py` | `f1d3a8ba11a888464c81776818962185202fa6b78c7edd98c1240d6202df3a25` |
| `scripts/fullpaper_risk_contract.py` | `0cc9656cf6c07d88f1fb84d4f3f0d61b082cb44d3e224a46b7c8f2c6be6c0c7b` |
| `scripts/select_arr_valid_risk_thresholds.py` | `3bb7b90eba16a7426bee346cf706e9d290974da1bf353668a523c960bbb24d5d` |
| `scripts/build_sft_outputs_for_risk_tuning.py` | `ad45155160961898614e2741a3a231895e45d6a635daf9975b44e75802b7fbe5` |
| `scripts/train_gemma_risk_tune_from_sft.py` | `a034a383226a09ee62dd9bcebf82b4f22e859208a7be56d9967f7442cb37be1b` |
| `scripts/selective_risk_refinement_utils.py` | `510bd8bc1d0b842f96676aa54d349489d96cdf3fdff6d235a0ab1cf366512581` |
| `scripts/run_gemma_selective_risk_refinement.py` | `b9625086dbb9cf9d6424d017a11fc37bb4a3ba10208fd7894d8e5f3b919c6a28` |
| `scripts/build_frozen_revision2_test_partial_export.py` | `087bbcf3ac571243bdb5f73d42937dd6bbeb4fded32945bb27663f0d26d0b0d9` |
| `scripts/prepare_rubric_aligned_gpt41_rejudge.py` | `d3f9c24436fab1c134e67e735c63d1ca3fe3e3911a44abb7d5941eeefd750185` |
| `scripts/run_test60_rubric_aligned_selection.py` | `6cb515b79797d39f9e645099121ac1085fc413658238475b6af3da735f1a3314` |
| `scripts/run_test40_extraction_fix.py` | `28a1f4b8212fd0c2a70e063cff4b8469964c2d44d6368d754c43653de0f6a3db` |
| `scripts/run_corrected_test60_pointwise_publication.py` | `f5abfe7a0205fd8dbb25862ecf86cd1802de4de3610380c5f8fe5edc5b413325` |
| `scripts/run_gpt41_checkpoint_queue.py` | `9579c506e27be4023deeb805d5e970cb8555a850e0f50e788efd082f1c8a5083` |
| `scripts/run_llama_rubric_aligned_rejudge.py` | `f90ce48e73567b8da00ebf87e54913d577b9aefef97e9e4c6b4cc925c02f17bb` |
| `scripts/run_acl_component_llama_pointwise.py` | `22ae33c6ee069e6a8bb7cfa60a5f322608922fd3584e09c98004f588d5869d82` |
| `scripts/run_acl_component_pointwise.py` | `318bf071d08432d58eb256e8932d41c5700003756af826da78216c4b4fc76c8b` |
| `scripts/build_rubric_aligned_expert_packet.py` | `b0d19af67d8ff7236b68f4d828cdf9172c1ed70f6e3fdccb7ea9f619808009d0` |

## Public Reproduction Limits

The base repository's dependency files do not establish an exact environment
lock for all production stages. Production launchers also reference a local
Python environment, locally cached snapshots, and local kernel support.
Package/CUDA/driver versions and portable path configuration require a separate
verified release manifest. No fresh installation or environment compatibility
test was performed here.

The inspected checkpoints and source/derived data are local dependencies, not
included release assets. Exact construction also depends on recorded reviews
and model/API outputs. Temperature zero does not itself establish identical
future hosted-model output. A complete public reproduction claim therefore
requires the integrated source, reviewed environment manifest, authorized
dependencies, and the unresolved evaluation/selection artifacts above.

CIKM short, exp295, early corruption pilots, masked reconstruction variants,
and unsuccessful/pilot training families remain preserved as historical work.
They are not interchangeable with the frozen ACL production configuration.
See [LEGACY.md](LEGACY.md) for the scope distinction. Anonymous release risks,
including machine paths and Git history, are tracked separately; no history
rewrite or dataset publication is authorized by this document.
