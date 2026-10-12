# ACL ARR 2026 Reproducibility Record

This record describes the manuscript **SFT-First Selective Refinement for
Mental-Health Question Answering**, using the production artifacts inspected on
2026-10-12. It records evidence and remaining dependencies; it is not a claim
that a fresh public clone can reproduce every manuscript result.

The latest reviewed manuscript source was `acl_latex.tex`, SHA256
`204dc44ad24cf3ad4c3d8064c99434c0654b063feeb833d9c526106e960f26e3`.
Only that member changed between the two inspected manuscript archives;
their other 15 members and all reported table numbers are unchanged. The latest
revision clarifies the invocation/quality tradeoff and ordinary-input transfer,
reorders discussion, simplifies input notation, and changes table emphasis.
The manuscript itself was not edited or included in the reviewer export.
The integrated implementation now contains the construction/QC, data export,
risk models, SFT/refinement, and corrected TEST60 GPT selection/evaluation
entry points. Their presence, original prompt definitions and production
settings were checked statically against the saved execution evidence.
The source inventory distinguishes unchanged integrated implementation from
bounded adapters of the separately executed Llama and ordinary-draft routines.
Source presence does not establish availability of frozen data, weights or a
complete evaluation reproduction.

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
| Auxiliary Llama matrix | Raw results/cache, assignments, manifests, logs and publication tables | SFT/accept-all/selected 58/60 rows and paired transitions verified; bounded code adapters included |
| Ordinary drafts | Original generation/Llama records plus later frozen two-order GPT replay | 60 drafts, 25 candidates, 50 selector requests and eight adoptions linked to existing Llama judgments |
| Expert evaluation | Two completed six-question forms, extraction code and verified ratings table | Reported means verified locally; forms and internal extraction audit excluded from public export; agreement/procedural limits below |

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
   DPO is a separate SFT-initialized comparator: render the same frozen
   question/draft preference pairs with
   [render_fullpaper_dpo_for_backbone.py](../scripts/render_fullpaper_dpo_for_backbone.py),
   then train with [train_dpo_minimal.py](../scripts/train_dpo_minimal.py)
   using `--data_contract fullpaper`. Its legacy filename/docstring does not
   make this run the exp295 experiment.
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
| Candidate selector, including ordinary replay | `gpt-4.1-2025-04-14` | Temperature 0; JSON object; 256 output tokens per order |
| GPT pointwise evaluator | `gpt-4.1-2025-04-14` | Temperature 0; JSON object; 384 output tokens |
| Auxiliary Llama evaluator | `meta-llama/Llama-3.1-8B-Instruct`, `0e9e39f249a16976918f6564b8830bc894c89659` | Greedy; input 16384; output 384 for the matrix, 768 for the separate ordinary queue |
| Ordinary-draft generator | `mistralai/Mistral-7B-Instruct-v0.3`, `c170c708c41dac9275d15a8fff4eca08d52bab71` | Question only; helpful-assistant system message; greedy; input 16384; output 768 |

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

These are the frozen run settings, not an invitation to use every script's CLI
defaults. Some generic defaults retain earlier token limits, threshold 0.35 or
masked input modes. Reproduction must restore the recorded settings explicitly;
the integrated first-backbone launcher records the production overrides.

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

DPO uses the saved SFT adapter, beta 0.1, learning rate 1e-6, batch 1 and
gradient accumulation 16. The launcher computes three-epoch-equivalent steps
from the frozen training count and obtains complete prompt/completion limits
from the length audit rather than truncating preferences to a legacy default.
The saved DPO manifest records 54 steps, prompt limit 1024, completion limit
704 and seed 20260910.
The rendered files are `gemma/dpo_train_rendered.jsonl` and
`gemma/dpo_valid_rendered.jsonl`; checkpoint output is `gemma/dpo/final`.

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
The latest manuscript's shortened `Serialize(q,u,SFT,g)` notation does not
change this serialization: its figure, training text and Appendix still
describe both complete SFT fields.

The implementation's oversampling comparison is `>= threshold`, while the
manuscript writes `> threshold`. This boundary difference is documented, not
silently fixed. No equivalence claim is made for hypothetical scores exactly
equal to the threshold.
There is also a literal lexical-inventory difference: the latest Appendix
prints `you're`, whereas the frozen stopword set in
`scripts/selective_risk_refinement_utils.py` contains `youre` and the tokenizer
retains apostrophes. Neither the manuscript nor frozen rule was changed here.

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
| Ordinary drafts | `scripts/run_natural.py`: system message `You are a helpful assistant.` and the question alone; no reference or deliberate corruption in the generator input |
| Ordinary GPT selection | `scripts/run_ordinary_gpt_selector.py` reads the original frozen selector contract and checks both prompt hashes; no new prompt file or rewritten rubric |

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

## Evaluation Provenance And Execution Order

### Auxiliary Llama TEST60

The exact table source is the saved `acl_component_ablation_matrix_20261009_v1`
artifact root, not a new evaluation. Its identity chain is:

```text
inputs/test60_sft_frozen.jsonl + inputs/test60_main_proposed_frozen.jsonl
results/final_method_outputs.jsonl
  -> inputs/all_pointwise_assignments.jsonl
  -> requests/expected_pointwise_requests.jsonl
  -> requests/new_llama_pointwise_requests.jsonl
  -> results/new_llama_pointwise_results.jsonl
     + results/reused_llama_pointwise_cache.jsonl
  -> results/llama_assignment_scores.jsonl
  -> tables/llama_final_quality_summary.csv
     + tables/llama_final_all_pairs.jsonl
```

[run_acl_component_llama_pointwise.py](../scripts/run_acl_component_llama_pointwise.py)
prepares/runs the local queue; the retained scoring and aggregation routines in
[run_acl_component_pointwise.py](../scripts/run_acl_component_pointwise.py)
join exact question/response identities and exclude schema-invalid judgments.
The original publication/ZIP packaging and unrelated risk-scoring operations
are not part of that bounded public adapter. Existing model loading, chat
rendering, EOS handling and JSON parsing functions are shared through
[acl_local_judge_utils.py](../scripts/acl_local_judge_utils.py), extracted from
the helper actually used by the historical run, not a new judge implementation.

The saved contract is `llama_pointwise_contract.json`; completion is recorded in
`results/new_llama_pointwise_results.manifest.json` and `logs/postprocess.log`.
There are 436 unique question-response requests: 384 new request identities
plus 52 exact cached judgments. Repeated JSONL attempts are resolved by the
original request-ID logic, not counted as independent judgments. The completed
new queue has 360 schema-valid and 24 invalid judgments. Final batch size was
16; the runner's generic default is not the recorded invocation setting.

| Final condition | Stored method | Valid / expected | U | Medical yes |
| --- | --- | ---: | ---: | ---: |
| SFT | `SFT` | 58 / 60 | 3.149 | 19 |
| Accept all invoked | `GatedNoSelector` | 58 / 60 | 3.155 | 19 |
| Selective | `Proposed` | 58 / 60 | 3.328 | 18 |

The selective/SFT common set has 58 judgments, with two yes-to-no and one
no-to-yes transition. Accept-all/SFT has 57 common judgments, with three of
each transition. These records match the manuscript. The separate historical
233-pair collection and its 11 missing no-label medical rationales have their
own saved `independent_*` queue and extraction convention; they are not the
matrix table inputs. The latest Appendix distinguishes collection sizes and
condition-specific valid denominators. Do not replace one run with another or
carry its missingness policy into a different table.

### Ordinary Drafts And Frozen GPT Replay

The original ordinary generation/evaluation root is
`manuscript_revision_fixed_policy_20261009/`. In dependency order:

1. [run_natural.py](../scripts/run_natural.py) reads the held-out question
   identities from `canonical_input.jsonl` and writes `natural_drafts.jsonl`.
   Its Mistral prompt contains only the helpful-assistant system message and
   question. It reuses the existing SFT/refinement entry points to write
   `natural_sft.jsonl` and `natural_final.jsonl`, with their manifests and
   `natural_sft.log` / `natural_refinement.log`. There are 60 drafts and
   SFT responses, 25 invoked candidates, and seven rule-based adoptions.
2. [evaluate_natural.py](../scripts/evaluate_natural.py) reuses the local
   [run_independent.py](../scripts/run_independent.py) queue routines. It writes
   `natural_judge_requests.jsonl`, `natural_judge_assignments.jsonl`,
   `natural_judgments.jsonl`, `natural_judgments.manifest.json` and
   `natural_evaluation_summary.json`; `natural_evaluation.log` records the run.
   There are 145 distinct pairs: 60 drafts, 60 SFT responses and 25 candidates.
   All have integer quality scores and medical categories. Seventeen no-label
   judgments omit a medical-specific rationale; the historical ordinary table
   retains their returned scores/categories without imputation.
   Its rubric is the corrected pointwise publication contract,
   `test60_corrected_gpt41_pointwise_publication_20261009_v1/run_contract.json`,
   with `prompts.system_prompt` and `prompts.user_template`, not the selector
   replay contract.
3. The separate subsequent `ordinary_gpt_selector_20261011_v1/` artifact
   applies the frozen GPT selector to those same 25 candidates. Its
   `run_contract.json` records 11 source-file hashes and the request hash.
   Every original source and preserved `source_snapshot/` copy matches those
   hashes. Both prompt strings and every rendered request match the original
   frozen contract. `requests.jsonl` / `results.jsonl` contain 50 successful
   comparisons, two orders per candidate. `results.manifest.json` is complete.
4. `selector_decisions.jsonl` records eight candidates winning both orders,
   15 SFT wins and two disagreements. `final60.jsonl` and `final60_audit.csv`
   map exact final text hashes to existing Llama evaluation IDs. The final
   scores are inherited from `natural_judgments.jsonl`, not new pointwise calls.
   `aggregate.json`, `validation.json` and the saved `ordinary_aggregation.log`
   record the checks and summaries. A separate selector stdout log was not
   located; the raw API results and completed manifest are present.

The preserved [run_ordinary_gpt_selector.py](../scripts/run_ordinary_gpt_selector.py)
is a verification/aggregation adapter of that actual subsequent script. It
does not prepare requests or call the API. It requires the original private
snapshot/contract and evidence audit; it writes only to a new output directory,
never back into the frozen run. Original prompt definitions remain in the
existing preparation module, rather than a duplicate prompt file.

| Ordinary condition | Adoptions | U | Medical yes |
| --- | ---: | ---: | ---: |
| SFT | 0 | 3.311 | 21 |
| `LegacyStrict` risk/lexical comparator | 7 | 3.228 | 22 |
| GPT-Select | 8 | 3.506 | 20 |

The GPT-selected/SFT change is +0.194 with stored paired interval [0.000, 0.428],
two medical corrections and one introduction. The lower endpoint is numerical
zero, not a strictly positive lower bound. The ordinary interval uses 20,000
samples and seed 20261011; the matrix uses 10,000 and seed 20261009. Bootstrap
was not rerun during this review. Score-complete and strict-schema summaries
are saved separately; the full-schema GPT final marginal has 57/60, and its
SFT common subset has 56. Do not substitute them for the historical N=60 row.
The original evidence-mismatch flags are preserved, not repaired.

### Expert Assessment

Two completed six-question/18-response forms were located and their hashes
match the recorded input manifest. The existing private extraction routine
maps anonymous packet identities to SFT/DPO/selected responses and writes
`expert_ratings_verified.csv`, `expert_extraction_audit.json` and
`revision_audit.json`. The saved U means match the manuscript: rater 1 has
SFT 2.444 and selected 3.611; rater 2 has SFT 2.111 and selected 4.000.
These forms concern the six corruption-reconstruction adoptions, not the
eight ordinary-draft adoptions. Completed forms, response text, filenames with
rater identities and internal extraction/audit documents are not public assets.

The two original forms contain identical Korean study-instruction blocks:
anonymous independent A/B/C ratings, 1-5 quality, categorical medical labels,
and English response quotations for yes/unsure evidence. The instruction-only
block hash is `a3ce7d0002eb06b164ec16adcac2be3b40ad6ef7d40db3f596d8a7dc7503a99e`.
The complete English instruction translation, agreement-analysis code/execution
records and independent recruitment/compensation records remain unverified.
Their absence from the inspected public sources is an availability limit, not
a judgment that the reported evaluation is incorrect. No agreement statistics
were recomputed and no unlocated instructions were invented for this release.

## Source Inventory

These are content fingerprints, not a claim that every included file's current
bytes were captured at historical execution time. The table records unchanged
integrated-source hashes. DPO entry points already present in the repository
are linked above; bounded evaluation adapters are described below.
All code paths are repository-relative.

Three included files differ from the earlier inspected source. The response
utility now strips standalone/Markdown response labels and enclosing quotes;
its prompt templates are unchanged, but extracted response bytes can differ.
The refinement runner adds the explicit `surface_relaxed_v1` policy and logging
alongside the legacy policy. The queue reader uses `splitlines()` rather than
splitting only on newline. These changes prevent a blanket byte-equivalence
claim for a historical run. The TEST corruption runner and typed QC source
fingerprints still match their recorded contract. Frozen response identities
and the corrected TEST60 selector remain the evaluation dependencies.

| Source path | Availability | SHA256 |
| --- | --- | --- |
| `scripts/prepare_fullpaper_phase1.py` | Included | `8eb19a3f1840c7b2681b3609d68c663d463498f6606551d03c7afa4357395131` |
| `scripts/fullpaper_acl_pipeline.py` | Included | `74ff3a8a271c1d1442d5763f566290aa34c9f5875e9a8b2635b50c8fe760a5e8` |
| `scripts/run_clean_target_qc.py` | Included | `df13e6141669f10a714f84ee325a68a608c46818f24354db96ef5c8de08b685c` |
| `scripts/run_paired_generator_diagnostic.py` | Included | `572288faf87fcd4ba7c94b372537c6f0b64004593674049fabcca8e2cc7bbc8d` |
| `scripts/run_fullpaper_corruption_production.py` | Included | `f4f5481bb7c3713582d5dab3ef098d9d703f110a8d2a48ae603ba0fc4a48b86f` |
| `scripts/local_qwen_production_qc_v4.py` | Included | `b06e342838c21c33eea369ce8ec64e06effa5a0b26fd0f350b1e05de24faee3d` |
| `scripts/corruption_contract_v2.py` | Included | `441f14b796d815d434fddde9380c24ff6c88c8b1b4c915f0260b57975fd2be41` |
| `scripts/source_integrity_contract.py` | Included | `ebc35a802cfa735d863fb121a29d1cece4036f77134704c86d6f523231867ffe` |
| `scripts/rejudge_production_reuse_subset.py` | Included | `387ca175eb92d94088550d4cf3ec2cc6026cd95bb6b36a879914b973c8aea5dc` |
| `scripts/build_production_reuse_export.py` | Included | `6ce42d8b7b4a57ed310a04a6bc9a641e73e2fe772e254c9aa41c6f16dcddd55b` |
| `scripts/fullpaper_scorer_export_contract.py` | Included | `d506f12d34e89415e03cfaca59a6baf080ed458fddfa48a90dd3f69289234973` |
| `scripts/run_arr_revision2_first_backbone.sh` | Included | `877d9834dc845a7234f210928fe54f12cad314034b90864ea20bd71fb9461818` |
| `scripts/train_professor_peft_refiner_textonly.py` | Included | `8f92d0cab089542845d657320496340847ea03c0fa0760935ca47d36d380821d` |
| `scripts/train_professor_peft_refiner.py` | Included | `b7dfa42f6cfb227050462e4a1cc18230038e32826528366781bfbb0e6487a846` |
| `scripts/train_fullpaper_risk_model.py` | Included | `f1d3a8ba11a888464c81776818962185202fa6b78c7edd98c1240d6202df3a25` |
| `scripts/fullpaper_risk_contract.py` | Included | `0cc9656cf6c07d88f1fb84d4f3f0d61b082cb44d3e224a46b7c8f2c6be6c0c7b` |
| `scripts/select_arr_valid_risk_thresholds.py` | Included | `3bb7b90eba16a7426bee346cf706e9d290974da1bf353668a523c960bbb24d5d` |
| `scripts/build_sft_outputs_for_risk_tuning.py` | Included | `ad45155160961898614e2741a3a231895e45d6a635daf9975b44e75802b7fbe5` |
| `scripts/train_gemma_risk_tune_from_sft.py` | Included | `a034a383226a09ee62dd9bcebf82b4f22e859208a7be56d9967f7442cb37be1b` |
| `scripts/selective_risk_refinement_utils.py` | Included | `58c129810f51de62e3a6f98ba56105b7836526e821b221e0f39da856fc3ca020` |
| `scripts/run_gemma_selective_risk_refinement.py` | Included | `1da35327b4609ecada24ffff08f66d4df8e14f718c083c7696daa43d8c9439c8` |
| `scripts/build_frozen_revision2_test_partial_export.py` | Included | `087bbcf3ac571243bdb5f73d42937dd6bbeb4fded32945bb27663f0d26d0b0d9` |
| `scripts/prepare_rubric_aligned_gpt41_rejudge.py` | Included | `d3f9c24436fab1c134e67e735c63d1ca3fe3e3911a44abb7d5941eeefd750185` |
| `scripts/run_test60_rubric_aligned_selection.py` | Included | `6cb515b79797d39f9e645099121ac1085fc413658238475b6af3da735f1a3314` |
| `scripts/run_test40_extraction_fix.py` | Included | `28a1f4b8212fd0c2a70e063cff4b8469964c2d44d6368d754c43653de0f6a3db` |
| `scripts/run_corrected_test60_pointwise_publication.py` | Included | `f5abfe7a0205fd8dbb25862ecf86cd1802de4de3610380c5f8fe5edc5b413325` |
| `scripts/run_gpt41_checkpoint_queue.py` | Included | `fb089db0ba1fdddb08dc1981fb10530deb7b1edfdcf7af3f07eb601f62649ead` |

These next fingerprints identify the **historical source**, not the portable
adapters. Resource paths/imports and entry-point/output handling are adapted;
numerical scoring/selection routines are preserved and compared statically.
The local export manifest records the actual public file hashes.

| Included adapter | Historical source SHA256 | Scope |
| --- | --- | --- |
| `scripts/run_acl_component_llama_pointwise.py` | `22ae33c6ee069e6a8bb7cfa60a5f322608922fd3584e09c98004f588d5869d82` | Matrix local queue and original strict parser |
| `scripts/run_acl_component_pointwise.py` | `318bf071d08432d58eb256e8932d41c5700003756af826da78216c4b4fc76c8b` | Request/cache joins, scoring, summaries and paired aggregation; excludes packaging and unrelated risk scoring |
| `scripts/run_ordinary_gpt_selector.py` | `c29f521ef00892376d17782d6efbef0b5c480aa9078aa6c0583decb528c4aad2` | Existing replay verification and aggregation only |
| `scripts/run_independent.py` | `003507260448bad8a49371c37be04e3581c26c0d28f90a6e46a6db9bbba31c34` | Separate local pointwise queue, including ordinary evaluation |
| `scripts/run_natural.py` | `8caa2e37536586cc43f103bd24b7a5bfbbda6af7dbf6d0da48d957a5a4630d27` | Original question-only generation and existing reconstruction/refinement drivers |
| `scripts/evaluate_natural.py` | `8cccf9c5656b8028f5503fb2e03d10031bd249851242d600ceb01187cc5e4996` | Original ordinary assignment and local-evaluation wrapper |
| `scripts/acl_local_judge_utils.py` | `d912f82ca3d4539155714bfb2873f95d10f3461992d8e4329dde39d735dc6432` | Only required helper functions from the historical development runner; not its full experiment |

Historical local cache keys include the model's filesystem path. A new portable
path is not byte-identical to that historical cache identity; use saved results
and request IDs for archival aggregation. Do not treat a relocated model or
adapted source hash as an identical fresh run. The original development
sources and all frozen artifacts remain unchanged outside the reviewer export.

### Adapter Interfaces

Paths below are explicit dependencies, not bundled artifacts. The commands
document interfaces only; no generation, evaluation or aggregation run was
performed for this source release. `--help` and syntax checks do not load models.

| Entry point | Required resources and modes |
| --- | --- |
| `run_acl_component_llama_pointwise.py` | `prepare` or `run`; `--root MATRIX --model LLAMA_SNAPSHOT`; use recorded `--batch-size 16`; exact cache roots via repeated `--cache-root` |
| `run_acl_component_pointwise.py` | `aggregate --root MATRIX` reads saved evaluations; `prepare` additionally needs `--frozen-contract-dir` and reviewed cache roots |
| `run_independent.py` | `--root INDEPENDENT --model LLAMA_SNAPSHOT --batch-size 8`; historical separate pointwise queue |
| `run_natural.py` | `--source-dir ORDINARY --checkpoint-dir RUN --mistral-model MISTRAL_SNAPSHOT --gemma-model GEMMA_SNAPSHOT`; expensive generation, not a documentation check |
| `evaluate_natural.py` | `--source-dir ORDINARY --rubric-contract CORRECTED_POINTWISE_CONTRACT --llama-model LLAMA_SNAPSHOT --batch-size 8`; contract keys `prompts.system_prompt` / `user_template`; expensive local judging |
| `run_ordinary_gpt_selector.py` | `verify --run-dir REPLAY` checks the saved snapshot and requests only; `aggregate` additionally requires `--evidence-audit` and a new `--output-dir`; no API or request-preparation mode |

The ordinary adapter validates all 11 original snapshot hashes and the frozen
request hash, rather than weakening the original checks to accept modified
inputs. It also checks all 50 rendered prompts and question/response identities.
Its aggregate outputs must go to a new directory outside the frozen run and
snapshot. Natural generation/evaluation drivers refuse existing outputs before
loading models; use a new authorized working directory, not a frozen run or
a resume workflow. An authorized user
still needs the private replay snapshot, judgments and evidence audit; this
release does not make those inputs available.
Matrix and independent queue/aggregation routines retain historical append or
output-writing behavior under `--root`. Use authorized working copies of the
required artifacts, never the frozen archival roots. Their presence is not a
claim that every entry point is read-only.

## Public Reproduction Limits

No tracked dependency/environment lock was verified for these production
stages. Ordinary verification/aggregation imports NumPy; local model loading
requires PyTorch, Transformers and Accelerate, and the existing model-training
drivers have additional dependencies. Production launchers also reference a local
Python environment, locally cached snapshots, and local kernel support.
Package/CUDA/driver versions and portable path configuration require a separate
verified release manifest. No fresh installation or environment compatibility
test was performed here.
Historical selection/extraction launchers also retain original commit and
provenance checks in development. The reviewer copy replaces only the
development-commit literal with `PRIVATE_PROVENANCE_COMMIT`; model and data
revision pins are unchanged. A history-free source export does not satisfy
those private execution dependencies merely by configuring model and data mounts.

The inspected checkpoints and source/derived data are local dependencies, not
included release assets. Exact construction also depends on recorded reviews
and model/API outputs. Temperature zero does not itself establish identical
future hosted-model output. A complete public reproduction claim therefore
requires the integrated source, reviewed environment manifest and authorized
dependencies. The target Llama and ordinary-selection provenance is verified,
but their private inputs/results and the expert study are not bundled assets.

CIKM short, exp295, early corruption pilots, masked reconstruction variants,
and unsuccessful/pilot training families remain preserved as historical work.
They are not interchangeable with the frozen ACL production configuration.
Older development notes are retained separately from reviewer-facing materials.
From the development checkout, `scripts/export_acl_review.py` creates a new
source-only directory using an explicit allowlist. The exporter itself is a
development release tool, not included in the reviewer source directory. It
excludes Git/history, manuscript archives, internal documents, legacy notes,
data, completed expert ratings, checkpoints and logs, and checks syntax,
dependency closure and path/identity exclusions. Exported machine-path literals
refer to anonymous `/workspace` mounts; they are configuration placeholders,
not available weights/data. The reviewed development-commit literal is also
neutralized only in the export. No prompt or model setting is anonymized away.
Only README and this guide are reviewer-facing documentation.

The development GitHub repository and its history are not an anonymous mirror.
Actual mirror contents, authentication-free reviewer access and environment
compatibility remain separate release checks. This document does not authorize
a history rewrite, dataset publication or upload of private evaluation records.
