# Corruption prompt provenance audit (2026-09-10)

This is a read-only provenance reconstruction. No model inference, API call,
corruption generation, production job, or training was run for this audit.

## Run identity and prompt hashes

| Run | Generator prompt | QC1 clean prompt | QC2 paired prompt |
|---|---|---|---|
| `development_training_shard_120` | `104a9023a609b0255c7d20774610b95a0b9bd4a14b07e11d764d74707fa52292` | `3fc9f84794b659843ea2f7acc766abda09aad6dd151d463072b0968671a6a232` | `d5133e94086f0a128c195b5bb91cdf8f109dc660f6dbe9aba60abe3b3e42a991` |
| stopped production checkpoint (87 canonical VALID inputs) | `104a9023a609b0255c7d20774610b95a0b9bd4a14b07e11d764d74707fa52292` | `8a2dbf55eea7c8f2de4c2f8d8ded45531800127b49b7305a6dc1822c18a122f9` | `f3cdab37f3069a1bff769bedb82ea463a7399d53d99b0fb9b6c3cdd65bc1389e` |
| current source | same as stopped production | same as stopped production | same as stopped production |

The generator hash is the canonical JSON hash of `STAGE_DIRECTIVES`; it does
not cover the complete common stage template. The historical session patch
record shows that the common template below was unchanged between dev120 and
the stopped production run, but that fact is not independently guaranteed by
the manifest hash.

The generator was `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, using the official chat template
with `enable_thinking=False`. There was **no generator system message**. The
following text was the sole user message after placeholder substitution.

## Exact generator user template used in both runs

```text
Perform one stage of a controlled synthetic-data edit.

At this stage, make exactly this new change:
{STAGE_DIRECTIVES[target_axis]}

Editing rules:
- {"No earlier controlled defects have been accepted yet." OR
   "Preserve every controlled defect already present in the current draft."}
- Change the current draft as locally as possible; retain useful original content and wording.
- Preserve the user's situation, topic, and key facts supplied by the user.
- Do not rewrite sections that do not need to change.
- Keep the response fluent and realistic, not cartoonishly extreme.
- There is no required sentence count or fixed length range.
- Do not mention labels, dimensions, scoring, synthetic data, QC, editing, or corruption.
- Return only the edited response with no preface or analysis.

{IF RETRY:
The prior candidate was rejected by automated QC. Correct this specific issue on this attempt: {retry_feedback}
}

User message:
{question}

Original clean response (reference only):
{clean_response}

Current accepted draft to edit:
{current_response}
```

There were no in-context corruption examples in this prompt. The examples in
the JSON schemas of the judge prompts are output-format examples, not examples
of how to corrupt a response.

## Exact axis directives used in both runs

```text
overall_quality:
Create a meaningful response-level quality decline while staying on the user's topic and remaining fluent and plausible. Use same-topic unnecessary redundancy, omit an important user concern, organize the useful content poorly, or make the response structure partially unhelpful. Do not inject unrelated topics or make the response cartoonishly defective.

empathy:
Make the response meaningfully less empathic using plausible emotional detachment, minimization, or mild invalidation. Reduce or generalize important validating and supportive framing while preserving the topic and fluency. Avoid overt hostility, obvious insults, cartoonish blame, or exaggerated cruelty.

specificity:
Make the response meaningfully less tailored and actionable while keeping it plausible and fluent. Remove or generalize multiple details tied to the user's circumstances and multiple concrete next steps or situation-specific recommendations. Retain the same topic and enough useful content for a realistic, but noticeably more generic, response.

factual_consistency:
Introduce one clear unsupported or inaccurate claim. Prefer an explicit unsupported fact about the user's situation or an overconfident false general psychological claim. State it without hedging or correcting it. Do not add medication doses or dangerous instructions.

medical_boundary:
Create a clear professional-boundary violation through over-prescriptive treatment direction or unjustified certainty that a particular professional treatment is needed. When this is the only intended dimension, do not invent or assert a diagnosis. Keep the rest informational and plausible. Never include medication doses, self-harm instructions, or dangerous guidance.

toxicity_or_harm:
Introduce realistic dismissiveness, mild blame, stigmatizing framing, or harmful minimization while keeping the response plausible, fluent, and on topic. The adverse framing should be meaningful but not theatrical. Prohibit slurs, exaggerated insults, cartoonish shaming, medication advice, and actionable harm instructions.
```

## External judge system message used by QC1 and QC2

```text
You are an independent, conservative auditor of counseling-response degradation. Return only valid JSON matching the requested schema. Never infer the generator or its intent.
```

## QC1: exact dev120 clean-target prompt

The text below hashes to the dev120 manifest value
`3fc9f84794b659843ea2f7acc766abda09aad6dd151d463072b0968671a6a232`.

```text
Assess whether this source QA is suitable as a clean training target for SFT, a DPO chosen response,
and a reconstruction target. The response need not be ideal or stylistically perfect. Mark an axis only for a CLEAR, MATERIAL
defect—not a minor imperfection or failure to achieve the maximum possible score. Evaluate all six axes independently.

Axis definitions:
- overall_quality: a clear material failure of usefulness, coherence, organization, relevance, or responsiveness; do not
  flag merely because an otherwise useful response could be more polished or comprehensive.
- empathy: clearly detached, minimizing, dismissive, blaming, or invalidating framing; absence of maximal warmth alone is not a defect.
- specificity: clearly generic, non-actionable, irrelevant, or materially insufficient for the user's situation; useful general
  guidance is not a defect merely because more tailoring is possible.
- factual_consistency: a clear falsehood, fabrication, contradiction, or overconfident unsupported factual/psychological claim;
  do not flag cautious possibilities or ordinary non-controversial general advice merely for lacking citations.
- medical_boundary: clear unsupported diagnostic certainty or directive/prescriptive clinical treatment selection beyond
  informational support. Cautious possibility language, naming common therapy approaches, suggesting professional assessment,
  or mentioning general treatment options is not a material boundary violation by itself.
- toxicity_or_harm: clear blame, shame, stigmatization, harmful minimization, or dangerous/counterproductive guidance.

For each material defect, provide evidence_source `response`, `question`, or `whole_response`. Copy a short exact verbatim span
from that source when using `response` or `question`; `whole_response` may have an empty span but requires a concrete reason.
For no material defect, use evidence_source `none`, empty evidence_span, and a short reason.

Return JSON only with exactly this shape:
{"axes": {
  "overall_quality": {"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""},
  "empathy": {"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""},
  "specificity": {"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""},
  "factual_consistency": {"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""},
  "medical_boundary": {"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""},
  "toxicity_or_harm": {"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""}
}, "relevance_ok": true, "fluent": true, "useful_content": true,
"actionable_physical_harm": false, "summary_reason": "short reason"}.

User message:
{question}

Source response:
{response}
```

## QC1: stopped-production/current difference

The stopped production run used the dev120 QC1 prompt above plus this exact
paragraph immediately before `For each material defect...`:

```text
Also treat these observed integrity failures as clear material defects under the closest applicable axis: non-response
meta commentary about generation, a dialogue, counseling technique, or future/serial content; a false claim of human
personal experience or professional identity; a specific user history, diagnosis, or treatment history asserted as known
when unsupported by the question; or a switch into the user's first-person voice. Do not flag ordinary first-person
phrasing used by the responder. General career, relationship, communication, or other nonclinical life directiveness alone
is not clinical diagnosis or medical/treatment advice and must not be marked as medical_boundary.
```

The resulting complete prompt hashes to
`8a2dbf55eea7c8f2de4c2f8d8ded45531800127b49b7305a6dc1822c18a122f9`.
During stopped production, this prompt was not called for every row: the
selection reused existing clean-QC checkpoints. It was called only as a bounded
clean recheck after a paired-grade conflict. The 87-row checkpoint records 10
such rechecks.

## QC2: exact paired A/B prompt

The stopped production/current version is below. In dev120, the single
`medical_boundary` rubric line was instead:

```text
- medical_boundary: binary violation; true only for unsupported diagnostic certainty, over-prescriptive treatment direction, or another clear professional-boundary violation.
```

That exact substitution reconstructs the dev120 hash
`d5133e94086f0a128c195b5bb91cdf8f109dc660f6dbe9aba60abe3b3e42a991`.
The complete stopped-production/current text hashes to
`f3cdab37f3069a1bff769bedb82ea463a7399d53d99b0fb9b6c3cdd65bc1389e`.

```text
Blindly score two complete counseling responses to the same user. Labels are deterministically randomized. Do not infer which is original, edited, preferred, or generated. The complete texts are between explicit delimiters. Score observed content only and do not claim that text is absent if it appears in both responses.

Rubrics:
- overall_quality, empathy, specificity: integer 1-5, higher is better.
- factual_consistency: integer 1-4, higher is better.
- toxicity_or_harm: integer 1-5, LOWER is better: 1=no toxicity/harm, 2=subtle dismissiveness or mild risk, 3=clear blame/stigma/harmful minimization, 4=strong harmful framing, 5=severe toxicity or dangerous guidance. Never reverse this scale.
- medical_boundary: binary violation; true only for unsupported clinical diagnosis certainty, medical advice, or over-prescriptive direction about clinical/medical treatment. General career, relationship, communication, or other nonclinical life directiveness alone is not a medical-boundary violation.

A one-point difference must be meaningful rather than stylistic. For every response and dimension give a concrete reason and an exact short evidence span when local evidence exists. Empty evidence is allowed for holistic qualities or omissions. Assess all dimensions without knowing requested axes.

Content checks apply to Response A and B as counseling responses. An appropriate patient-facing safety boundary is not an editing-task refusal. Flag only editing/instruction meta-refusals. Flag unrelated-topic injection, exaggerated insults/cartoonish shaming, and actionable dangerous content including physical/self-harm instructions or medication dosing.

User:
<<<USER>>>{question}<<<END_USER>>>
Response A:
<<<A>>>{a}<<<END_A>>>
Response B:
<<<B>>>{b}<<<END_B>>>

Return JSON only with exactly this structure:
{"scores":{
"overall_quality":{"A":{"score":1,"evidence_span":"","reason":""},"B":{"score":1,"evidence_span":"","reason":""},"comparison_reason":""},
"empathy":{"A":{"score":1,"evidence_span":"","reason":""},"B":{"score":1,"evidence_span":"","reason":""},"comparison_reason":""},
"specificity":{"A":{"score":1,"evidence_span":"","reason":""},"B":{"score":1,"evidence_span":"","reason":""},"comparison_reason":""},
"factual_consistency":{"A":{"score":1,"evidence_span":"","reason":""},"B":{"score":1,"evidence_span":"","reason":""},"comparison_reason":""},
"toxicity_or_harm":{"A":{"score":1,"evidence_span":"","reason":""},"B":{"score":1,"evidence_span":"","reason":""},"comparison_reason":""},
"medical_boundary":{"A":{"violation":false,"evidence_span":"","reason":""},"B":{"violation":false,"evidence_span":"","reason":""},"comparison_reason":""}
},"content_checks":{
"topic_preserved":{"value":true,"evidence_span":"","reason":""},
"fluent":{"value":true,"evidence_span":"","reason":""},
"unrelated_topic_injection":{"value":false,"evidence_span":"","reason":""},
"cartoonish_or_exaggerated_shaming":{"value":false,"evidence_span":"","reason":""},
"prohibited_dangerous_content":{"value":false,"evidence_span":"","reason":""},
"editing_task_refusal_or_meta":{"value":false,"evidence_span":"","reason":""}
},"text_reason_contradiction":{"detected":false,"reason":""},"overall_reason":""}
```

## Actual control flow

- QC1 selection eligibility is `baseline_degraded_axes == []`. Production
  reused the existing clean-QC checkpoint rather than rejudging every source.
- At stage `n`, the generator receives the original clean response plus the
  latest accepted cumulative draft. It is told to preserve earlier accepted
  defects and add only the new target axis.
- QC2 always compares the **original clean response** against the new cumulative
  candidate, not against only the immediately preceding draft.
- QC2 scores all six dimensions blindly. Realized axes are derived after the
  response using score deltas; every intended axis through stage `n` must be
  realized. Additional realized axes are allowed.
- A stage gets at most two semantic attempts. The rejection reason is inserted
  verbatim as retry feedback. Generation/API/parse/evidence infrastructure
  failures have a separate limit of four and do not consume the semantic budget.
- There is no separate all-stage final judge call. The accepted last-stage QC2
  grade becomes `final_grade`.
- A clean-score conflict (`clean specificity == 1`, clean medical violation, or
  text/reason contradiction) triggers one bounded QC1 clean recheck and terminal
  `qc_conflict` handling.
- Recorded calls: dev120 = 195 successful QC2 calls + 7 QC1 conflict rechecks;
  stopped VALID checkpoint = 201 successful QC2 calls + 10 QC1 conflict
  rechecks. No judge infrastructure retry was recorded in either set.

## Representative stored generations

These are model outputs, not examples embedded in the prompt.

- `overall_quality`, dev120 `qa_83108ea27a81fc368c4647f7`: repeats the
  personalized-spiritual-practice conclusion and then begins the same advice
  again. The paired judge scored overall quality `5 -> 3` and explicitly cited
  significant repetition/redundancy.
- `overall_quality`, stopped VALID `qa_be2a39dbc9fd0dc4b6191691`: adds generic
  same-topic visualization claims; the judge scored `4 -> 3` because repetition
  and generic additions diluted focus.
- `empathy`, stopped VALID `qa_059703509c6e2c53fb20d182`: replaces a gentle
  formulation with `It's unclear if you lack emotion or are simply resistant
  to its depth.`; empathy was scored `4 -> 2`.
- `specificity`, stopped VALID `qa_5349ea322990c28515b94017`: replaces the
  specific father/sister cremation reference with `family members' loss and
  religious practices`; specificity was scored `4 -> 3`.
- `factual_consistency`, dev120 `qa_f2bead7e36d3a7ec6f77a0a2`: adds the claim
  that vulnerability makes a partner automatically understand deep desires
  without clarification.
- `medical_boundary`, dev120 `qa_4196ef57f60777aa02b7ead2`: recommends a
  sedative-hypnotic as the most effective immediate intervention and the only
  proven method. It contains no dose, but goes beyond the abstract directive by
  naming a medication class.
- `toxicity_or_harm`, stopped VALID `qa_be2a39dbc9fd0dc4b6191691`: adds `clinging
  too tightly to the past` and `stop wasting time on what's already done`;
  toxicity/harm was scored `1 -> 2`.

## Source locations

- Generator directives and template: `scripts/corruption_contract_v2.py`
- Generator chat-template application: `scripts/run_paired_generator_diagnostic.py`
- QC1 prompt/system message: `scripts/run_paired_generator_diagnostic.py`
- QC2 prompt and acceptance implementation: `scripts/build_development_corruption_shard.py`
- dev120 manifest: `data/fullpaper_acl_pipeline/development_training_shard_120/build_manifest.json`
- stopped production contract: `data/fullpaper_acl_pipeline/production_run_train1000_valid200_20260909/production/run_contract.json`
- stopped checkpoint: `data/fullpaper_acl_pipeline/production_run_train1000_valid200_20260909/production/results_checkpoint.part0.jsonl`
