# ACL Documentation Verification Record

Date: 2026-10-12 (client date, Asia/Seoul). Scope: static inspection of the
manuscript, source, saved contracts and manifests. This is not a new training,
generation, paid-judge, or end-to-end reproduction run.

## Revision Boundaries

| Item | Inspected revision or status |
| --- | --- |
| Public `origin/main`, initial check | `756f42d7b912c862cc98bd58d46fb37fc7ad39dc` |
| Public `main`, pre-commit live remote recheck | `756f42d7b912c862cc98bd58d46fb37fc7ad39dc`, unchanged |
| Documentation branch | `docs/acl-arr-2026-h100`, based on that public-main snapshot |
| Separate H100 production-source checkout | `ec69e056873f72f8277648d2ec5318871051120b` |
| Manuscript archive SHA-256 | `36cb997415a42b2a5f186474fcf1d49f91db489b0351ec207a4c7d407c6e8eee` |
| A100 integration completion | Not established; do not infer completion from an unchanged remote HEAD |

The documentation branch changes prose and a proposed Appendix addition only.
The development repository and this PR are identifying collaboration artifacts,
not the anonymous reviewer export. The manuscript archive is retained unchanged.

## Evidence-Based Checks

The reproducibility document separates saved execution evidence, static source
inspection and manuscript-only claims. In particular:

- The reviewed training export has 276 training and 43 validation pairs and
  1,590/249 Scorer passages. Recorded counts and hashes were compared with the
  local frozen files without exporting their counseling text.
- The corrected TEST60 reports and contracts record 18 invocations and 6 GPT
  two-order adoptions. The post-extraction-correction report takes precedence
  over intermediate generation counts. TEST100 remains a separate extension.
- Production configuration uses complete-response, mask-off refinement with
  uniform target-token loss weights. Original prompts and model/settings
  identities are located in the source/contract inventory, not reconstructed
  from historical CIKM notes.
- Primary pointwise evaluation and selection both use GPT-4.1. The Llama paper
  table and the separate independent rerun are not interchangeable provenance.
- The tracked manuscript ZIP was inspected without changing it. Targeted
  identity/credential checks and the unauthenticated mirror access check are
  documented in [ACL_ANONYMIZATION.md](ACL_ANONYMIZATION.md).

## Unresolved Reproduction and Release Checks

1. Reconcile the source-hash inventory with the **completed A100 integration**,
   including imported-helper closure, exact prompts, model revisions, context
   limits and evaluation parsers. The initial public-main tree lacks most of
   these files. A path appearing in a later commit alone is not equivalence.
2. Resolve the Llama table/Appendix provenance discrepancy; locate the ordinary
   draft GPT selector decisions and aggregation producing the manuscript result.
3. Locate completed expert ratings, agreement calculations and the claimed
   Korean/English expert instruments. Blank export packets do not verify the
   reported ratings or supplementary availability.
4. Reconcile the high-risk oversampling boundary notation described in the
   reproducibility document. TRAIN cluster IDs are populated on all 276 rows
   with 275 distinct values; the repeated within-TRAIN value is not missing data.
5. Establish a tested dependency manifest and access to the frozen inputs,
   weights and services. No clean-checkout rerun was attempted. Dataset and
   derived-artifact redistribution are not authorized by this update.
6. Finish anonymous-export cleanup and inspect its actual files, download and
   reviewer access. The observed mirror response was HTTP 401; an available,
   anonymized mirror is not established.

No GPU training, generation, research API calls, process interruption, frozen
TEST changes, dataset publication, or history rewrite is part of these checks.

## Final Static Check Results

- All 31 reviewed source fingerprints match the separate H100 source checkout.
  AST parsing passed for 30 Python sources and `bash -n` passed for the launcher.
  These checks do not import dependencies or execute the research workflow.
- Of those 31 paths, live public `main` is missing 29. Its two present files,
  `train_professor_peft_refiner_textonly.py` and
  `train_professor_peft_refiner.py`, differ from the reviewed source. None of the
  31 is established as source-equivalent in that public-main snapshot.
- All 26 checked local Markdown links resolve across the README, ACL documents,
  legacy index and five retained historical notes.
- The five historical documents retain their headings and complete original
  bodies; only legacy banners were added. No experiment source was modified.
- Manuscript source SHA-256 matches the reproducibility record and the archive
  checksum above. The archive was not modified.
- Documentation whitespace checks pass. Targeted credential-pattern checks
  found no high-confidence credential values in the new text; a substring of
  the version name `fullpaper-risk-contract-v2-20260924` is not an API key.
- Final live remote-main inspection still resolves to the initial SHA. A100's
  completed integration and equivalence with it remain **unverified**. The
  documentation PR is a draft pending that comparison and the stated blockers.

After A100 reports completion, fetch its specific remote revision into a
separate checkout, compare this inventory and each documented contract against
that tree, and review any intentional differences. Update this record and the
README before claiming integration or a runnable anonymous release. No such
post-completion check is recorded here.
