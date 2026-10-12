# ACL ARR Anonymous Artifact Audit

Audit date: 2026-10-12 (Asia/Seoul). This is an internal release-preparation
record, not a certification that an anonymous artifact has been published.
Do not include this audit in the anonymous review export: it points to
identity-bearing material in the development repository.

## Scope and Status

The inspected documentation branch starts from the remote `main` snapshot
`756f42d`. That snapshot contains the older CIKM implementation and manuscript
archive, not the latest ACL production implementation. Production code was
also inspected read-only in the H100 handoff checkout at `ec69e05` and its
referenced sibling paths. A100 integration and a final comparison with its
completed `main` remain separate requirements; neither is established by
this audit. Nothing in the manuscript archive, source scripts, datasets,
checkpoints, Git history, or mirror settings was changed by this audit.

The [ARR call for papers](https://aclrollingreview.org/cfp#instructions-for-two-way-anonymized-review)
requires anonymous submissions and supplementary materials, including linked
repositories. The [author checklist](https://aclrollingreview.org/authorchecklist)
also calls for removing acknowledgements and revision leftovers and avoiding
links to non-anonymous repositories. Non-anonymous preprints are permitted
under the [updated anonymity policy](https://aclrollingreview.org/anonymity/);
that does not make an identifying review artifact anonymous. These official
pages were checked on the audit date.

## Findings and Proposed Fixes

| Surface | Verified observation | Proposed review-export treatment | Status |
| --- | --- | --- | --- |
| `ACL_Hyunseo (2).zip` | The tracked archive filename contains a personal identifier. It contains 16 manuscript/template/asset files, including source-only revision notes. | Keep the development original intact. Exclude it from the code export. If manuscript source is supplied separately, create a reviewed copy with a neutral filename and sanitized contents. | Blocking if exported unchanged |
| Archive `acl_latex.tex` | `\usepackage[review]{acl}` and `\author{Anonymous ACL Submission}` are present. Lines 17-30 contain an `\iffalse` revision-note block with an advisor reference. The main source has no acknowledgements section or identified affiliation/email in the targeted scan. | Remove the revision-note block only in a separate reviewed submission-source copy. Retain review mode and anonymous author. Inspect the final compiled PDF separately. | Source cleanup pending; compiled manuscript PDF unavailable |
| Archive bibliography | The bibliography names cited researchers and dataset/model creators. No direct match to the inspected owner identifiers was found. The PAIR-SAFE citation is a third-party work; surname matches alone do not establish self-identification. | Preserve valid citations and third-person discussion. Review actual self-citations with author knowledge, rather than deleting third-party attribution. | Targeted scan complete; author relationship not inferred |
| Archive PDF figures | All four figure PDFs were checked with `pdfinfo` and `pdftotext -layout`. Metadata identifies presentation/plotting software; no personal author name was found. `Figure1_vector.pdf` has a software author and creation timestamp. Extracted figure text contains no targeted owner identifier or external repository URL; extraction can miss image-only text. | Inspect the final compiled PDF, including visible text and document properties, in the separate submission copy. | Figure metadata/text checked; final manuscript PDF unavailable |
| Archive PNG assets | Both PNGs were visually inspected; they show method diagrams with no visible author names, affiliations or repository URLs. Both chunk structures were inspected. `model_final (3).png` has no text/EXIF chunks. `data_pipeline.png` contains a 23,654-byte `caBX` C2PA manifest with generation software, timestamps, instance identifiers and provider certificates. No direct author identifier was found in inspected strings. | Review provenance for possible linkage to another public copy. Preserve truthful generation provenance and any required disclosure; do not silently remove or falsify it. | Visible PNGs and metadata checked; provenance linkage unverified |
| Unused archive illustrations | Neither PNG is referenced by `acl_latex.tex`, which includes the four PDF figures. `model_final (3).png` depicts masked-span corruption and risk/specificity acceptance rather than the current complete-response, two-order GPT selection. | Keep the original archive intact. Include only referenced, checked assets in a separate submission-source export; do not present the unused older PNG as the ACL production method. | Legacy illustration identified; original preserved |
| Archive template files | The embedded template README contains upstream ACL style/Overleaf maintenance links; the LuaLaTeX sample contains placeholder author fields. These are not established submission-author identities. | Export only manuscript dependencies when source is requested. Keep upstream attribution/license notices where required. Avoid unrelated template-maintenance material. | Proposed, not applied |
| Expert supplementary material | `acl_latex.tex:574` says anonymized Korean instructions and an English translation accompany the paper. Neither is present in the inspected 16-file archive. | Locate the actual blank production instruments, verify identity removal and availability, and reconcile the manuscript claim before submission. Do not publish completed response packets or researcher mappings. | Availability and anonymization unverified |
| Git owner and metadata | `origin` resolves to an owner-identifying GitHub repository. Reachable commits expose two author handles and three email variants, including an institutional address. Current local commit configuration is identifying. | Never give reviewers the development repository, its PR, `.git` directory, log, refs, patch headers, or history-preserving clone. Use a separate reviewed file export. | Blocking for a raw repository/clone |
| Legacy documentation | `docs/experiment_context_expanded_denoising.md:162` names another owner-associated repository. Legacy experiment notes describe development rather than ACL production. | Preserve legacy material in development, but exclude legacy notes from the anonymous export. | Blocking if mirrored unchanged |
| Production scripts and launchers | H100 production/evaluation helpers contain owner-associated absolute home/SSD paths and imports from sibling worktrees. Examples include `run_test60_rubric_aligned_selection.py`, `run_corrected_test60_pointwise_publication.py`, and `build_rubric_aligned_expert_packet.py`. | In a separately reviewed export, replace location defaults with a repository root derived from `__file__` and explicit resource-root arguments/environment variables; include every imported helper. Validate behavior and contracts before publication. | Blocking for portability and possible identification; scripts untouched |
| `tatus`, local instructions, handoff records | The tracked `tatus` file is an old generated diff summary, not project source. Handoff instructions and agent records may expose workstation layout or upstream development context. | Exclude `tatus`, agent records and handoff bundles from the review export. | Proposed, not applied |
| Secrets and sensitive artifacts | Targeted credential-pattern scans found no matches in the inspected base README/docs/scripts or archive text. This is not a full-history credential audit. Raw counseling text, frozen requests/results, expert mappings, model weights and local `.env` are separate resources. | Use an explicit source allowlist; exclude credentials, datasets, generated response/request files, private expert records, checkpoints, logs and caches. Do not assume ignored files or a renamed archive are safe. | Targeted scan only; data release not performed |

Git history is an independent disclosure channel: sanitizing current files or
making the next commit anonymous does not remove older author metadata. No
history rewrite, force-push, author reconfiguration, or remote change is
authorized or proposed for this development repository. A separate export
without development history avoids shipping that metadata, but does not
erase already public associations or prevent source-text matching.

## Anonymous Mirror

The archive's abstract links to
`https://anonymous.4open.science/r/mh-denoise-E0B0`. A read-only web fetch could
not retrieve it, and an unauthenticated HTTP request returned `401` on the
audit date. This establishes neither public reviewer access nor the mirror's
contents. Its tree, raw-file views, downloadable archive, outbound links,
metadata, and mapping to a source revision were not verified. Do not replace
the manuscript URL with the identifying development repository or PR URL.

Before describing the mirror as available, verify unauthenticated reviewer
access to the exact exported tree and downloads, confirm that no source-owner
link or identifying redirect is exposed, and compare its files to the final
reviewed export. Updating the mirror is a separate publishing action; it was
not performed here.

## Proposed Export Allowlist

This is a review plan, not a ready-made release command. The final paths must
come from the completed A100 integration and the production inventory in
`ACL_REPRODUCIBILITY.md`; importing an entire `scripts/` directory would also
include legacy and unverified experiments.

| Proposed destination | Allowed content after review |
| --- | --- |
| `README.md` | ACL method, limitations, installation/resource requirements and data-use versus redistribution distinction; remove development-repository, internal-audit, verification-record and excluded legacy links. |
| `docs/ACL_REPRODUCIBILITY.md` | Verified production inventory with repository-relative paths; remove identifying workstation paths, development Git commit IDs/branch references and internal-audit links while retaining explicit missing-dependency status and content fingerprints. |
| `docs/ACL_APPENDIX_ADDITION.tex` | Minimal artifact-availability text, after checking it against the actual export. |
| `scripts/` | Only production entrypoints and their complete import closure confirmed in the reproducibility inventory. Candidate path groups include `fullpaper_acl_pipeline.py`, QC/construction contracts and builders, `train_fullpaper_risk_model.py`, SFT/refinement training and inference helpers, corrected TEST60 selection/evaluation helpers and component controls. These candidates are not all present in base `main`; inclusion requires final integration and an identity/portability review. |
| `prompts/` or documented source constants | Exact original production templates and frozen model/settings contracts without instantiated counseling text, provider request identifiers or local paths. Separate prompt constants from private request/result files. |
| `evaluation/` | Only reviewed, blank expert instructions/rubrics and aggregation code if their production use and redistribution are established; no assessor contacts, identities, researcher mappings, or completed packets. |
| Dependency/license files | Only manifests and license notices that actually apply and have been checked; do not invent package pins or grant rights to third-party data/models. |

Explicit exclusions: `.git`, Git logs/patches/refs, the personal-name manuscript
ZIP and all other archives, `tatus`, legacy experiment notes/scripts, this
internal audit, `docs/ACL_VERIFICATION.md`, `AGENTS.md`, agent/handoff records, `.env`, source and derived
datasets, raw responses, instantiated judge queues/results, expert response
packets/mappings, model/adaptor checkpoints, logs, caches and environments.
This does not delete or alter those resources in development.

Original dataset access and redistribution of derived counseling text are
different permissions. No source or derived dataset is being released by
this documentation work. Any later release requires its own source-terms,
consent and identifying-information review; the
[Responsible NLP guidance](https://aclrollingreview.org/responsibleNLPresearch/)
asks authors to document artifact versions, terms, intended use and privacy
measures.

Release remains blocked until the production integration is checked, the
export's imports and resource requirements are complete, identifying paths
are addressed, submission-source cleanup and final-PDF review are complete,
and the actual mirror's anonymous access and contents are verified.
