---
name: handoff
description: Compact the current conversation into a handoff document for another agent to pick up.
argument-hint: "What will the next session be used for?"
disable-model-invocation: true
---

Write a handoff document summarising the current conversation so a fresh agent can continue the work. Save to the temporary directory of the user's OS (`$TMPDIR`, else `/tmp`; `%TEMP%` on Windows) - not the current workspace.

Report the document's exact absolute path to the user after saving it.

Include the verified progress and current status, supporting evidence files, blockers, and concrete next actions. State which measurable completion criteria are satisfied and which remain.

Include a "suggested skills" section naming which Codex skills the next agent should invoke. If a skill cannot be invoked directly, tell the next agent to read that skill's `SKILL.md` before following it.

Do not duplicate content already captured in other artifacts (specs, plans, ADRs, issues, commits, diffs). Reference them by path or URL instead.

Redact any sensitive information, such as API keys, passwords, or personally identifiable information. Never copy secret values or raw counselling content into the handoff.

If the user passed arguments, treat them as a description of what the next session will focus on and tailor the doc accordingly.
