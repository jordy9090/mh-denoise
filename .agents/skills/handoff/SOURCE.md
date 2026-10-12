# Source and local adaptations

- Upstream repository: https://github.com/mattpocock/skills
- Upstream skill: `skills/productivity/handoff`
- Verified revision: `f3fc5632f401156837ee3872f14fe33ccf1024ea`
- Retrieved: 2026-10-08
- Upstream `SKILL.md` blob: `a224edc643a20ddb989f9bca1547440169159810`
- Upstream `agents/openai.yaml` blob: `6e1d8da121beceddc2f9ec7fa8714c1d914dd11a`
- Repository `LICENSE` blob: `f1dd2c09108dde1a5f56097cee8461b3ea834499`

The MIT `LICENSE` is copied unchanged from the upstream repository root at the verified revision. The upstream `agents/openai.yaml` already disabled implicit invocation and is retained unchanged.

Local changes to `SKILL.md`:

- Replaced the Claude-specific instruction to call the Skill tool with Codex skill invocation, falling back to reading the relevant `SKILL.md`.
- Required verified status, evidence paths, blockers, measurable completion state, and next actions in each handoff.
- Required the exact OS-temporary output path to be reported.
- Explicitly prohibited copying secret values or raw counselling content.
