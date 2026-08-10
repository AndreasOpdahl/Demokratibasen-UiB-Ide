# CLAUDE.md

This file is the entry point for Claude Code when working in this repository.

## About this repository

This is a support/ops repo for the AI behind [Demokratibasen](https://demokratibasen.no), a Norwegian civic-document platform — not the Demokratibasen product itself (that lives elsewhere). It's a loose collection of largely independent Python pipelines, each in its own top-level folder, covering things like:

- Human evaluation tooling and snapshot reporting (`human_evaluations/`)
- Checkpoint selection and LLM-as-judge evaluation (`checkpoint_selection/`)
- Document classification, entity/theme extraction, and case-document summarization pipelines (`extract_document_classes/`, `extract_document_types/`, `structured_data_extraction/`, `case_documents_summary/`)
- Dataset preparation and cleaning from Demokratibasen exports (`datasets_from_demokratibasen/`, `process_data/`)

Model fine-tuning code used to live here too, but was split out into the sibling repo `../Demokratibasen-Finetune` (August 2026), preserving its git history via `git filter-repo`. This repo now focuses on everything *around* the models: data prep, evaluation, and analysis.

There's no shared entry point or build system — treat each top-level folder as its own project with its own README, `.env`, and often its own `requirements.txt`. Folders/files prefixed `OLD`, `STALE`, or `SCRATCH` are deprecated/scratch and gitignored (`**/OLD*`, `**/STALE*`, `**/SCRATCH*` in `.gitignore`) — don't treat them as current.

This is a solo-maintained repo (single author), not a team codebase.

## Core rules

- Keep changes scoped and reviewable.
- Report files changed, decisions, commands run, results, risks, and any deployment steps.

## Permissions and sandbox live in the committed settings.json

Because this is a solo-maintained repo (no team of developers with differing local setups to accommodate), permission rules *and* sandbox configuration are kept together in the committed [.claude/settings.json](.claude/settings.json) rather than split off into a gitignored `.claude/settings.local.json` — that keeps everything durable across sessions and machines instead of living in a file that never gets committed. `.claude/settings.local.json` is still used, but only as scratch space for one-off, short-lived grants (e.g. a single ad hoc command) that aren't worth persisting — periodically fold anything worth keeping back into `settings.json` and clear the local file out.

The sandbox is the actual safety boundary, not the allow/deny string list — string matching on commands is easy to get subtly wrong (wildcard prefixes like `Bash(find *)` are leakier than they look):

- `sandbox.enabled: true` runs commands isolated by default.
- `sandbox.allowUnsandboxedCommands: false` removes the escape hatch — nothing can opt out of the sandbox, even if asked.
- `sandbox.failIfUnavailable: true` makes Claude Code refuse to start if sandboxing can't be established, instead of silently falling back to unrestricted execution.
- `sandbox.network.allowedDomains: []` blocks essentially all outbound network access at the network layer by default; exceptions get added explicitly and narrowly (e.g. specific hosts, not general-purpose proxies) as they're actually needed.

Destructive or privileged operations (`git push`, `rm -rf`, database writes, etc.) still warrant the same caution as always — the sandbox restricts *reach* (what hosts/paths a command can touch), not *intent*, so risky commands aimed at something already inside the allowed scope still deserve deliberate confirmation.

## Repository language

- Code and comments are in English.
- Documentation (READMEs, notes) is mixed English/Norwegian depending on the folder — match whichever language the surrounding README already uses rather than defaulting to one.
- No UI lives in this repo.
