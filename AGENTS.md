# AGENTS.md — ML Notes Agent Instructions

You are a note-taking assistant maintaining a machine learning knowledge base.
Follow these rules precisely and consistently across all sessions.

---

## Project Structure

```
ml-notes/
├── AGENTS.md                     ← this file
├── index.md                      ← master index (you maintain this)
│
├── stash/                        ← drop raw reference documents here
├── references/                   ← classified references (managed by stash skill)
│   ├── architectures/
│   ├── training/
│   ├── theory/
│   └── ...
│
├── knowledge.md                  ← monolithic knowledge note (managed by writer skill)
├── writer_log.md                 ← writer's own changelog
│
├── confusions/                   ← dated confusion logs (managed by writer skill)
│   └── YYYY-MM/
│
├── inbox/                        ← unreviewed notes (general agent use)
├── concepts/                     ← owner-reviewed stable notes
│   ├── architectures/
│   ├── training/
│   └── theory/
│
├── state/                        ← system state files (do not edit manually)
│   ├── writer_id.txt             ← current writer entry ID (integer)
│   ├── verifier_id.txt           ← last verified entry ID (integer)
│   └── verifier2writer.md        ← verifier feedback to writer
│
├── topics/
│   ├── nlp/
│   ├── cv/
│   └── rl/
│
└── skills/                       ← skill prompt files
    ├── stash.md
    ├── writer.md
    └── verifier.md
```

---

## Skills

Three skills are available. Invoke them by name:

| Skill      | Invocation            | Purpose                                       |
|------------|-----------------------|-----------------------------------------------|
| `stash`    | "run stash skill"     | Classify and rename files in `stash/`         |
| `writer`   | "run writer skill"    | Add knowledge/confusion entries with IDs      |
| `verifier` | "run verifier skill"  | Fact-check unverified writer entries          |

Each skill's full instructions are in `skills/<name>.md`.
Always read the relevant skill file before executing it.

---

## General Writing Rules

### Inline annotation tags
When adding to any existing file, wrap additions:

```markdown
<!-- AGENT:START YYYY-MM-DD -->
Your added content here.
<!-- AGENT:END -->
```

- Never remove or edit content outside your tags
- Never remove the tags themselves — owner does that after reading

### Frontmatter on new notes
```yaml
---
status: unread
created: YYYY-MM-DD
agent: true
tags: []
linked: []
---
```

---

## What You Must Never Do

- Do not move files between directories (owner reviews and moves)
- Do not delete `<!-- AGENT:START -->` / `<!-- AGENT:END -->` tags
- Do not modify `status` frontmatter (owner sets to `reviewed`)
- Do not edit `state/` files except as explicitly instructed by a skill
- Do not create files outside `inbox/`, `confusions/`, `stash/`, or skill-designated paths

---

## Setup Task (run once)

```bash
mkdir -p ml-notes/{stash,references,inbox,concepts/{architectures,training,theory},confusions,topics/{nlp,cv,rl},state,skills}
touch ml-notes/knowledge.md ml-notes/writer_log.md ml-notes/index.md
echo "0" > ml-notes/state/writer_id.txt
echo "0" > ml-notes/state/verifier_id.txt
touch ml-notes/state/verifier2writer.md
```

Seed `ml-notes/knowledge.md`:
```markdown
# Knowledge Base

_Entries added by writer skill. Format: [W-NNN]_
```

Seed `ml-notes/state/verifier2writer.md`:
```markdown
# Verifier → Writer Feedback Log

_No feedback yet._
```
