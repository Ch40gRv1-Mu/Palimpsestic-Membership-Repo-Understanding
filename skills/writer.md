# Skill: writer
# Triggered when: owner says "run writer skill" or asks to add knowledge/confusion entries

You are executing the **writer** skill. Your job is to:
1. Check for verifier feedback and apply any corrections first
2. Add new knowledge entries to `knowledge.md` with incremental IDs
3. Add confusion entries to `confusions/` with matching IDs
4. Update your writer log

Always execute all steps in order. Do not skip steps.

---

## Step 1 — Check `state/verifier2writer.md` for pending feedback

Read `state/verifier2writer.md`.

Look for any feedback entries marked `status: pending`. These are corrections the
verifier has requested on previous entries.

For each `pending` feedback item:

#### 1a. Find the original entry in `knowledge.md`
Locate the entry by its ID (e.g., `[W-007]`).

#### 1b. Apply the correction
Wrap your correction in annotation tags directly below the original paragraph:

```markdown
<!-- WRITER-CORRECTION:START YYYY-MM-DD feedback-ref:FB-NNN -->
[Corrected content replacing or clarifying the original]
<!-- WRITER-CORRECTION:END -->
```

Do NOT delete or alter the original text — corrections are additive and visible.

#### 1c. Mark the feedback item as applied
In `state/verifier2writer.md`, change that item's `status: pending` to
`status: applied`, and add `applied_date: YYYY-MM-DD`.

#### 1d. Log the correction in `writer_log.md`
```markdown
## Correction W-NNN — YYYY-MM-DD
Applied verifier feedback FB-NNN: [one sentence summary of what was corrected]
```

---

## Step 2 — Read current writer ID

Read `state/writer_id.txt`. This is an integer N.
Your next entry will be ID `N+1`. Increment for each new entry this session.

---

## Step 3 — Add new knowledge entries to `knowledge.md`

For each piece of knowledge to add, append to `knowledge.md`:

```markdown
<!-- AGENT:START YYYY-MM-DD -->
## [W-NNN] Title
**Tags:** tag1, tag2, tag3
**Date:** YYYY-MM-DD
**Linked:** [optional path to related concept note]

[Content. Be precise and factual. Use sub-headers if needed.
Cite sources inline if available: (Source: filename or URL)]
<!-- AGENT:END -->
```

Rules:
- IDs are zero-padded to 3 digits: `W-001`, `W-042`, `W-100`
- One entry = one discrete concept or fact
- Do not bundle unrelated concepts into one entry
- Always include tags

---

## Step 4 — Add confusion entries to `confusions/`

If the session includes a confusion or question from the owner, create:
```
confusions/YYYY-MM/YYYY-MM-DD_topic-slug.md
```

Template:
```markdown
---
status: unread
created: YYYY-MM-DD
writer_id: NNN
tags: []
linked: []
---

# [W-NNN] Topic — YYYY-MM-DD

**Confusion:**
[What was unclear or asked]

**Explanation:**
[Your answer]

**See also:**
[Link to related knowledge.md entry or concept note]
```

Note: the `writer_id` field links the confusion to the corresponding knowledge entry.

---

## Step 5 — Update `state/writer_id.txt`

Write the final N used this session as a plain integer:
```
42
```

---

## Step 6 — Append to `writer_log.md`

```markdown
## Session YYYY-MM-DD

Added entries: W-NNN, W-NNN, ...
Confusions logged: YYYY-MM/YYYY-MM-DD_slug.md (if any)
Corrections applied: FB-NNN (if any)
```

---

## Step 7 — Update `index.md`

Add new entries under `## Knowledge`:
```markdown
- [[W-NNN] Title](knowledge.md#w-nnn) — one sentence summary
```

---

## Rules

- Never modify entries below your own `<!-- AGENT:START -->` blocks
- Never edit `state/verifier_id.txt` — that is the verifier's file
- IDs must be strictly sequential — never reuse or skip an ID
- If `state/writer_id.txt` is missing, start from 0
