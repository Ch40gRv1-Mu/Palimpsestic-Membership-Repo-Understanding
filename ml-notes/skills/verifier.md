# Skill: verifier
# Triggered when: owner says "run verifier skill"

You are executing the **verifier** skill. Your job is to fact-check all writer
entries that have not yet been verified, and either advance the verifier ID or
report issues back to the writer.

---

## Step 1 — Read both IDs

Read `state/verifier_id.txt` → integer V (last verified entry)
Read `state/writer_id.txt`   → integer W (last written entry)

If V == W, report: "All entries verified up to W-NNN. Nothing to check." and stop.
If V > W, report an error: "verifier_id exceeds writer_id — state is corrupted."

Entries to verify: W-(V+1) through W-(W).

---

## Step 2 — Extract entries to verify

Open `knowledge.md`. Locate all entries with IDs from `[W-(V+1)]` to `[W-W]`.
Collect each entry's full content between its `<!-- AGENT:START -->` and
`<!-- AGENT:END -->` tags.

---

## Step 3 — Fact-check each entry

For each entry, verify:

1. **Factual accuracy** — Are the technical claims correct to the best of your
   knowledge? Flag anything that is wrong, misleading, or unsupported.

2. **Internal consistency** — Does this entry contradict any earlier verified
   entry (W-1 through W-V)?

3. **Tag relevance** — Are the listed tags accurate and specific?

Assign each entry one of:
- `PASS` — all content is correct
- `FLAG` — one or more issues found

---

## Step 4 — If all entries PASS

Update `state/verifier_id.txt` to W (the current writer ID):
```
42
```

Append to `state/verifier2writer.md`:
```markdown
## Verification YYYY-MM-DD
Verified entries: W-NNN through W-NNN
Result: ALL PASS
New verifier ID: NNN
```

Report to owner: "Verified W-(V+1) through W-W. All pass. Verifier ID advanced to W."

---

## Step 5 — If any entry is FLAGGED

Do NOT update `state/verifier_id.txt`.

For each flagged entry, assign a feedback ID `FB-NNN` (increment from the last
FB ID in `verifier2writer.md`, starting at FB-001 if none exist).

Append to `state/verifier2writer.md`:
```markdown
## Verification YYYY-MM-DD
Checked entries: W-NNN through W-NNN

### FB-NNN
status: pending
entry: W-NNN
issue: [Clear description of what is wrong]
correction: [What the correct information is, with reasoning]

### FB-NNN
status: pending
entry: W-NNN
issue: ...
correction: ...
```

Verifier ID stays at V (unchanged).

Report to owner:
- Which entries passed
- Which entries were flagged and why
- That the writer skill must be run to apply corrections before the verifier ID advances

---

## Step 6 — Partial pass handling

If some entries pass and some are flagged:
- Still do NOT advance the verifier ID
- The verifier ID only advances when the entire checked range is clean
- This ensures no unverified content exists between V and W

---

## Rules

- Never edit `knowledge.md` directly — corrections go through the writer
- Never edit `state/writer_id.txt`
- Only write to `state/verifier_id.txt` and `state/verifier2writer.md`
- Be conservative: if you are uncertain about a claim, flag it with your
  uncertainty noted in the `issue` field rather than passing it silently
- Do not re-verify entries at or below the current verifier ID
