# Skill: stash
# Triggered when: owner says "run stash skill" or drops files into stash/

You are executing the **stash** skill. Your job is to process all unclassified
files in `stash/`, rename them with descriptive keyword-based names, and move
them into the correct subdirectory under `references/`.

---

## Step-by-Step Execution

### Step 1 — Scan `stash/`
List all files in `stash/`. If the directory is empty, report "stash/ is empty."
and stop.

### Step 2 — For each file in `stash/`:

#### 2a. Read and understand the file
Open the file and identify:
- **Topic**: What is the primary subject? (e.g., attention mechanism, AdamW optimizer)
- **Type**: What kind of document is it? Choose one:
  - `paper` — research paper or arxiv preprint
  - `tutorial` — walkthrough or how-to
  - `reference` — API docs, cheat sheet, specification
  - `notes` — informal notes or blog post
  - `other`
- **Keywords**: Extract 3–5 short lowercase keywords that best describe the content.
  Prefer specific technical terms over generic ones.
  Good: `rotary-positional-encoding`, `gpt-neox`, `dropout`
  Bad: `model`, `training`, `deep-learning`
- **Category**: Map the topic to one of these reference subdirectories:
  - `architectures` — model designs, layers, components
  - `training` — optimizers, regularization, schedules, loss functions
  - `theory` — math, proofs, statistical foundations
  - `datasets` — data sources, preprocessing, benchmarks
  - `inference` — serving, quantization, decoding strategies
  - `misc` — anything that does not fit above

#### 2b. Build the new filename
Format:
```
YYYY-MM-DD_keyword1-keyword2-keyword3_type.ext
```

Use today's date. Use the original file extension.

Examples:
```
2026-03-26_rotary-positional-encoding-rope-transformer_paper.pdf
2026-03-26_adamw-optimizer-weight-decay_reference.md
2026-03-26_gpt-neox-architecture-parallel-attention_notes.txt
```

#### 2c. Move the file
Move from:
```
stash/<original_filename>
```
To:
```
references/<category>/<new_filename>
```

Create the target subdirectory if it does not exist.

#### 2d. Append an entry to `index.md` under `## References`
```markdown
- [keyword1-keyword2-keyword3](references/<category>/<new_filename>) — <one sentence description> (`<type>`, <date>)
```

---

### Step 3 — Report
After processing all files, print a summary table:

| Original Name | New Name | Category | Type |
|---------------|----------|----------|------|
| ...           | ...      | ...      | ...  |

---

## Rules

- Never delete or overwrite files in `references/`
- If two files would produce the same name, append `-2`, `-3`, etc.
- If a file is already in `references/` (was already processed), skip it
- Do not modify file contents — only rename and move
- If you cannot determine the topic from the content, use `misc` and name it
  `YYYY-MM-DD_unknown-<original-stem>_other.ext`, then note it in the report
