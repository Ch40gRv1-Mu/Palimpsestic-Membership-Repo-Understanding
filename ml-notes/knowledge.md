# Knowledge Base

_Entries added by writer skill. Format: [W-NNN]_

---

## Questions & Confusions from Code Reading

### [Q-001] What is `document_index` and how is it structured?
**File:** `ref/tracing/query/statistics.py`
**Answer:** `document_index` is a `DocumentIndex` instance (from `tracing/index.py`). It wraps Alice's training transcript:
- `document_index.index["texts"]`: `List[str]` — the actual training sequences (decoded text)
- `document_index.index["order"]`: `List[int]` — position each sequence was seen during Alice's training
- `document_index.num_docs`: `int` — total number of sequences

Constructed from a HuggingFace transcript dataset by decoding token IDs and pairing with training order indices.

### [Q-002] What is `tqdm` and why is it imported?
**File:** `ref/tracing/query/metrics.py`, `ref/tracing/llm.py`
**Answer:** Full name is **"taqaddum"** (تقدّم, Arabic for "progress"). It provides a terminal progress bar for long-running loops. **Classification: HOUSEWORK** — removing it changes nothing about the computation, only loses the progress display. Used in `get_pplx()`, `evaluate_model()`, `train_model()`, and `distill_model()`.

### [Q-003] What are the options for `module_type` in `evaluate.load()`?
**File:** `ref/tracing/query/metrics.py` — `evaluate.load("perplexity", module_type="metric")`
**Answer:** Three valid values in the HuggingFace `evaluate` library:
- `"metric"` (default) — standard evaluation metrics (accuracy, perplexity, BLEU, etc.)
- `"comparison"` — comparing results between models (e.g., McNemar's test)
- `"measurement"` — measuring dataset properties (e.g., word length, toxicity)

The explicit `module_type="metric"` here is technically redundant since it's the default.

### [Q-004] How does `shift_logits` work in `compute_per_token_pplx`?
**File:** `ref/tracing/query/metrics.py`
**Answer:** A causal LM at position `i` predicts token at position `i+1`. Logits and labels are off by one, so shifting aligns them:
- `shift_logits = logits[:, :-1, :]` — drop last position (no ground truth for it)
- `labels = labels[:, 1:]` — drop first position (no prediction for it)

Example with `["The", "cat", "sat", "on"]`:
```
shift_logits[0] (pred for pos 1) vs label "cat" ✓
shift_logits[1] (pred for pos 2) vs label "sat" ✓
shift_logits[2] (pred for pos 3) vs label "on"  ✓
```

### [Q-005] Where is prefix masking done?
**File:** `ref/tracing/query/metrics.py`
**Answer:** NOT inside `compute_per_token_pplx` — it's done by the **caller** `get_pplx()`:
```python
labels = encoded_inputs['input_ids'].clone()
labels[:, :prefix_len] = -100    # ← masking happens here
```
`compute_per_token_pplx` receives labels with `-100` already baked in. PyTorch's `CrossEntropyLoss` automatically skips `-100` positions (outputs 0.0).

### [Q-006] What does `labels.view(-1)` do?
**File:** `ref/tracing/query/metrics.py`
**Answer:** `view(-1)` flattens a tensor into 1D — `-1` means "infer size from total elements". Needed because `CrossEntropyLoss` expects flat inputs: predictions `[N, vocab_size]` and labels `[N]`.
- batch=1: `[1, 3]` → `[3]`
- batch=2: `[2, 3]` → `[6]` (collapses batch and seq dims)

### [Q-007] Why is `.contiguous()` called after slicing?
**File:** `ref/tracing/query/metrics.py`
**Answer:** `[:, :-1, :]` and `[:, 1:]` return **views** (no copy, point to same memory but skip elements). The underlying bytes are not sequential. `.view()` requires contiguous memory, so `.contiguous()` forces a copy into a fresh sequential block. Without it: `RuntimeError: view size is not compatible with input tensor's size and stride`.

**Rule of thumb:** `.contiguous()` ensures memory layout works for the subsequent `.view(-1, ...)` reshape. Whenever you slice a tensor and then need to `.view()` it, call `.contiguous()` in between.

### [Q-008] Why do we need padding at all?
**File:** `ref/tracing/query/metrics.py`
**Answer:** GPUs are massively parallel — processing one sequence at a time wastes ~99% of the hardware. Batching sends multiple sequences through the model simultaneously in a single matrix multiplication (~10-100x faster). But **tensors are rectangular**: a `[batch_size, seq_len]` matrix must have the same number of columns in every row. You can't have row 1 with 3 tokens and row 2 with 5 — that's a jagged array, which GPUs don't support. Padding fills shorter sequences to match the longest, making the tensor rectangular. The attention mask then tells the model to ignore padded positions.

```
Without padding:  [3 tokens]     → can't stack into one tensor
                  [2 tokens]        with [5 tokens]
                  [5 tokens]

With padding:     [3 tokens + 2 PAD] → [batch=3, seq_len=5] ✓
                  [2 tokens + 3 PAD]    rectangular tensor
                  [5 tokens + 0 PAD]    GPU-friendly
```

### [Q-009] What is `pad_token` and why left-pad for causal LMs?
**File:** `ref/tracing/query/metrics.py`
**Answer:** Two issues on top of needing padding:
1. **Many causal LMs (GPT, Pythia, LLaMA) have no pad token by default** — they were trained on single sequences. Must set one manually (e.g., `'<|padding|>'`).
2. **Left-pad, not right-pad** — causal LMs generate left-to-right, appending tokens to the right. The key is where the last real token ends up:

```
RIGHT-pad:  ["Hello", "world", <PAD>, <PAD>]
  → Last real token ("world") at pos 1, followed by <PAD>s.
    For generation: model tries to "continue" from <PAD> — nonsensical.

LEFT-pad:   [<PAD>, <PAD>, "Hello", "world"]
  → Last real token ("world") at the rightmost position.
    Generation naturally appends the next token to the right.
    Attention mask zeros out <PAD> positions on the left.
```

**Clarification:** Neither side preserves the original positional encodings (unpadded "Hello" was at pos 0; with left-pad it's at pos 2). The attention mask handles this. Left-pad is preferred because it also works correctly for generation, which right-pad does not.

### [Q-010] Pythia checkpoints and the `revision` parameter
**File:** `ref/tracing/query/metrics.py` — `AutoModelForCausalLM.from_pretrained(..., revision=revision)`
**Answer:** The `revision` parameter selects which checkpoint to load from a HuggingFace model repo (which is a git repo — each checkpoint is a branch/tag).

Pythia is a suite of LLMs designed for controlled scientific experiments on model behavior. It provides **154 checkpoints per model**:
- `step0` — initial (untrained) weights
- 10 log-spaced: `step{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}`
- 143 evenly-spaced: `step1000`, `step2000`, ..., `step143000`

These are hosted on HuggingFace as **branches**. Branch `step143000` corresponds exactly to the `main` branch.

Example usage:
```python
# Load the fully trained model (same as main)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision="step143000")

# Load an early checkpoint
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision="step1000")

# Load the untrained model
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision="step0")
```

This is critical for the palimpsestic memorization experiments — comparing model behavior across training checkpoints reveals how memorization evolves over training.
