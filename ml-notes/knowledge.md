# Knowledge Base

_Entries added by writer skill. Format: [W-NNN]_

## [W-001] Why prefix tokens are masked in perplexity computation

When a causal LM predicts token 1, it only sees token 0. When it predicts token 2, it only sees tokens 0–1. With so little context, **every** model — trained on this data or not — will predict poorly. The loss on early tokens is dominated by insufficient context, not by memorization.

So the prefix is noise: it's high loss for everyone, and carries no signal about whether Bob's model was derived from Alice's. By masking it out (`-100`), `get_pplx` focuses on the window where the model has enough context that memorization differences actually show up.

**Source**: `ref/tracing/query/metrics.py`, lines 89–90 (`labels[:, :prefix_len] = -100`)

## [W-002] `get_pplx` vs `pplx` — two different tools in the same file

`get_pplx` is defined in `ref/tracing/query/metrics.py` but never called anywhere in the codebase. It serves a different purpose than `pplx`:

- **`pplx(model_path, texts) → array`**: Matches the `(model_path, texts)` signature required by `BasicStatistic`. Returns a per-sequence array of log-perplexities. Used in the Spearman correlation test pipeline.
- **`get_pplx(sequences, model_id, revision, prefix_len, window_size, batch_size) → scalar`**: Different signature, returns a single mean loss value. Supports `revision` for comparing checkpoints. Cannot be used as a metric for the statistical test (scalar breaks Spearman correlation). Likely a standalone utility for ad-hoc analysis or leftover development code.

**Takeaway**: Not every function in a file serves the same pipeline. Always check the call sites to understand what's actually used.
