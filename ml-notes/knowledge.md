# Knowledge Base

_Entries added by writer skill. Format: [W-NNN]_

## [W-001] Why prefix tokens are masked in perplexity computation

When a causal LM predicts token 1, it only sees token 0. When it predicts token 2, it only sees tokens 0–1. With so little context, **every** model — trained on this data or not — will predict poorly. The loss on early tokens is dominated by insufficient context, not by memorization.

So the prefix is noise: it's high loss for everyone, and carries no signal about whether Bob's model was derived from Alice's. By masking it out (`-100`), `get_pplx` focuses on the window where the model has enough context that memorization differences actually show up.

**Source**: `ref/tracing/query/metrics.py`, lines 89–90 (`labels[:, :prefix_len] = -100`)
