"""
Metrics for the QUERY setting: compute per-sequence loss/perplexity.

In the query setting, Alice can run Bob's model on her training sequences.
She computes per-sequence perplexity (or loss), which captures how well
the model "remembers" each sequence. Sequences seen LATER in training
will have lower loss (better memorized) → the signal tested by the statistic.
"""

import numpy as np
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def eval_model(model_path, texts, metric):
    """Evaluate a model on texts using the given metric function."""
    stats = metric(model_path,texts)
    return stats

def pplx(model_path, texts):
    """Compute log-perplexity for each text using HuggingFace's evaluate library.

    This is the simplest metric: load model, compute perplexity per sequence.
    Returns an array of log-perplexities (one per text).
    Lower log-perplexity → the model assigns higher probability to the text
    → likely better memorized.
    """
    perplexity = evaluate.load("perplexity", module_type="metric")
    result = perplexity.compute(model_id=model_path,
                                add_start_token=True,
                                predictions=texts)
    pplx = np.log(result['perplexities'])

    return pplx

def compute_per_token_pplx(model, encoded_inputs, labels):
    """Compute per-token cross-entropy loss (unreduced).

    Returns a [batch_size, seq_len] tensor of per-token losses.
    Positions with label=-100 are ignored (used for masking the prefix).
    """
    with torch.no_grad():
        outputs = model(encoded_inputs['input_ids'], labels=labels)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = outputs.logits[:, :-1, :].contiguous()   # predict next token
        labels = labels[:, 1:].contiguous()                      # shift labels
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                    labels.view(-1))
        loss = loss.view(labels.size(0), -1)                     # reshape back to [batch, seq]
        return loss

# NOTE: get_pplx is defined but never called anywhere in the codebase.
# It has a different signature (sequences, model_id, revision, ...) than
# the (model_path, texts) → array contract required by BasicStatistic.
# It returns a scalar (mean loss), not a per-sequence array, so it cannot
# be used as a metric for the Spearman correlation test.
# Likely a standalone utility for ad-hoc checkpoint analysis (e.g., notebooks).
def get_pplx(sequences,
             model_id,
             revision='main',
             prefix_len=32,
             window_size=32,
             batch_size=32):
    """Compute windowed perplexity: average per-token loss over a fixed window
    after a prefix. This controls for prefix effects and focuses on the
    "completion" part of each sequence.

    Args:
        prefix_len: number of prefix tokens to mask (label=-100, not scored)
        window_size: number of tokens after the prefix to average loss over
        revision: model revision/checkpoint (for models with multiple checkpoints)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = '<|padding|>'
    tokenizer.padding_side = 'left'            # left-pad for causal LMs
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map='auto',
        torch_dtype=torch_dtype,
        revision=revision
    )
    model = model.eval()

    seq_to_pplx = {}
    for i in tqdm(range(0, len(sequences), batch_size)):
        encoded_inputs = tokenizer(sequences[i:i + batch_size],
                                    return_tensors='pt',
                                    max_length=96,
                                    truncation=True,
                                    padding='max_length').to(model.device)
        labels = encoded_inputs['input_ids'].clone()
        labels[:, :prefix_len] = -100           # mask prefix tokens (not scored)
        pplx = compute_per_token_pplx(model, encoded_inputs, labels)
        for b_i in range(len(pplx)):
            seq_to_pplx[sequences[i + b_i]] = pplx[b_i]

    # Average loss over the scoring window [prefix_len, prefix_len + window_size]
    pplx = []
    for seq in seq_to_pplx.keys():
        pplx.append(seq_to_pplx[seq][prefix_len : prefix_len + window_size].mean().tolist())

    return pplx.mean()
