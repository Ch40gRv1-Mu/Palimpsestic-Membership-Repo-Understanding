"""
Metrics for the QUERY setting: compute per-sequence loss/perplexity.

In the query setting, Alice can run Bob's model on her training sequences.
She computes per-sequence perplexity (or loss), which captures how well
the model "remembers" each sequence. Sequences seen LATER in training
will have lower loss (better memorized) → the signal tested by the statistic.
"""

import numpy as np                                                  # [HOUSEWORK] numerical arrays
import evaluate                                                     # [HOUSEWORK] HF evaluate library
import torch                                                        # [HOUSEWORK] tensor operations
from transformers import AutoTokenizer, AutoModelForCausalLM        # [HOUSEWORK] model/tokenizer loading
# tqdm — full name "taqaddum" (تقدّم, Arabic for "progress") — provides a
# terminal progress bar for long-running loops. Pure housework: removing it
# changes nothing about the computation, only loses the progress display.
from tqdm import tqdm                                               # [HOUSEWORK] progress bar

def eval_model(model_path, texts, metric):                          # [HOUSEWORK] thin dispatch wrapper
    """Evaluate a model on texts using the given metric function."""
    stats = metric(model_path,texts)                                # [HOUSEWORK] delegates to metric
    return stats                                                    # [HOUSEWORK] pass-through return

def pplx(model_path, texts):
    """Compute log-perplexity for each text using HuggingFace's evaluate library.

    This is the simplest metric: load model, compute perplexity per sequence.
    Returns an array of log-perplexities (one per text).
    Lower log-perplexity → the model assigns higher probability to the text
    → likely better memorized.
    """
    perplexity = evaluate.load("perplexity", module_type="metric")  # [HOUSEWORK] load HF metric module
    result = perplexity.compute(model_id=model_path,                # [IMPORTANT] run model inference, compute per-sequence perplexity
                                add_start_token=True,
                                predictions=texts)
    pplx = np.log(result['perplexities'])                           # [IMPORTANT] log-transform perplexities → the signal used by the test statistic

    return pplx                                                     # [HOUSEWORK] return result

def compute_per_token_pplx(model, encoded_inputs, labels):
    """Compute per-token cross-entropy loss (unreduced).

    Returns a [batch_size, seq_len] tensor of per-token losses.
    Positions with label=-100 are ignored (used for masking the prefix).

    ── Why shift? ──────────────────────────────────────────────────────────
    A causal LM predicts the NEXT token from each position. So the logit at
    position i is the model's prediction for what token comes at position i+1.
    To compute loss, we need to align predictions with targets:

    Example: input tokens = ["The", "cat", "sat", "on"]  (positions 0,1,2,3)

      outputs.logits (what the model predicts NEXT from each position):
        pos 0 → prediction for pos 1  (model tries to predict "cat")
        pos 1 → prediction for pos 2  (model tries to predict "sat")
        pos 2 → prediction for pos 3  (model tries to predict "on")
        pos 3 → prediction for pos 4  (nothing to compare against)

      shift_logits = logits[:, :-1, :]   → drop pos 3 (no ground truth for it)
        = [pred_for_1, pred_for_2, pred_for_3]

      shifted labels = labels[:, 1:]     → drop pos 0 (no prediction for it)
        = ["cat", "sat", "on"]

      Now they align:
        shift_logits[0] vs labels[0]:  model predicted → "cat" actual ✓
        shift_logits[1] vs labels[1]:  model predicted → "sat" actual ✓
        shift_logits[2] vs labels[2]:  model predicted → "on"  actual ✓

    With prefix masking (labels[:, :prefix_len] = -100):
      labels = [-100, -100, "sat", "on"]   (prefix_len=2, mask "The","cat")
      shifted labels = [-100, "sat", "on"]
      → CrossEntropyLoss ignores -100 positions, so only "sat" and "on" are scored.
      This is the "completion window" — only measuring how well the model
      predicts tokens AFTER the prefix.
    """
    # ── Concrete trace (batch=1, prefix_len=2) ──────────────────────────
    # Caller (get_pplx) passes in:
    #   encoded_inputs['input_ids'] = [[0, 1, 2, 3]]       # "The"=0, "cat"=1, "sat"=2, "on"=3
    #   labels                      = [[-100, -100, 2, 3]]  # caller set [:, :2] = -100
    #
    # NOTE: prefix masking is NOT done here — it's done by the caller (get_pplx):
    #   labels = encoded_inputs['input_ids'].clone()
    #   labels[:, :prefix_len] = -100
    with torch.no_grad():                                           # [HOUSEWORK] disable gradient tracking (inference only)

        outputs = model(encoded_inputs['input_ids'], labels=labels)  # [IMPORTANT] forward pass through the model
        # outputs.logits shape: [1, 4, vocab_size]
        #   logits[0][0] = [0.1, 0.8, ...]   pos 0 → predicts what comes at pos 1
        #   logits[0][1] = [0.2, 0.1, ...]   pos 1 → predicts what comes at pos 2
        #   logits[0][2] = [0.1, 0.1, ...]   pos 2 → predicts what comes at pos 3
        #   logits[0][3] = [0.3, 0.2, ...]   pos 3 → predicts pos 4 (no ground truth)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')       # [IMPORTANT] unreduced CE loss — returns one loss per position (no averaging)

        shift_logits = outputs.logits[:, :-1, :].contiguous()       # [IMPORTANT] drop last position (no ground truth for it)
        # shift_logits shape: [1, 3, vocab_size]
        #   = [pred_for_pos1, pred_for_pos2, pred_for_pos3]

        labels = labels[:, 1:].contiguous()                         # [IMPORTANT] drop first position (no prediction for it)
        # labels shape: [1, 3]
        #   [[-100, -100, 2, 3]] → [[-100, 2, 3]]
        #
        # Now aligned:
        #   shift_logits[0] (pred for pos 1) vs label=-100 → SKIPPED (masked prefix)
        #   shift_logits[1] (pred for pos 2) vs label=2    → CE loss for "sat"
        #   shift_logits[2] (pred for pos 3) vs label=3    → CE loss for "on"

        # view(-1) flattens a tensor into 1D — the -1 means "infer size from total elements".
        # view(-1, K) reshapes into 2D with K columns.
        # CrossEntropyLoss expects flat inputs: predictions [N, vocab_size] and labels [N].
        #
        # batch=1:  shift_logits [1, 3, V] → .view(-1, V) → [3, V]
        #           labels       [1, 3]    → .view(-1)    → [3]
        #
        # batch=2:  shift_logits [2, 3, V] → .view(-1, V) → [6, V]
        #           labels  [[-100, 2, 3],
        #                    [-100, 5, 1]] → .view(-1) → [-100, 2, 3, -100, 5, 1]  (flat [6])
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),  # [IMPORTANT] flatten to [N, vocab_size], compute per-token CE
                    labels.view(-1))                                  # flatten to [N]
        # loss = [0.0, 1.23, 0.45]
        #          ↑     ↑     ↑
        #       ignored  CE    CE       (-100 → PyTorch outputs 0.0 and skips)
        #       prefix  "sat" "on"

        loss = loss.view(labels.size(0), -1)                        # [HOUSEWORK] reshape back to [batch, seq] → [[0.0, 1.23, 0.45]]
        return loss                                                 # [HOUSEWORK] return [1, 3] tensor
        # Back in get_pplx(), the caller slices loss[prefix_len : prefix_len+window_size]
        # → [1.23, 0.45], then .mean() → 0.84 (only the completion window is scored)

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
    tokenizer = AutoTokenizer.from_pretrained(model_id)             # [HOUSEWORK] load tokenizer

    # ── Why do we need padding at all? ──────────────────────────────────
    # GPUs are massively parallel — processing one sequence at a time wastes
    # ~99% of the hardware. Batching sends multiple sequences through the
    # model simultaneously in a single matrix multiplication (~10-100x faster).
    #
    # But tensors are RECTANGULAR: a [batch_size, seq_len] matrix must have
    # the same number of columns in every row. You can't have row 1 with
    # 3 tokens and row 2 with 5 tokens — that's a jagged array, which GPUs
    # don't support.
    #
    #   Without padding:  [3 tokens]     → can't stack into one tensor
    #                     [2 tokens]        with [5 tokens]
    #                     [5 tokens]
    #
    #   With padding:     [3 tokens + 2 PAD] → [batch=3, seq_len=5] ✓
    #                     [2 tokens + 3 PAD]    rectangular tensor
    #                     [5 tokens + 0 PAD]    GPU-friendly
    #
    # The attention mask then tells the model to ignore padded positions
    # so they don't affect the computation.
    #
    # ── Why pad_token and padding_side? ───────────────────────────────
    # Problem 1: many causal LMs (GPT, Pythia, LLaMA) have NO pad token
    # defined by default — they were trained on single sequences. We must
    # set one manually so the tokenizer knows what to fill with.
    #
    # Problem 2: which SIDE to pad matters for causal LMs.
    #
    # Causal LMs generate left-to-right, appending tokens to the RIGHT.
    # The key question: where is the last real token in the padded sequence?
    #
    #   RIGHT-pad:
    #     seq1: ["The", "cat", "sat", "on", "the", "mat"]
    #     seq2: ["Hello", "world", <PAD>, <PAD>, <PAD>, <PAD>]
    #     → Last real token ("world") is at pos 1, followed by <PAD>s.
    #       For GENERATION: model would try to "continue" from a <PAD>,
    #       which is nonsensical — it has never seen <PAD> during training.
    #       For INFERENCE: attention mask can handle this in modern HF,
    #       but historically caused subtle bugs in some implementations.
    #
    #   LEFT-pad (correct for causal LMs):
    #     seq1: ["The", "cat", "sat", "on", "the", "mat"]
    #     seq2: [<PAD>, <PAD>, <PAD>, <PAD>, "Hello", "world"]
    #     → Last real token ("world") is at the RIGHTMOST position.
    #       Generation naturally appends the next token to the right.
    #       Attention mask zeros out <PAD> positions on the left so the
    #       model never attends to them.
    #
    # Note: neither side preserves the original positional encodings
    # (unpadded "Hello" was at pos 0; here it's at pos 4). The attention
    # mask handles this. Left-pad is preferred because it also works
    # correctly for generation, which right-pad does not.
    #
    tokenizer.pad_token = '<|padding|>'                             # [HOUSEWORK] set pad token (many causal LMs lack one by default)
    tokenizer.padding_side = 'left'                                 # [HOUSEWORK] left-pad so real tokens are right-aligned (correct for causal LMs)
    torch_dtype = torch.bfloat16                                    # [HOUSEWORK] precision setting

    # ── Loading a model from HuggingFace Hub ────────────────────────────
    # revision: THIS IS HOW YOU LOAD DIFFERENT CHECKPOINTS of the same model.
    #   HuggingFace models are git repos — each checkpoint is a branch/tag.
    #   Example inputs for revision:
    #     revision="main"          → latest version (default)
    #     revision="step100000"    → checkpoint at training step 100k
    #     revision="step143000"    → checkpoint at training step 143k
    #     revision="v1.0"          → a tagged release
    #   This is critical for Pythia models which publish every 1000-step
    #   checkpoint, e.g. EleutherAI/pythia-6.9b has "step0", "step1",
    #   "step1000", "step2000", ..., "step143000".
    model = AutoModelForCausalLM.from_pretrained(                   # [HOUSEWORK] load model from HF hub
        model_id,                                                   # e.g. "EleutherAI/pythia-6.9b-deduped"
        low_cpu_mem_usage=True,                                     # load weights shard-by-shard (saves ~50% CPU RAM)
        device_map='auto',                                          # auto-distribute layers across GPUs/CPU/disk
        torch_dtype=torch_dtype,                                    # bfloat16: 2 bytes/param (7B model ≈ 14GB vs 28GB at float32)
        revision=revision                                           # which checkpoint to load (see above)
    )
    model = model.eval()                                            # [HOUSEWORK] switch to eval mode (disables dropout etc.)

    seq_to_pplx = {}                                                # [HOUSEWORK] accumulator dict
    for i in tqdm(range(0, len(sequences), batch_size)):            # [HOUSEWORK] batch loop with progress bar
        # ── Tokenize a batch and trace dimensions ─────────────────────
        # Example: batch_size=3, max_length=96, sequences = ["The cat sat", "Hello world", "I like dogs a lot"]
        #
        # encoded_inputs is a dict with two tensors:
        #   'input_ids':      [batch_size, max_length] = [3, 96]
        #     Each row is one sequence, tokenized and padded/truncated to exactly 96 tokens.
        #     [[<PAD>, <PAD>, ..., 464, 3797, 3027],    ← "The cat sat" (left-padded)
        #      [<PAD>, <PAD>, ..., 12092, 995],          ← "Hello world" (left-padded)
        #      [<PAD>, ..., 40, 588, 6922, 64, 1256]]    ← "I like dogs a lot" (left-padded)
        #
        #   'attention_mask': [batch_size, max_length] = [3, 96]
        #     1 = real token, 0 = padding (tells model to ignore these positions)
        #     [[0, 0, ..., 1, 1, 1],
        #      [0, 0, ..., 1, 1],
        #      [0, ..., 1, 1, 1, 1, 1]]
        #
        # return_tensors='pt': return PyTorch tensors (not lists or numpy)
        # max_length=96:       truncate sequences longer than 96 tokens
        # truncation=True:     enable truncation (required with max_length)
        # padding='max_length': pad ALL sequences to exactly max_length (not just to longest in batch)
        # .to(model.device):   move tensors to same device as model (e.g., cuda:0)
        encoded_inputs = tokenizer(sequences[i:i + batch_size],     # [HOUSEWORK] tokenize batch
                                    return_tensors='pt',
                                    max_length=96,
                                    truncation=True,
                                    padding='max_length').to(model.device)

        labels = encoded_inputs['input_ids'].clone()                # [HOUSEWORK] copy input_ids → labels, shape [3, 96]
        labels[:, :prefix_len] = -100                               # [IMPORTANT] mask prefix tokens — labels[:, :32] = -100, only score positions 32..96
        # labels shape still [3, 96], but first 32 cols are -100 (ignored by CrossEntropyLoss)

        pplx = compute_per_token_pplx(model, encoded_inputs, labels)  # [IMPORTANT] returns [3, 95] — per-token loss (95 = 96-1 after shift)
        for b_i in range(len(pplx)):                                # [HOUSEWORK] store results per sequence
            seq_to_pplx[sequences[i + b_i]] = pplx[b_i]            # pplx[b_i] shape: [95] — one loss value per token position

    # Average loss over the scoring window [prefix_len, prefix_len + window_size]
    pplx = []                                                       # [HOUSEWORK] accumulator
    for seq in seq_to_pplx.keys():                                  # [HOUSEWORK] iterate sequences
        pplx.append(seq_to_pplx[seq][prefix_len : prefix_len + window_size].mean().tolist())  # [IMPORTANT] windowed mean loss — the per-sequence metric

    return pplx.mean()                                              # [IMPORTANT] aggregate to single scalar
