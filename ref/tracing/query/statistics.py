import numpy as np
import scipy

# ── Abstract base class for test statistics ──────────────────────────────────
class TestStatistic:
    def __init__(self):
        pass

    def __call__(self,model_path,shuffle=False):
        pass

# ── φ_{query}^{ref}: Query-setting test statistic with a reference model ────
# This implements Equation 2 from the paper.
# Core idea: compute Spearman rank correlation between:
#   (a) the training order of sequences (when Alice saw them during training)
#   (b) the difference in loss between Bob's model and a reference model
# If Bob's model is derived from Alice's, sequences seen LATER in training
# will have lower loss (better memorized), producing a significant correlation.
#
# ── document_index data structure ───────────────────────────────────────────
# document_index is a DocumentIndex instance (from tracing/index.py) wrapping
# Alice's training transcript. It exposes:
#
#   document_index.index = {
#       "texts":  List[str],   # the actual training sequences (decoded text)
#       "order":  List[int],   # position each sequence was seen during Alice's
#                              #   training (i.e., training step index)
#   }
#   document_index.num_docs = int  # total number of sequences (len(texts))
#
# Example — constructing a document_index from a HuggingFace transcript:
#
#   from tracing.index import DocumentIndex
#   from datasets import load_dataset
#   from transformers import AutoTokenizer
#
#   transcript = load_dataset("hij/sequence_samples/pythia_deduped_100k",
#                             split="train")
#   tokens = list(transcript["tokens"])[:1000]   # token IDs per sequence
#   order  = list(transcript["index"])[:1000]     # training-order position
#   tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
#   texts = tokenizer.batch_decode(tokens)
#
#   document_index = DocumentIndex(texts, order)
#   # document_index.index["texts"]  → ["The quick brown ...", "In 1492 ...", ...]
#   # document_index.index["order"]  → [4821, 120, 9953, ...]
#
class BasicStatistic:
    def __init__(self,document_index,metric,n=None,reference_path=None):
        self.texts = document_index.index["texts"]    # the training sequences (transcript)
        self.order = document_index.index["order"]     # the order Alice saw them during training
        self.metric = metric                           # function to compute per-sequence loss (e.g., perplexity)
        self.n = n if n is not None else len(self.texts)
        self.reference_path = reference_path           # μ_0: reference model to subtract off baseline effects

    def __call__(self,model_path,shuffle=False):
        # Compute per-sequence loss for Bob's model (μ_β) and the reference model (μ_0)
        model_stats = eval_model(model_path,self.texts,self.metric)
        ref_stats = eval_model(self.reference_path,self.texts,self.metric)

        if shuffle:
            # Permutation test: randomly shuffle the training order
            # Under H0 (Bob didn't use Alice's model), shuffled order should
            # give the same correlation as the true order
            perm = np.random.permutation(self.n)
            order = perm[self.order]
        else:
            order = self.order

        # Spearman correlation between rank-of-training-order and (loss_bob - loss_ref)
        # Negative correlation → sequences seen later have lower loss → evidence of derivation
        return scipy.stats.spearmanr(np.argsort(order), model_stats-ref_stats)

def eval_model(model_path, texts, metric):
    """Compute a per-sequence metric (e.g., perplexity) for a given model on the texts."""
    stats = metric(model_path,texts)

    return stats