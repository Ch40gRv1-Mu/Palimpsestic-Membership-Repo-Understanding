"""
Computes p-value from \phi_{query}^{ref} (see Equation 2).

This is the SIMPLEST and most important script in the codebase.
It answers: "Given Alice's training transcript and Bob's model,
is there evidence that Bob's model was trained on Alice's data?"

The pipeline:
  1. Load a "transcript" — an ordered list of training sequences with their
     position in Alice's training order.
  2. Compute per-sequence loss on Bob's model (μ_β) and a reference model (μ_0).
  3. Compute Spearman correlation between training order and (loss_β - loss_0).
  4. Return the p-value. A tiny p-value → strong evidence Bob used Alice's model.

Command-line arguments
  model (\mu_\beta): HuggingFace model ID for Bob's model (the one being audited)
  ref_model (\mu_0): HuggingFace model ID for the reference/baseline model
  n_samples (n): Number of sequences from the transcript to use
  transcript: HF dataset with `index` (training order) and `tokens` columns
  metric_column_name: Use precomputed losses instead of recomputing
  ref_metric_column_name: Use precomputed reference losses

Example usage:
python blackbox-model-tracing/scripts/query/run_query_test.py \
    --model EleutherAI/pythia-6.9b-deduped \
    --ref_model EleutherAI/pythia-6.9b \
    --n_samples 100000 \
    --transcript hij/sequence_samples/pythia_deduped_100k

This tests whether EleutherAI/pythia-6.9b-deduped is trained on
the Pile deduped dataset.
It takes about 3 hrs to compute the logprob of the sequences on an A100.
The program should output a p-value around 1e-50.
"""
import sys
# Add the path to the blackbox-model-tracing dir.
sys.path.append('blackbox-model-tracing')


import argparse
import numpy as np

from datasets import load_dataset, load_from_disk
from tracing.index import DocumentIndex
from tracing.query.metrics import pplx
from tracing.query.statistics import BasicStatistic
from transformers import AutoTokenizer


def load_transcript(transcript_name_or_path):
  """Load Alice's training transcript (ordered training data with position indices).

  A transcript is a dataset where each row is a training sequence, with:
    - 'tokens': the token IDs of the sequence
    - 'index': the position in Alice's training order (when she saw it)
  """
  try:
    # First try to load the transcript as a HF dataset.
    if transcript_name_or_path.count('/') < 2:
      dataset = load_dataset(transcript_name_or_path, split="train")
    else:
      # The dataset might contain subsets (e.g., "hij/sequence_samples/pythia_deduped_100k").
      dataset_name, subset_name = transcript_name_or_path.rsplit("/", 1)
      dataset = load_dataset(dataset_name, subset_name, split="train")
  except:
    # Try to load the transcript as a local dataset.
    dataset = load_from_disk(transcript_name_or_path)
  return dataset


def phi_qr(args, document_index, metric_fn):
  """Compute φ_{query}^{ref}: the query-setting test statistic with a reference model.

  Creates a BasicStatistic (Spearman correlation between training order and
  loss difference), then evaluates it on Bob's model.
  Returns a SignificanceResult with (statistic, pvalue).
  """
  phi = BasicStatistic(document_index, metric_fn, reference_path=args.ref_model)
  return phi(args.model)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str,                        # μ_β: Bob's model to audit
                      default="EleutherAI/pythia-6.9b-deduped")
  parser.add_argument("--ref_model", type=str,                    # μ_0: reference model (controls for general language difficulty)
                      default="EleutherAI/pythia-6.9b")
  parser.add_argument("--n_samples", type=int, default=100000)    # n: how many transcript sequences to use
  parser.add_argument("--transcript", type=str,                   # Alice's ordered training data
                      default="hij/sequence_samples/pythia_deduped_100k")
  parser.add_argument("--metric_column_name", type=str, default=None)      # skip recomputing Bob's losses
  parser.add_argument("--ref_metric_column_name", type=str, default=None)  # skip recomputing ref losses
  args = parser.parse_args()

  # ── Step 1: Load the transcript ──────────────────────────────────────────
  transcript = load_transcript(args.transcript)

  tokens = list(transcript["tokens"])[:args.n_samples]   # token IDs for each sequence
  order = list(transcript["index"])[:args.n_samples]      # training order position for each sequence

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  texts = tokenizer.batch_decode(tokens)                  # decode back to text for the metric function

  # ── Step 2: Build the document index (texts + their training order) ──────
  document_index = DocumentIndex(texts, order)

  # ── Step 3: Compute test statistic and p-value ──────────────────────────
  if not args.metric_column_name:
    # Compute metrics from scratch (slow: runs model inference on all sequences).
    print(phi_qr(args, document_index, pplx))
  else:
    # Use pre-computed metrics (fast: just loads from the dataset columns).
    metrics = list(transcript[args.metric_column_name])[:args.n_samples]
    metrics = np.array([np.mean(x) for x in metrics])     # average per-token loss → per-sequence loss
    ref_metrics = None
    if args.ref_metric_column_name:
      ref_metrics = list(
          transcript[args.ref_metric_column_name])[:args.n_samples]
      ref_metrics = np.array([np.mean(x) for x in ref_metrics])
    # Pass a lambda that returns precomputed metrics based on which model is being queried
    print(phi_qr(args, document_index,
                lambda x, y: metrics if x == args.model else ref_metrics))
