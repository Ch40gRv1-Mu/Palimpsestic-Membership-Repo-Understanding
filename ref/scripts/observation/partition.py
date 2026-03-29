"""
φ_{obs}^{part}: Observational setting test using n-gram partitioning (Section 4.3.1).

This is the OBSERVATIONAL counterpart to run_query_test.py.
Instead of querying Bob's model directly, Alice only has TEXT that Bob's model
produced. She checks whether n-grams in that text correlate with her training order.

Pipeline:
  1. Load Bob's generated texts (from a pickle file)
  2. Load Alice's training data index (pre-built InfiniGram suffix array)
  3. For each text, find the longest matching n-gram at each position
  4. Count how many n-gram matches fall at each training step
  5. Compute Spearman correlation between step index and match count
  6. Report the p-value

Uses InfiniGram for efficient n-gram lookup across large corpora.
See: https://infini-gram.readthedocs.io/en/latest/

Example:
  python partition.py --texts_paths gens.pkl --infinigram_index_dir /path/to/index --n_texts 100000
"""

import argparse
import random
import pickle
from transformers import AutoTokenizer

from ..index import InfiniGramIndex
from ..tracing.observation.metrics import spearman_matches

def phi_op(index, texts, k):
  """Compute φ_{obs}^{part}: observational test statistic via n-gram partitioning.

  For each text, find n-gram matches (up to length k) in the training index,
  then compute Spearman correlation between training step and match count.
  """
  matched_steps = index.match_ngrams_to_steps_list(texts, n_max=k)  # match n-grams → training steps
  n_train_steps = index.get_training_steps                           # total number of training steps
  return spearman_matches(n_train_steps, matched_texts_to_steps)     # Spearman(step, count)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--infinigram_index_dir', type=str, required=True)   # path to pre-built InfiniGram index
  parser.add_argument('--tokenizer_name',
                      type=str,
                      default='EleutherAI/pythia-6.9b-deduped',
                      help='The name of the tokenizer used to build the index.')
  parser.add_argument('--k', type=int, default=8)                          # max n-gram length to match
  parser.add_argument('--texts_path', type=str, required=True)             # pickle file with Bob's generated texts
  parser.add_argument('--n_texts', type=int, default=5000)                 # how many texts to use
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()

  random.seed(args.seed)

  # Load the InfiniGram index over Alice's training data
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
  index = InfiniGramIndex(tokenizer)
  index.load_index(index_dir=args.infinigram_index_dir,
                   eos_token_id=tokenizer.eos_token_id)

  # Load Bob's generated texts
  texts = pickle.load(open(args.texts_path, 'rb'))

  # Shuffle and run the test
  random.shuffle(texts)
  print(phi_op(index, texts[:args.n_texts], args.k))
  
