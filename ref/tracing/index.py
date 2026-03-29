"""
Index structures that map n-grams (or documents) to training step positions.

This module is central to the OBSERVATIONAL setting:
  Given text that Bob's model produced, find which n-grams in that text also
  appear in Alice's training data, and WHERE in the training order they appeared.

If Bob's text contains n-grams that disproportionately come from LATER in
Alice's training, that's evidence Bob's model is derived from Alice's
(because later-trained data is more memorized → more likely to be generated).

Two index types:
  - SimpleNGramIndex: brute-force in-memory index (for small experiments)
  - InfiniGramIndex: uses the InfiniGram engine for efficient large-scale lookup
"""

from collections import defaultdict

import json

from infini_gram.engine import InfiniGramEngine
import timeit

from .utils import timeout

# ── Simple wrapper: stores texts and their training order ────────────────────
# Used in the QUERY setting where we don't need n-gram lookup,
# just a mapping from text → training position.
class DocumentIndex:
    def __init__(self, texts, order):
        self.num_docs = len(texts)
        self.index = {
            "texts": texts,    # the actual text sequences
            "order": order,    # their position in Alice's training order
        }

    def get_training_steps(self, input_ids):
        pass

# ── Base class for n-gram based indices (observational setting) ──────────────
class NGramIndex:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.num_docs = None

    def get_training_steps(self, input_ids):
        """Given token IDs of an n-gram, return list of training steps where it appeared."""
        pass

    def match_ngrams_to_steps(self, text, n_max=None, print_stats=False):
        """For each position in `text`, find the longest n-gram (up to n_max)
        that appears in the training data, and return the training steps where
        it was found.

        This implements the core n-gram matching from Algorithm 2:
          For each token position, try the longest possible n-gram first,
          then back off to shorter n-grams until we find a match.

        Returns: list of lists — for each matched position, the list of
                 training step indices where that n-gram appeared.
        """
        tokens = self.tokenizer.encode(text)
        if n_max is None:
            n_max = len(tokens)

        matched_steps = []
        for pos in range(len(tokens)-n_max):
            n = n_max
            # Try longest n-gram first, back off until we find a match
            while n > 0:
                ngram = tokens[pos:pos+n+1]
                start_time = timeit.default_timer()
                steps = self.get_training_steps(ngram)   # look up in index
                end_time = timeit.default_timer()
                if print_stats:
                    print("INDEX STATS:")
                    print(f"Looked for the following {n}-gram: {repr(self.tokenizer.decode(ngram))}")
                    print(f"Found {len(steps)} matches in {end_time - start_time} seconds")

                if len(steps) > 0:
                    matched_steps.append(steps)  # found a match → record the training steps
                    break
                n -= 1                           # back off to shorter n-gram

        return matched_steps

    def match_ngrams_to_steps_list(self, texts, n_max=None, print_stats=False):
        """Run match_ngrams_to_steps on a list of texts."""
        full_matches = []

        for i in range(len(texts)):
            start_time = timeit.default_timer()
            matched_steps = match_ngrams_to_steps(texts[i], n_max, print_stats)
            end_time = timeit.default_timer()

            if print_stats:
                print(f"Time taken for {i}-th text: {end_time - start_time} seconds")
            full_matches.append(matched_steps)

        return full_matches

# ── In-memory brute-force n-gram index (for small-scale experiments) ─────────
class SimpleNGramIndex(NGramIndex):
    def get_training_steps(self, input_ids):
        """Look up the n-gram in the in-memory dict, return document indices."""
        return  [info['idx'] for info in self.index[input_ids]]

    def train_index(self, texts, n_max, save_path=None):
        """Build the index: for every n-gram of length 1..n_max in every
        training document, store which document it came from and its position.
        """
        self.index = defaultdict(list)
        self.num_docs = len(texts)
        for n in range(1, n_max + 1):
            for idx, text in enumerate(texts):
                tokens = self.tokenizer.encode(text)
                for pos in range(len(tokens) - n):
                    kgram = tuple(tokens[pos:pos+n])
                    kgram_dict = {
                        "idx": idx,          # which training document
                        "pos": pos,          # position within that document
                        "next_token": tokens[pos+n],  # the token following the n-gram
                    }
                    self.index[kgram].append(kgram_dict)


# ── InfiniGram-backed index (for large-scale experiments) ────────────────────
# InfiniGram is a suffix-array-based engine that can efficiently look up
# arbitrary n-grams across terabyte-scale corpora.
# See: https://infini-gram.readthedocs.io/en/latest/
class InfiniGramIndex(NGramIndex):
    def load_index(self, index_path, **index_kwargs):
        """Load a pre-built InfiniGram index from disk."""
        self.index = InfiniGramEngine(index_path, **index_kwargs)
        self.num_docs = self.index.get_total_doc_cnt()

    @timeout(0.01)  # hard timeout: skip if lookup takes >10ms (prevents hanging on common n-grams)
    def get_training_steps(self,input_ids):
        """Look up an n-gram in the InfiniGram index and return the training
        step of every document that contains it.

        Each document in the index has metadata with a 'step' field indicating
        when it was seen during training.
        """
        results = self.index.find(input_ids=input_ids)
        segments = results['segment_by_shard']      # results split by index shard
        all_steps = []
        # Iterate over all matching documents across all shards
        for shard, rank_range in enumerate(segments):
            for rank in range(*rank_range):
                docs = self.index.get_doc_by_rank(s=shard, rank=rank, max_disp_len=10)
                metadata = json.loads(docs['metadata'])
                all_steps.append(metadata['step'])  # extract the training step
        return all_steps
