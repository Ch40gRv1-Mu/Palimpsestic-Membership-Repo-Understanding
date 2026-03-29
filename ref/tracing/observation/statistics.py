"""
Test statistics for the OBSERVATIONAL setting.

Combines n-gram matching (from index.py) with a metric (from metrics.py)
to produce a single test statistic that can be used in a permutation test.

The shuffle parameter enables the permutation test:
  - shuffle=False: compute the real statistic using the true training order
  - shuffle=True: randomly permute the training step labels → null distribution
  If the real statistic is far from the null → reject H0 → evidence of derivation.
"""

import numpy as np

class TestStatistic:
    """Abstract base class."""
    def __init__(self):
        pass

    def __call__(self,texts,shuffle=False):
        pass

class BasicNGramStatistic:
    """Observational test statistic using n-gram matching.

    Pipeline:
      1. For each text Bob produced, find n-gram matches in Alice's training data
      2. Optionally permute training step labels (for null distribution)
      3. Summarize matches using a metric (e.g., spearman_matches)
    """
    def __init__(self,ngram_index,metric):
        self.index = ngram_index    # NGramIndex that maps n-grams → training steps
        self.metric = metric        # function from observation/metrics.py

    def __call__(self,texts,shuffle=False):
        # Step 1: Match n-grams in each text to training steps
        matched_text_to_steps = []
        for text in texts:
            matched_text_to_steps.append(self.index.match_ngrams_to_steps(text))

        if shuffle:
            # Step 2 (permutation test): randomly relabel all training step indices
            # This breaks any real correlation between training order and generation,
            # creating a sample under the null hypothesis
            perm = np.random.permutation(self.index.num_docs)
            for text in range(len(matched_text_to_steps)):
                for pos in range(len(matched_text_to_steps[text])):
                    matched_text_to_steps[text][pos] = [perm[doc_idx] for doc_idx in matched_text_to_steps[text][pos]]

        # Step 3: Summarize all matches into a single scalar
        return self.metric(matched_text_to_steps)
