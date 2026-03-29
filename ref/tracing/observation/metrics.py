"""
Metrics for the OBSERVATIONAL setting.

After matching n-grams in Bob's generated text to training steps in Alice's
data (via index.py), these functions summarize the matches into a single
scalar that can be tested for significance.

The key question: do the matched training steps skew toward LATER in training?
If yes → Bob's model is likely derived from Alice's.

Input format: matched_texts_to_steps is a nested list:
  [text_0_matches, text_1_matches, ...]
  where each text_i_matches = [match_at_pos_0, match_at_pos_1, ...]
  where each match_at_pos_j = [step_a, step_b, ...]  (training steps where the n-gram appeared)
"""

import numpy as np
import scipy
from ..utils import flatten_list

def avg(matched_text_to_steps):
    """Simple average of all matched training steps.
    If the average is high (late in training) → evidence of derivation."""
    return np.mean(flatten_list(matched_text_to_steps))

def spearman_matches(n_steps, matched_texts_to_steps):
    """Spearman correlation between training step and n-gram match count.

    This is φ_{obs}^{part} from the paper (Algorithm 2):
      1. Count how many n-gram matches fall in each training step
      2. Compute Spearman correlation between step index and count
    A positive correlation means later steps have more matches → evidence.
    """
    counts = np.zeros(n_steps)                              # one bin per training step
    matched_text_to_steps = flatten_list(matched_texts_to_steps)
    for step in matched_text_to_steps:
        counts[step] += 1                                   # tally matches per step

    # Spearman rank correlation: step_index vs match_count
    return scipy.stats.spearmanr(np.arange(n_steps), counts)

def stratified_avg(matched_text_to_steps):
    """Stratified average: for each (text, position) pair, randomly sample
    one matched step, then average. Avoids bias from positions with many matches."""
    sampled_values = []
    for text in matched_text_to_steps:
        for pos in text:
            if pos:
                sampled_values.append(np.random.choice(matched_text_to_steps[text][pos]))

    return np.mean(sampled_values)

def single_match(matched_text_to_steps):
    """Average only over positions where the n-gram matched exactly one
    training step (unique matches). These are the most informative matches
    since there's no ambiguity about which step produced the n-gram."""
    single_values = []
    for text in matched_text_to_steps:
        for pos in text:
            if pos:
                if len(matched_text_to_steps[text][pos]) == 1:
                    single_values.append(matched_text_to_steps[text][pos][0])

    return np.mean(single_values)

