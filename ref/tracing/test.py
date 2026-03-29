"""
Permutation tests for computing p-values.

These implement the statistical hypothesis testing framework:
  H0: Bob's model is NOT derived from Alice's model
       (i.e., there's no correlation between training order and model behavior)
  H1: Bob's model IS derived, so training order correlates with behavior

The permutation test works by:
  1. Compute the real test statistic (e.g., Spearman correlation)
  2. Repeatedly shuffle the training order labels and recompute
  3. If the real statistic is extreme compared to the shuffled ones → reject H0
"""

import numpy as np
import scipy

def exact_permutation_test(T, statistic, **statistic_kwargs):
    """Exact permutation test: compute the rank of the real statistic among T shuffled ones.

    Returns p-value = (rank + 1) / (T + 1), where rank = number of shuffled
    statistics that are >= the real statistic.
    """
    original_statistic = statistic(**statistic_kwargs)           # real statistic
    rank = 0
    for _ in range(T):
        exchangeable_statistic = statistic(**statistic_kwargs,shuffle=True)  # null statistic
        if exchangeable_statistic >= original_statistic:
            rank += 1

    return (rank+1)/(T+1)     # p-value: fraction of null stats as extreme as real

def approximate_permutation_test(T, statistic, **statistic_kwargs):
    """Approximate permutation test: assume null distribution is normal.

    Instead of counting the rank, fit a Gaussian to the T shuffled statistics
    and compute the z-score of the real statistic. More powerful when T is small.
    """
    original_statistic = statistic(**statistic_kwargs)           # real statistic

    exchangeable_statistics = []
    for _ in range(T):
        exchangeable_statistics.append(statistic(**statistic_kwargs,shuffle=True))  # null samples

    # z-score: how many standard deviations is the real stat from the null mean?
    z_score = (original_statistic - np.mean(exchangeable_statistics)) / np.std(exchangeable_statistics)
    # Two-sided p-value from the normal distribution
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))

    return p_value