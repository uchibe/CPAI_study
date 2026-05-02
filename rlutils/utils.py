"""Small shared utilities."""

import numpy as np


def argmax_random_tie(x, rng=None):
    """Return an argmax of *x*, breaking ties uniformly at random.

    Parameters
    ----------
    x : array_like
        1-D array of values.
    rng : numpy.random.Generator or None
        Random number generator for tie-breaking.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x)
    max_val = np.max(x)
    best = np.flatnonzero(x == max_val)
    return int(rng.choice(best))
