"""Small mask utilities shared by the Litho Lab modules (no GUI code)."""
from __future__ import annotations

import numpy as np


def runs(mask01):
    """Contiguous True regions as (start, end-exclusive) index pairs."""
    values = (np.asarray(mask01) > 0.5).astype(int)
    d = np.diff(np.concatenate(([0], values, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def mrc_clean(mask, min_px):
    """Remove narrow interior openings/chrome segments below the mask-rule width."""
    m = (np.asarray(mask) > 0.5).astype(int).copy()
    n = len(m)
    for value in (1, 0):
        for a, b in runs(m == value):
            if (b - a) < int(min_px) and a > 0 and b < n:
                m[a:b] = 1 - value
    return m.astype(float)


def make_idea_mask(target, bias_px=4):
    """Under-sized idea-mask seed; the bias adapts to each opening width.

    The per-opening cap keeps at least a three-pixel clear core, including when the
    target CD is close to the resolution limit.
    """
    seed = (np.asarray(target) > 0.5).astype(float).copy()
    for a, b in runs(seed):
        width = b - a
        local_bias = min(int(bias_px), max(1, (width - 3) // 2))
        if width > 2 * local_bias + 2:
            seed[a:a + local_bias] = 0.0
            seed[b - local_bias:b] = 0.0
    return seed


def spans(binary, x, dx):
    """Open (True) intervals of a binary array as (left, right) in the units of x."""
    return [(x[a] - dx / 2, x[b - 1] + dx / 2) for a, b in runs(binary)]
