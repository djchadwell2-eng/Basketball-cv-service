"""hoop_anchor.project_point -- pure homogeneous-projection math, no video.
Correctness here matters: it's the same operation used to carry the
user-confirmed rim pixel through every frame of the pan (Phase 5 step 3A).
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from hoop_anchor import project_point  # noqa: E402


def test_identity_is_a_no_op():
    assert project_point(np.eye(3), 10.0, 20.0) == (10.0, 20.0)


def test_pure_translation():
    M = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], dtype=float)
    assert project_point(M, 1.0, 1.0) == (6.0, -2.0)


def test_pure_scale():
    M = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    assert project_point(M, 3.0, 4.0) == (6.0, 8.0)


def test_perspective_divide_is_applied():
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)  # w scales by 2
    x, y = project_point(M, 10.0, 10.0)
    assert (round(x, 6), round(y, 6)) == (5.0, 5.0)


def test_degenerate_w_returns_none_instead_of_dividing_by_zero():
    M = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -10]], dtype=float)
    assert project_point(M, 10.0, 0.0) is None


def test_chained_homographies_compose_like_matrix_multiply():
    """The real use: T = Hs_opt[pos] @ Hfk, applied via inv(T). Verify
    projecting through a composed matrix equals projecting step by step."""
    A = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]], dtype=float)   # translate
    B = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)   # scale
    combined = project_point(A @ B, 1.0, 1.0)
    stepwise = project_point(A, *project_point(B, 1.0, 1.0))
    assert combined == stepwise
