"""Jersey-color tiebreak -- pure classification tests (synthetic crops, no
video). The safety property: classify_team/classify_identity must ABSTAIN
(return None) whenever the color evidence isn't clearly separated -- never
guess a team, same discipline as the OCR confirm threshold.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from color_tiebreak import (  # noqa: E402
    build_team_centroids, classify_identity, classify_team,
    crop_color_signature,
)


def solid(bgr, size=20):
    return np.full((size, size, 3), bgr, dtype=np.uint8)


RED = (0, 0, 255)      # BGR
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (127, 127, 127)


def test_crop_color_signature_is_exact_mean():
    assert crop_color_signature(solid(RED)) == (0.0, 0.0, 255.0)


def test_crop_color_signature_handles_empty_crop():
    assert crop_color_signature(np.zeros((0, 0, 3), dtype=np.uint8)) == (0.0, 0.0, 0.0)
    assert crop_color_signature(None) == (0.0, 0.0, 0.0)


def test_build_team_centroids_averages_and_omits_empty_teams():
    centroids = build_team_centroids({
        "A": [solid(RED), solid(RED)],
        "B": [],
    })
    assert centroids["A"] == (0.0, 0.0, 255.0)
    assert "B" not in centroids, "a team with zero reference crops must never get a fabricated centroid"


def test_classify_team_separates_clearly_distinct_colors():
    centroids = {"A": (0.0, 0.0, 255.0), "B": (0.0, 255.0, 0.0)}   # red vs green
    assert classify_team(solid(RED), centroids) == "A"
    assert classify_team(solid(GREEN), centroids) == "B"


def test_classify_team_separates_white_vs_black():
    centroids = {"Milford": (255.0, 255.0, 255.0), "WW": (0.0, 0.0, 0.0)}
    assert classify_team(solid(WHITE), centroids) == "Milford"
    assert classify_team(solid(BLACK), centroids) == "WW"


def test_classify_team_abstains_on_equidistant_crop():
    centroids = {"A": (255.0, 255.0, 255.0), "B": (0.0, 0.0, 0.0)}
    assert classify_team(solid(GRAY), centroids) is None, \
        "a color exactly between the two centroids must abstain, never guess"


def test_classify_team_needs_at_least_two_centroids():
    assert classify_team(solid(RED), {"A": (0.0, 0.0, 255.0)}) is None


def test_classify_identity_majority_vote():
    centroids = {"A": (0.0, 0.0, 255.0), "B": (0.0, 255.0, 0.0)}
    crops = [solid(RED), solid(RED), solid(GREEN)]
    assert classify_identity(crops, centroids) == "A"


def test_classify_identity_abstains_on_tie():
    centroids = {"A": (0.0, 0.0, 255.0), "B": (0.0, 255.0, 0.0)}
    crops = [solid(RED), solid(GREEN)]
    assert classify_identity(crops, centroids) is None


def test_classify_identity_abstains_when_nothing_resolves():
    centroids = {"A": (255.0, 255.0, 255.0), "B": (0.0, 0.0, 0.0)}
    crops = [solid(GRAY), solid(GRAY)]
    assert classify_identity(crops, centroids) is None
