"""Jersey-color tiebreak -- resolves numbers that appear on BOTH rosters.

HARD has two such numbers (#3, #23): OCR alone can never team-attribute a
read of a number worn by both teams (phase2/DECISIONS.md section 8/9b). This
module classifies a player's TEAM from jersey color instead, using per-clip
TEAM COLOR CENTROIDS built automatically from crops the system ALREADY
trusts -- every CONFIRMED frame whose number is UNAMBIGUOUS (e.g. HARD's
#24, #10, #44) is free labeled data for what that team's jersey looks like
on THIS footage, same camera/lighting/court. No hardcoded RGB guesses, no
new human input.

Abstention-first, like everywhere else in this codebase: classify_team()
only returns an answer when the two centroids are clearly separated for a
given crop; otherwise it returns None and the caller must NOT guess.
classify_identity() aggregates several of one identity's frames (majority
vote) so one noisy crop can't flip a whole player's team.

Pure functions -- no I/O, no video reads, no roster/config coupling.
Callers (stage8_box_score.py) do the crop extraction and wiring.
"""

from __future__ import annotations

import numpy as np

# A crop's nearest centroid must be at least this many times closer than the
# next-nearest to resolve; otherwise abstain. Tune later by watching real
# misclassifications, same discipline as OCR_CONFIRM_THRESHOLD -- never
# loosen it just to shrink the AMBIGUOUS bucket.
COLOR_MARGIN_RATIO = 1.4


def crop_color_signature(crop_bgr) -> tuple[float, float, float]:
    """Mean (B, G, R) over a jersey crop -- the color a human eye would call
    'the jersey color', averaged over fabric plus minor skin/background noise."""
    if crop_bgr is None or crop_bgr.size == 0:
        return (0.0, 0.0, 0.0)
    mean = crop_bgr.reshape(-1, 3).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def build_team_centroids(crops_by_team: dict) -> dict:
    """{team_name: [crop_bgr, ...]} -> {team_name: (b, g, r) centroid}.
    A team with zero reference crops is OMITTED -- never a fabricated
    centroid standing in for missing evidence."""
    centroids = {}
    for team, crops in crops_by_team.items():
        sigs = [crop_color_signature(c) for c in crops
                if c is not None and c.size > 0]
        if sigs:
            centroids[team] = tuple(np.array(sigs).mean(axis=0))
    return centroids


def classify_team(crop_bgr, centroids: dict, margin_ratio: float = COLOR_MARGIN_RATIO):
    """One crop -> team name if the nearest centroid is clearly closer than
    the next-nearest (>= margin_ratio times its distance), else None
    (abstain -- the color evidence is not good enough to call). Needs at
    least 2 centroids to make any call."""
    if len(centroids) < 2:
        return None
    sig = crop_color_signature(crop_bgr)
    ranked = sorted(((team, _dist(sig, c)) for team, c in centroids.items()),
                    key=lambda r: r[1])
    (best_team, best_d), (_, second_d) = ranked[0], ranked[1]
    if second_d < margin_ratio * max(best_d, 1e-6):
        return None
    return best_team


def classify_identity(crops, centroids: dict, margin_ratio: float = COLOR_MARGIN_RATIO):
    """Several crops of ONE identity -> the majority team among crops that
    resolved, or None if nothing resolved or the vote ties (abstain -- a
    real player has one color; a tie means the evidence isn't good enough)."""
    votes: dict = {}
    for crop in crops:
        team = classify_team(crop, centroids, margin_ratio)
        if team is not None:
            votes[team] = votes.get(team, 0) + 1
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda r: -r[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]
