"""TOUCH TEAMS -- jersey colour attached to each touch, and when it refuses.

The design under test: measure the two real jersey colours FROM THE FOOTAGE
(clustering), then use the colours typed at setup only to decide which measured
cluster is which team. So the tests care most about the abstention paths -- a
wrong team label swaps every possession in the game.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import numpy as np  # noqa: E402

import clip_registry as cr  # noqa: E402
import touch_teams as tt  # noqa: E402

WHITE_REF = {"name": "Home", "jersey_color": "white", "bgr": (255.0, 255.0, 255.0)}
GREEN_REF = {"name": "Away", "jersey_color": "dark green", "bgr": (0.0, 70.0, 0.0)}
REFS = [WHITE_REF, GREEN_REF]

# TEST1's real teams, exactly as its config records them.
REFS_TEST1 = [
    {"name": "Milford", "jersey_color": "white/red",
     "bgr": cr.parse_jersey_color("white/red")},
    {"name": "Little Miami", "jersey_color": "green/yellow",
     "bgr": cr.parse_jersey_color("green/yellow")},
]


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# ------------------------------------------------------ colour name parsing --

def test_plain_colour_names_parse():
    assert cr.parse_jersey_color("white") == (255.0, 255.0, 255.0)
    assert cr.parse_jersey_color("black") == (0.0, 0.0, 0.0)


def test_dark_modifier_darkens():
    plain = cr.parse_jersey_color("green")
    dark = cr.parse_jersey_color("dark green")
    assert dark[1] < plain[1], "'dark green' must be darker than 'green'"


def test_two_colour_schools_average():
    """The setup form really does receive 'green/yellow' and 'white/red'."""
    assert cr.parse_jersey_color("white/red") is not None
    assert cr.parse_jersey_color("green/yellow") is not None
    assert cr.parse_jersey_color("white/red") != cr.parse_jersey_color("green/yellow")


def test_unrecognisable_colour_abstains_rather_than_defaulting():
    assert cr.parse_jersey_color("sparkly") is None
    assert cr.parse_jersey_color("") is None
    assert cr.parse_jersey_color(None) is None


def test_team_colors_needs_exactly_two_usable_teams():
    assert cr.team_colors({"teams": []}) is None
    assert cr.team_colors({"teams": [{"name": "A", "jersey_color": "white"}]}) is None
    assert cr.team_colors({"teams": [
        {"name": "A", "jersey_color": "white"},
        {"name": "B", "jersey_color": "nonsense"}]}) is None


def test_both_teams_the_same_colour_abstains():
    """Not separable -- either a setup mistake or genuinely unplayable footage."""
    assert cr.team_colors({"teams": [
        {"name": "A", "jersey_color": "white"},
        {"name": "B", "jersey_color": "white"}]}) is None


def test_team_colors_returns_both_when_usable():
    got = cr.team_colors({"teams": [
        {"name": "A", "jersey_color": "white"},
        {"name": "B", "jersey_color": "black"}]})
    assert [t["name"] for t in got] == ["A", "B"]


# ------------------------------------------------------------- torso crop ----

def test_torso_box_sits_inside_the_player_box():
    box = tt.torso_box((100, 200, 200, 500))
    assert box is not None
    x1, y1, x2, y2 = box
    assert 100 < x1 < x2 < 200, "horizontally inside, central column"
    assert 200 < y1 < y2 < 500, "vertically inside, chest height"


def test_degenerate_player_box_has_no_torso():
    assert tt.torso_box((10, 10, 10, 10)) is None
    assert tt.torso_box((10, 10, 5, 5)) is None


def test_sample_torso_reads_the_jersey_colour():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[:, :] = (10, 200, 10)                       # a green frame
    got = tt.sample_torso(frame, (100, 100, 200, 300))
    assert got is not None
    assert got[1] > got[0] and got[1] > got[2], "green channel dominates"


def test_a_box_off_the_edge_of_the_frame_abstains():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert tt.sample_torso(frame, (500, 500, 600, 700)) is None


# -------------------------------------------------------------- averaging ----

def test_a_track_with_too_few_samples_is_dropped():
    got = tt.average_colors({1: [(255, 255, 255)] * 2,      # below the floor
                             2: [(0, 0, 0)] * 5}, min_samples=3)
    assert 1 not in got and 2 in got


def test_none_samples_do_not_pollute_the_average():
    got = tt.average_colors({1: [(200, 200, 200), None, (200, 200, 200),
                                 (200, 200, 200)]}, min_samples=3)
    assert got[1] == (200.0, 200.0, 200.0)


# --------------------------------------------------------------- clustering --

def test_two_clear_kits_split_cleanly():
    colors = {1: (210, 210, 210), 2: (205, 215, 208),    # light kit
              3: (20, 70, 25), 4: (25, 65, 30)}          # dark kit
    got = tt.cluster_two(colors)
    assert got is not None
    c = got["cluster_of"]
    assert c[1] == c[2] and c[3] == c[4] and c[1] != c[3]


def test_one_colour_on_the_floor_abstains():
    """Warm-ups, or both teams genuinely in similar shirts."""
    colors = {1: (200, 200, 200), 2: (202, 198, 201),
              3: (199, 201, 200), 4: (201, 200, 199)}
    assert tt.cluster_two(colors) is None


def test_a_single_track_cannot_be_clustered():
    assert tt.cluster_two({1: (200, 200, 200)}) is None
    assert tt.cluster_two({}) is None


def test_clustering_is_deterministic():
    """No random seed to get lucky with -- same input, same answer, every run."""
    colors = {1: (210, 210, 210), 2: (20, 70, 25), 3: (205, 200, 208),
              4: (25, 65, 30), 5: (215, 205, 210)}
    first = tt.cluster_two(colors)["cluster_of"]
    for _ in range(5):
        assert tt.cluster_two(colors)["cluster_of"] == first


# ---------------------------------------------------------------- labelling --

def test_clusters_are_matched_to_the_typed_colours():
    light, dark = (215.0, 215.0, 215.0), (18.0, 72.0, 22.0)
    got = tt.label_clusters([light, dark], REFS)
    assert got["team_of_cluster"] == ["Home", "Away"]


def test_labelling_survives_the_clusters_arriving_in_the_other_order():
    light, dark = (215.0, 215.0, 215.0), (18.0, 72.0, 22.0)
    got = tt.label_clusters([dark, light], REFS)
    assert got["team_of_cluster"] == ["Away", "Home"]


def test_an_ambiguous_labelling_abstains_rather_than_coin_flipping():
    """Two clusters that sit on top of each other along the team-colour axis --
    picking one would swap every team in the game on a coin flip."""
    a = (100.0, 100.0, 100.0)
    b = (100.0, 100.0, 100.0)
    assert tt.label_clusters([a, b], REFS) is None


def test_labelling_needs_exactly_two_of_each():
    assert tt.label_clusters([(1, 1, 1)], REFS) is None
    assert tt.label_clusters([(1, 1, 1), (2, 2, 2)], [WHITE_REF]) is None


def test_two_nearly_identical_typed_colours_give_no_axis_to_order_by():
    near = [{"name": "A", "jersey_color": "white", "bgr": (255.0, 255.0, 255.0)},
            {"name": "B", "jersey_color": "silver", "bgr": (250.0, 250.0, 250.0)}]
    assert tt.label_clusters([(200.0, 200.0, 200.0), (60.0, 60.0, 60.0)],
                             near) is None


def test_muddy_real_world_centroids_still_label_correctly():
    """REGRESSION, TEST1 2026-08-02. The first version of label_clusters scored
    both pairings by absolute distance to the typed colour and could not answer
    on real footage: a torso crop carries skin, shadow, floor and gym light, so
    every jersey averages toward the same middling grey. These are the ACTUAL
    measured centroids from TEST1 -- a white kit and a green kit, and note how
    little either looks like 'white' or 'green'.

    Ground truth is eyeballed, not assumed: the crops were rendered and looked
    at, one white jersey (track 49) and three green (67/395/875).

    The fix is that only the DIRECTION between the two clusters is used, so the
    shared grey offset cancels. If someone ever swaps that back for a nearest-
    colour match, this test fails."""
    white_ish = (120.7, 96.9, 109.2)      # the white kit, as actually measured
    green_ish = (82.0, 93.1, 101.3)       # the green kit, as actually measured
    assert _dist(white_ish, WHITE_REF["bgr"]) > 200, \
        "the measured white is nowhere near typed white -- that is the point"

    got = tt.label_clusters([white_ish, green_ish], REFS_TEST1)
    assert got is not None, "must not abstain on real footage this clean"
    assert got["team_of_cluster"] == ["Milford", "Little Miami"]


def test_the_same_pair_labels_consistently_in_either_order():
    white_ish = (120.7, 96.9, 109.2)
    green_ish = (82.0, 93.1, 101.3)
    a = tt.label_clusters([white_ish, green_ish], REFS_TEST1)["team_of_cluster"]
    b = tt.label_clusters([green_ish, white_ish], REFS_TEST1)["team_of_cluster"]
    assert a == list(reversed(b)), "swapping the inputs must swap the answer"


# ----------------------------------------------------------------- assemble --

def test_team_of_tracks_end_to_end():
    colors = {1: (210, 210, 210), 2: (205, 215, 208),
              3: (20, 70, 25), 4: (25, 65, 30)}
    got, reason, detail = tt.team_of_tracks(colors, REFS)
    assert reason is None
    assert detail["axis_sep"] > 0
    assert got[1] == "Home" and got[2] == "Home"
    assert got[3] == "Away" and got[4] == "Away"


def test_team_of_tracks_explains_why_it_abstained():
    colors = {1: (200, 200, 200), 2: (201, 199, 200)}
    got, reason, detail = tt.team_of_tracks(colors, REFS)
    assert got == {} and detail is None
    assert reason and "two jersey colours" in reason


def test_attach_teams_leaves_unknown_tracks_teamless():
    touches = [{"track_id": 1}, {"track_id": 99}]
    tt.attach_teams(touches, {1: "Home"})
    assert touches[0]["team"] == "Home"
    assert touches[1]["team"] is None, "an unknown track must not be guessed"
