"""Jersey-keyed box score -- aggregation tests (written before the stage).

The box score is the coach-facing output, so its honesty rules are the tests:
confirmed (live) and confirmed_retroactive counted separately AND jointly;
numbers aggregate ACROSS windows; identities without a number are SURFACED as
an unnamed bucket (never dropped, never guessed); a number on both rosters is
flagged team-ambiguous; zone time uses only positioned confirmed/retro events.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from stage8_box_score import build_box_score  # noqa: E402

FPS = 30.0


def ev(window, ident, frame, state):
    return {"window": window, "frame": frame, "identity_id": ident,
            "track_id": ident + 100, "event": "on_floor_presence",
            "identity_state": state}


def merged_doc(rows):
    return {"clip": "T", "player_events": rows}


def reg(window, ident, number):
    return {"window": window, "identity_id": ident, "track_id": ident + 100,
            "roster_number": number, "final_state": "confirmed"}


def oncourt_doc(frames_tracks):
    """frames_tracks: {frame: {track_id: (cx, cy)}} -> minimal oncourt doc."""
    return {"frames": [
        {"frame_index": f,
         "tracks": {str(t): {"on": True, "court_feet": [x, y]}
                    for t, (x, y) in tr.items()}}
        for f, tr in sorted(frames_tracks.items())]}


def test_live_and_retro_counted_separately_and_jointly():
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]           # 30f live
    rows += [ev(0, 1, f, "confirmed_retroactive") for f in range(140, 155)]  # 15f
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 5)], {5: ["A"]},
                          FPS, oncourt_doc({}))
    line = doc["players"][0]
    assert line["number"] == 5 and line["team"] == "A"
    assert line["seconds_live"] == 1.0
    assert line["seconds_retro"] == 0.5
    assert line["seconds_total"] == 1.5


def test_numbers_aggregate_across_windows():
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]
    rows += [ev(1, 7, f, "confirmed") for f in range(160, 190)]          # same #5
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 5), reg(1, 7, 5)],
                          {5: ["A"]}, FPS, oncourt_doc({}))
    assert len(doc["players"]) == 1
    assert doc["players"][0]["seconds_total"] == 2.0
    assert doc["players"][0]["windows_present"] == 2


def test_unnamed_confirmed_surfaces_never_drops():
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]           # named #5
    rows += [ev(0, 2, f, "confirmed") for f in range(100, 130)]          # no number
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 5), reg(0, 2, None)],
                          {5: ["A"]}, FPS, oncourt_doc({}))
    assert doc["unnamed_confirmed"]["identities"] == 1
    assert doc["unnamed_confirmed"]["seconds_total"] == 1.0


def test_dual_team_number_flagged_ambiguous():
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 3)],
                          {3: ["A", "B"]}, FPS, oncourt_doc({}))
    assert "AMBIGUOUS" in doc["players"][0]["team"]


def test_candidates_and_unknowns_go_to_review_not_lines():
    rows = [ev(0, 1, f, "candidate") for f in range(100, 130)]
    rows += [ev(0, 2, f, "unknown") for f in range(100, 130)]
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 5)], {5: ["A"]},
                          FPS, oncourt_doc({}))
    assert doc["players"] == []
    assert doc["review"]["candidate_events"] == 30
    assert doc["review"]["unknown_events"] == 30


def test_zone_time_from_positioned_confirmed_events_only():
    # #5 confirmed for 30 frames at the left free-throw area (paint):
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]
    # plus 30 candidate frames at midcourt that must NOT count:
    rows += [ev(0, 2, f, "candidate") for f in range(100, 130)]
    onc = oncourt_doc({f: {101: (10.0, 25.0), 102: (42.0, 25.0)}
                       for f in range(100, 130)})
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 5), reg(0, 2, None)],
                          {5: ["A"]}, FPS, onc)
    line = doc["players"][0]
    assert line["zone_seconds"].get("PAINT") == 1.0
    assert sum(line["zone_seconds"].values()) == 1.0
    assert line["unpositioned_frames"] == 0


def test_simultaneous_same_number_frames_are_disputed_not_counted():
    """Two identities claiming #24 on the SAME frames: one body is wrong.
    Disputed frames are excluded from the line and surfaced, never counted."""
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]           # id1
    rows += [ev(0, 2, f, "confirmed") for f in range(120, 140)]          # id2 overlaps 120..129
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 24), reg(0, 2, 24)],
                          {24: ["A"]}, FPS, oncourt_doc({}))
    line = doc["players"][0]
    assert line["seconds_total"] == 30 / FPS, \
        "only solely-claimed frames count (20 from id1 + 10 from id2)"
    assert line["disputed_seconds"] == 10 / FPS
    assert line["seconds_total"] + line["disputed_seconds"] <= 40 / FPS


def test_merge_stamped_number_beats_missing_registry_number():
    """A human-resolved identity may have NO registry hypothesis -- its number
    lives on the merged events. The box score must honor the merge stamp."""
    rows = []
    for f in range(100, 130):
        e = ev(0, 7, f, "confirmed_retroactive")
        e["merge"] = {"number": 14, "source": "human"}
        rows.append(e)
    doc = build_box_score(merged_doc(rows), [reg(0, 7, None)], {14: ["A"]},
                          FPS, oncourt_doc({}))
    assert doc["players"][0]["number"] == 14
    assert doc["players"][0]["seconds_retro"] == 1.0
    assert doc["unnamed_confirmed"]["identities"] == 0


def test_dual_team_number_splits_when_color_tiebreak_resolves():
    """Two identities under the SAME dual-roster number, resolved to
    different teams by the color tiebreak -- must become two clean lines,
    not one blended AMBIGUOUS line."""
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]        # id1: 30f
    rows += [ev(0, 2, f, "confirmed") for f in range(200, 220)]       # id2: 20f, no overlap
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 3), reg(0, 2, 3)],
                          {3: ["A", "B"]}, FPS, oncourt_doc({}),
                          identity_team={(0, 1): "A", (0, 2): "B"})
    teams = {r["team"]: r["seconds_total"] for r in doc["players"]}
    assert teams == {"A": 1.0, "B": 20 / FPS}
    assert all("AMBIGUOUS" not in t for t in teams)


def test_dual_team_number_partial_resolution_leaves_residual_ambiguous():
    """One identity resolves, the other doesn't (color abstained) -- the
    unresolved one must stay in the AMBIGUOUS bucket, never guessed."""
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]
    rows += [ev(0, 2, f, "confirmed") for f in range(200, 220)]
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 3), reg(0, 2, 3)],
                          {3: ["A", "B"]}, FPS, oncourt_doc({}),
                          identity_team={(0, 1): "A"})              # id2 unresolved
    by_team = {r["team"]: r["seconds_total"] for r in doc["players"]}
    assert by_team["A"] == 1.0
    assert any("AMBIGUOUS" in t and secs == 20 / FPS
              for t, secs in by_team.items())


def test_disputed_frames_on_dual_team_number_never_attributed_to_a_team():
    """Disputed (conflicting-claim) frames under a dual-roster number must
    surface on the AMBIGUOUS row, even when every SOLE frame resolves
    cleanly -- never silently folded into a team's resolved credit."""
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]        # id1
    rows += [ev(0, 2, f, "confirmed") for f in range(120, 140)]       # id2 overlaps 120..129
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 3), reg(0, 2, 3)],
                          {3: ["A", "B"]}, FPS, oncourt_doc({}),
                          identity_team={(0, 1): "A", (0, 2): "A"})
    rows_by_team = {r["team"]: r for r in doc["players"]}
    assert "AMBIGUOUS (A / B)" in rows_by_team
    amb = rows_by_team["AMBIGUOUS (A / B)"]
    assert amb["disputed_seconds"] == 10 / FPS
    assert amb["seconds_total"] == 0.0
    assert rows_by_team["A"]["disputed_seconds"] == 0.0


def test_no_override_behaves_exactly_as_before():
    """identity_team omitted entirely -- byte-identical to pre-tiebreak
    behavior (default argument, existing callers unaffected)."""
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 130)]
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 3)],
                          {3: ["A", "B"]}, FPS, oncourt_doc({}))
    assert doc["players"][0]["team"] == "AMBIGUOUS (A / B)"


def test_unpositioned_frames_counted_honestly():
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 110)]
    onc = oncourt_doc({f: {101: (10.0, 25.0)} for f in range(100, 105)})  # half
    doc = build_box_score(merged_doc(rows), [reg(0, 1, 5)], {5: ["A"]},
                          FPS, onc)
    line = doc["players"][0]
    assert line["unpositioned_frames"] == 5
    assert abs(sum(line["zone_seconds"].values()) - 5 / FPS) < 1e-9
