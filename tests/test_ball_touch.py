"""BALL TOUCHES -- pure geometry + run-grouping tests (synthetic boxes, no
video, no caches).

A TOUCH is one player holding the ball until she gives it up -- NOT a
possession, which is the team-level concept phase2/possessions.py already owns.

These tests pin the two things that make the measurement honest rather than
merely plausible:
  1. The gate is in BODY HEIGHTS, not pixels, so the SAME pixel gap is a hold
     on a near-court body and a miss on a far-court one.
  2. Every ambiguous case ABSTAINS -- a near-tie credits nobody, a blink-length
     run is dropped, and an identity that is not uniformly CONFIRMED cannot
     become a stat.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from ball_touch import (  # noqa: E402
    HOLD_GATE_BODY_FRAC, MARGIN_BODY_FRAC, MAX_GAP_SECONDS, MIN_TOUCH_FRAMES,
    attribute, ball_position, build_touches, holder_at_frame, jersey_numbers,
    shooter_from_touches,
)


def track(tid, x1, y1, x2, y2):
    return {"track_id": tid, "bbox": [x1, y1, x2, y2]}


def holders(pairs):
    """[(frame, track_id or None), ...] straight through to build_touches."""
    return list(pairs)


# --------------------------------------------------------- ball position ----

def test_ball_position_is_the_bbox_centre_of_the_most_confident_detection():
    dets = [{"bbox": [0, 0, 10, 10], "conf": 0.9},
            {"bbox": [100, 100, 110, 110], "conf": 0.4}]
    assert ball_position(dets, 0.1) == (5.0, 5.0)


def test_detections_below_the_conf_floor_are_not_the_ball():
    dets = [{"bbox": [0, 0, 10, 10], "conf": 0.04}]
    assert ball_position(dets, 0.1) is None


def test_no_detections_means_no_ball_not_a_guessed_position():
    assert ball_position([], 0.1) is None


def test_two_detections_are_never_averaged_into_a_position_the_ball_never_was():
    # a mean of the real ball and a false positive sits between them, where
    # nothing is. The winner must be one of the actual detections.
    dets = [{"bbox": [0, 0, 10, 10], "conf": 0.9},
            {"bbox": [1000, 1000, 1010, 1010], "conf": 0.5}]
    assert ball_position(dets, 0.1) == (5.0, 5.0)


# ------------------------------------------------------ per-frame holder ----

def test_ball_inside_a_lone_box_is_held_by_that_player():
    v = holder_at_frame((50, 100), [track(7, 0, 0, 100, 200),
                                    track(9, 500, 0, 600, 200)])
    assert v["status"] == "held"
    assert v["track_id"] == 7
    assert v["dist_px"] == 0.0


def test_two_bodies_nearly_tied_credit_nobody():
    # ball sits 20px from each of two 200px-tall bodies: both well inside the
    # gate, neither clearly nearer. A rebound scrum must not name a holder.
    v = holder_at_frame((120, 100), [track(7, 0, 0, 100, 200),
                                     track(9, 140, 0, 240, 200)])
    assert v["status"] == "contested"
    assert sorted(v["track_ids"]) == [7, 9]


def test_ball_far_from_every_body_is_in_the_air_and_held_by_nobody():
    v = holder_at_frame((5000, 5000), [track(7, 0, 0, 100, 200)])
    assert v["status"] == "too_far"
    assert "track_id" not in v          # never names a holder it rejected


def test_no_ball_detected_abstains_rather_than_interpolating():
    assert holder_at_frame(None, [track(7, 0, 0, 100, 200)])["status"] == "no_ball"


def test_no_tracked_bodies_abstains():
    assert holder_at_frame((50, 100), [])["status"] == "no_tracks"


def test_degenerate_zero_height_box_carries_no_opinion():
    v = holder_at_frame((50, 100), [track(7, 0, 100, 100, 100)])
    assert v["status"] == "no_tracks"


# --- referees and bench cannot hold the ball ---------------------------------

def test_a_labelled_referee_is_never_credited_with_the_ball():
    # found on real footage: HARD t3 is a DJ-labelled ref standing under the
    # basket, credited with a touch while the ball was up at the rim.
    v = holder_at_frame((50, 100), [track(3, 0, 0, 100, 200)], exclude={3})
    assert v["status"] == "no_tracks"
    assert "track_id" not in v


def test_a_referee_beside_the_ball_handler_does_not_make_her_touch_contested():
    # the ref must be dropped from candidacy ENTIRELY, not just from winning --
    # otherwise his presence steals a real player's touch by forcing a tie.
    tracks = [track(7, 0, 0, 100, 200), track(3, 140, 0, 240, 200)]
    assert holder_at_frame((120, 100), tracks)["status"] == "contested"
    v = holder_at_frame((120, 100), tracks, exclude={3})
    assert v["status"] == "held"
    assert v["track_id"] == 7


def test_excluding_nobody_leaves_the_measurement_unchanged():
    tracks = [track(7, 0, 0, 100, 200)]
    assert (holder_at_frame((50, 100), tracks)
            == holder_at_frame((50, 100), tracks, exclude=frozenset()))


# --- the load-bearing property: the gate is BODY HEIGHTS, not pixels ---------

def test_same_pixel_gap_is_too_far_for_a_small_distant_body():
    # 50px from a 100px-tall body = half her height away. Not a hold.
    v = holder_at_frame((150, 50), [track(7, 0, 0, 100, 100)])
    assert v["status"] == "too_far"


def test_same_pixel_gap_is_a_hold_for_a_large_near_body():
    # the SAME 50px gap against a 400px-tall body is 1/8th of her height.
    v = holder_at_frame((150, 50), [track(7, 0, 0, 100, 400)])
    assert v["status"] == "held"
    assert v["track_id"] == 7


def test_the_frozen_thresholds_are_the_ones_declared_before_the_first_run():
    # tasks/todo.md froze these BEFORE any result was seen (the accel_y guard).
    # A change here is a DECISION that needs a new gate run, not a tweak.
    assert HOLD_GATE_BODY_FRAC == 0.30
    assert MARGIN_BODY_FRAC == 0.15
    assert MIN_TOUCH_FRAMES == 6


def test_the_unseen_ceiling_is_DJs_fifteen_seconds():
    # ADOPTED 2026-07-28 after DJ watched the overlay end to end and confirmed
    # the box stays on the right girl through every bridged stretch. His
    # number, from basketball. Changing it needs another eyeball pass.
    assert MAX_GAP_SECONDS == 15.0


# ------------------------------------------------------------- touch runs ---

def test_a_blink_length_run_is_dropped_because_it_is_a_pass_flying_past():
    ts = build_touches(holders([(f, 7) for f in range(3)]
                               + [(f, None) for f in range(3, 40)]))
    assert ts == []


def test_a_run_at_the_minimum_length_is_kept():
    ts = build_touches(holders([(f, 7) for f in range(MIN_TOUCH_FRAMES)]))
    assert len(ts) == 1
    assert ts[0]["held_frames"] == MIN_TOUCH_FRAMES


def test_a_short_detector_dropout_does_not_split_one_touch():
    seq = ([(f, 7) for f in range(10)]
           + [(f, None) for f in range(10, 14)]      # 4-frame gap
           + [(f, 7) for f in range(14, 20)])
    ts = build_touches(holders(seq))
    assert len(ts) == 1
    assert (ts[0]["start_frame"], ts[0]["end_frame"]) == (0, 19)
    assert ts[0]["held_frames"] == 16                # gap frames not credited


def test_a_gap_past_the_ceiling_does_split_into_two_touches():
    seq = ([(f, 7) for f in range(10)]
           + [(f, None) for f in range(10, 30)]      # 20-frame gap
           + [(f, 7) for f in range(30, 40)])
    ts = build_touches(holders(seq), max_gap=8)
    assert len(ts) == 2
    assert (ts[0]["start_frame"], ts[0]["end_frame"]) == (0, 9)
    assert (ts[1]["start_frame"], ts[1]["end_frame"]) == (30, 39)


def test_a_touch_never_outlasts_the_evidence_for_it():
    # she is credited through f9, then the system abstains, then a DIFFERENT
    # body appears. Her touch must end at 9 -- not at the tail of the gap.
    seq = ([(f, 7) for f in range(10)]
           + [(f, None) for f in range(10, 15)]
           + [(f, 3) for f in range(15, 25)])
    ts = build_touches(holders(seq))
    assert ts[0]["track_id"] == 7
    assert ts[0]["end_frame"] == 9


def test_the_ball_changing_hands_ends_one_touch_and_starts_another():
    seq = [(f, 7) for f in range(10)] + [(f, 3) for f in range(10, 20)]
    ts = build_touches(holders(seq))
    assert [t["track_id"] for t in ts] == [7, 3]
    assert (ts[0]["end_frame"], ts[1]["start_frame"]) == (9, 10)


def test_no_holder_anywhere_produces_no_touches():
    assert build_touches(holders([(f, None) for f in range(100)])) == []


# --- the FLICKER GUARD (DJ's occlusion worry, 2026-07-28) -------------------

def test_a_one_frame_flicker_cannot_steal_a_touch():
    # occlusion puts another body nearest the ball for a single frame.
    seq = [(f, 7) for f in range(10)] + [(10, 3)] + [(f, 7) for f in range(11, 20)]
    ts = build_touches(holders(seq))
    assert [t["track_id"] for t in ts] == [7]


def test_a_short_flicker_does_not_chop_the_real_touch_in_half():
    # the quieter harm: even a discarded blip used to END her run and start a
    # new one, costing credited time on BOTH sides of it.
    seq = [(f, 7) for f in range(10)] + [(10, 3)] + [(f, 7) for f in range(11, 20)]
    ts = build_touches(holders(seq))
    assert len(ts) == 1
    assert (ts[0]["start_frame"], ts[0]["end_frame"]) == (0, 19)
    assert ts[0]["held_frames"] == 19          # the flicker frame credits nobody


def test_a_five_frame_flicker_is_still_noise_at_the_six_frame_floor():
    # the longest flicker measured on real footage (HARD) was 5 frames.
    seq = ([(f, 7) for f in range(10)] + [(f, 3) for f in range(10, 15)]
           + [(f, 7) for f in range(15, 25)])
    ts = build_touches(holders(seq))
    assert [t["track_id"] for t in ts] == [7]


def test_a_SUSTAINED_change_of_hands_is_still_believed():
    # the guard must not make real handovers invisible: once the new holder
    # persists for min_frames, the handover is proven and accepted.
    seq = [(f, 7) for f in range(10)] + [(f, 3) for f in range(10, 20)]
    ts = build_touches(holders(seq))
    assert [t["track_id"] for t in ts] == [7, 3]
    assert (ts[0]["end_frame"], ts[1]["start_frame"]) == (9, 10)


def test_flicker_frames_are_credited_to_nobody_not_to_the_first_girl():
    # abstention, not a gift: the blip frames are excluded from her held count
    # even though they no longer break her run.
    seq = [(f, 7) for f in range(10)] + [(10, 3), (11, 3)] + [(f, 7) for f in range(12, 20)]
    t = build_touches(holders(seq))[0]
    assert t["span_frames"] == 20
    assert t["held_frames"] == 18          # 20 minus the 2 flicker frames


# ------------------------------------------------------- identity join ------

def _touch(tid=7, a=0, b=5):
    return {"track_id": tid, "start_frame": a, "end_frame": b}


def test_one_identity_confirmed_on_every_credited_frame_is_attributed():
    held = list(range(6))
    ident = {(f, 7): (0, 12, "confirmed") for f in held}
    out = attribute(_touch(), ident, held, {(0, 12): 23})
    assert out["status"] == "attributed"
    assert out["identity_key"] == [0, 12]
    assert out["jersey_number"] == 23


def test_a_single_unconfirmed_frame_blocks_attribution():
    held = list(range(6))
    ident = {(f, 7): (0, 12, "confirmed") for f in held}
    ident[(3, 7)] = (0, 12, "candidate")
    out = attribute(_touch(), ident, held)
    assert out["status"] == "review_item"


def test_a_touch_spanning_two_identities_is_never_credited_to_either():
    held = list(range(6))
    ident = {(f, 7): (0, 12 if f < 3 else 99, "confirmed") for f in held}
    out = attribute(_touch(), ident, held, {(0, 12): 23, (0, 99): 44})
    assert out["status"] == "review_item"
    assert out["identity_keys"] == [[0, 12], [0, 99]]
    assert out["identity_key"] is None
    assert out["jersey_number"] is None      # two candidates -> name NEITHER


def test_the_same_identity_id_in_two_windows_is_two_different_records():
    # identity_id is a PER-WINDOW counter, so (window, identity_id) is the key.
    # Ignoring the window would silently merge two records.
    held = list(range(6))
    ident = {(f, 7): (0 if f < 3 else 1, 12, "confirmed") for f in held}
    out = attribute(_touch(), ident, held)
    assert out["identity_keys"] == [[0, 12], [1, 12]]
    assert out["status"] == "review_item"     # unnamed records: cannot prove same


def test_two_identity_records_carrying_the_SAME_jersey_are_one_girl():
    # a touch crossing a window boundary picks up a second record for the same
    # player. #32 must not lose her number just for playing across a boundary.
    held = list(range(6))
    ident = {(f, 7): (0 if f < 3 else 1, 39 if f < 3 else 4, "confirmed")
             for f in held}
    out = attribute(_touch(), ident, held, {(0, 39): 32, (1, 4): 32})
    assert out["jersey_number"] == 32
    assert out["status"] == "attributed"


def test_two_records_with_DIFFERENT_jerseys_are_still_two_girls():
    held = list(range(6))
    ident = {(f, 7): (0 if f < 3 else 1, 39 if f < 3 else 4, "confirmed")
             for f in held}
    out = attribute(_touch(), ident, held, {(0, 39): 32, (1, 4): 44})
    assert out["jersey_number"] is None
    assert out["status"] == "review_item"


def test_a_named_record_plus_an_unnamed_one_refuses_rather_than_assuming():
    held = list(range(6))
    ident = {(f, 7): (0 if f < 3 else 1, 39 if f < 3 else 4, "confirmed")
             for f in held}
    out = attribute(_touch(), ident, held, {(0, 39): 32})      # (1,4) unnamed
    assert out["jersey_number"] is None
    assert out["status"] == "review_item"


def test_partial_identity_coverage_is_a_review_item_not_an_attribution():
    held = list(range(6))
    ident = {(f, 7): (0, 12, "confirmed") for f in held if f != 5}
    out = attribute(_touch(), ident, held)
    assert out["status"] == "review_item"
    assert out["frames_with_identity"] == 5
    assert out["frames_credited"] == 6


def test_no_identity_data_says_so_rather_than_naming_anyone():
    out = attribute(_touch(), {}, list(range(6)))
    assert out["status"] == "no_identity_data"
    assert out["jersey_number"] is None


# ------------------------------- seen vs filled in (DJ, 2026-07-27) ---------

def test_a_touch_reports_seconds_seen_and_seconds_filled_in_separately():
    # 10 credited frames, a 4-frame dropout bridged, 6 more credited = 20 span,
    # 16 seen. Reporting only the total would let an assumption become a stat.
    seq = ([(f, 7) for f in range(10)]
           + [(f, None) for f in range(10, 14)]
           + [(f, 7) for f in range(14, 20)])
    t = build_touches(holders(seq))[0]
    assert t["held_frames"] == 16
    assert t["span_frames"] == 20
    assert t["span_frames"] - t["held_frames"] == 4      # the filled-in part


def test_a_touch_with_no_dropout_has_nothing_filled_in():
    t = build_touches(holders([(f, 7) for f in range(10)]))[0]
    assert t["held_frames"] == t["span_frames"]


# ------------------------------- who shot it (DJ's proposal) ----------------

def _t(tid, a, b, jersey=None):
    return {"track_id": tid, "start_frame": a, "end_frame": b,
            "identity": {"jersey_number": jersey}}


def test_the_shot_goes_to_the_last_player_seen_holding_the_ball():
    ts = [_t(7, 0, 20, 13), _t(3, 30, 50, 24)]
    assert shooter_from_touches(ts, 60)["track_id"] == 3


def test_a_touch_that_began_after_the_shot_cannot_be_the_shooter():
    ts = [_t(7, 0, 20), _t(3, 70, 90)]
    assert shooter_from_touches(ts, 60)["track_id"] == 7


def test_a_shot_out_of_a_players_own_ongoing_touch_credits_her():
    ts = [_t(7, 40, 80)]
    assert shooter_from_touches(ts, 60)["track_id"] == 7


def test_a_stale_memory_is_not_evidence_about_this_shot():
    # she last held it 100 frames (>3s) ago -- long enough to have passed it.
    ts = [_t(7, 0, 20)]
    assert shooter_from_touches(ts, 120, max_back_frames=60) is None
    assert shooter_from_touches(ts, 120, max_back_frames=None)["track_id"] == 7


def test_no_touch_before_the_shot_abstains_rather_than_reaching_further():
    assert shooter_from_touches([], 60) is None


def test_an_identity_the_registry_cannot_name_reports_unnamed_not_its_raw_id():
    # THE HAZARD THIS PINS: identity_id is an internal counter. TEST1 has a
    # real #13 on the roster AND an identity_id 13 belonging to someone else.
    # An unnamed identity must never leak its id where a jersey is expected.
    held = list(range(6))
    ident = {(f, 7): (0, 13, "confirmed") for f in held}
    out = attribute(_touch(), ident, held, numbers={})
    assert out["status"] == "attributed"
    assert out["jersey_number"] is None


def test_jersey_numbers_reads_the_registry_the_box_score_uses():
    reg = {"identities": [
        {"window": 0, "identity_id": 2, "roster_number": 13},
        {"window": 0, "identity_id": 3, "roster_number": None},
        {"window": 1, "identity_id": 2, "roster_number": 24}]}
    assert jersey_numbers(reg) == {(0, 2): 13, (1, 2): 24}
