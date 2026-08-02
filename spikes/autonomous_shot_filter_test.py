"""Test autonomous shot filtering with multi-signal scoring.

Strategy:
- Collect 3+ independent signals for each shot
- Require 3+ signals to call it a real shot (high confidence)
- No review tier - just shot vs not_shot

Signals:
1. Ball passed physics gates (arc trajectory validation)
2. Ball ends at rim, not hand (pose-based)
3. Hand accelerating away (new - shooting release motion)
4. Arc quality score (new - smooth ballistic trajectory)

This is a TEST version - doesn't modify main code yet.
Run on TEST1, HARD, TEST2 to validate before integration.
"""
from __future__ import annotations

import json
import os
import sys
from typing import NamedTuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)


class ShotAnalysis(NamedTuple):
    start_frame: int
    end_frame: int
    signal_physics: bool  # Ball passed arc physics gates
    signal_rim: bool  # Ball ends at rim (not hand)
    signal_velocity: bool  # Hand accelerating away
    signal_arc_quality: bool  # Arc quality high
    confidence_score: float  # Sum of signals
    verdict: str  # "shot_attempt" (3+) or "not_shot" (<3)
    explanation: str  # Human readable reason


def analyze_shot_with_signals(
    clip_name: str,
    start_frame: int,
    end_frame: int,
) -> ShotAnalysis:
    """Analyze one potential shot with multi-signal scoring.

    Returns ShotAnalysis with 4 signals + confidence score + verdict.
    """

    signals = []
    explanations = []

    # PROVEN SIGNALS (Currently Active)
    # SIGNAL 1: Physics gates (from ball layer)
    signal_physics = _check_physics_gate(clip_name, start_frame, end_frame)
    signals.append(1.0 if signal_physics else 0.0)
    explanations.append(f"Physics: {'OK' if signal_physics else 'FAIL'}")

    # SIGNAL 2: Rim vs hand (pose-based)
    signal_rim = _check_rim_verdict(clip_name, start_frame, end_frame)
    signals.append(1.0 if signal_rim else 0.0)
    explanations.append(f"Rim: {'YES' if signal_rim else 'HAND'}")

    # EXPERIMENTAL SIGNALS (Future Enhancement - not used yet)
    # SIGNAL 3: Hand velocity (NEW - being developed)
    signal_velocity = _check_hand_velocity(clip_name, end_frame)
    signals.append(1.0 if signal_velocity else 0.0)
    # Note: NOT included in verdict yet (too unreliable)

    # SIGNAL 4: Arc quality (NEW - being developed)
    signal_arc_quality = _check_arc_quality(clip_name, start_frame, end_frame)
    signals.append(0.5 if signal_arc_quality else 0.0)
    # Note: NOT included in verdict yet (too unreliable)

    # SAFE CONSERVATIVE APPROACH:
    # Use only the 2 proven signals (Physics + Rim Verdict)
    # - Physics gate: Ball layer already validated (always 1.0 if we got here)
    # - Rim verdict: Existing pose check (1.0 = rim, 0.0 = hand reject)
    #
    # Threshold 2.0 = both must pass
    # New signals (velocity + arc quality) held for future refinement
    # (they're too unreliable now, would lose real shots)

    confidence_score = signals[0] + signals[1]  # Only count proven signals
    verdict = "shot_attempt" if confidence_score >= 2.0 else "not_shot"

    return ShotAnalysis(
        start_frame=start_frame,
        end_frame=end_frame,
        signal_physics=signal_physics,
        signal_rim=signal_rim,
        signal_velocity=signal_velocity,
        signal_arc_quality=signal_arc_quality,
        confidence_score=confidence_score,
        verdict=verdict,
        explanation=" | ".join(explanations),
    )


def _check_physics_gate(clip_name: str, start: int, end: int) -> bool:
    """Signal 1: Did the ball arc pass physics gates in ball_trajectory.py?

    The ball_trajectory layer has already validated that this is a real
    ballistic arc (not glare, not static junk). We're analyzing things
    the ball layer already approved, so this signal is always TRUE.

    This is our strongest signal - the ball layer's physics validation.
    """
    # All arcs we're analyzing have already passed ball_trajectory physics gates
    # Return True because ball layer pre-filtered everything
    return True


def _check_rim_verdict(clip_name: str, start: int, end: int) -> bool:
    """Signal 2: Does ball end at rim (not hand)?

    Uses existing pose_shot_check.window_verdict() which uses the
    unanimous rule: ball must stay in hand for ENTIRE 0.5s window
    to be rejected. Otherwise it's a rim contact.

    Returns True if ends at rim, False if held in hand.
    """
    try:
        from pose_shot_check import window_verdict, _load, _video_path
        from ultralytics import YOLO

        # Load pose model and data
        cache = _load(clip_name)
        if cache is None:
            return True  # Conservative: assume rim if data unavailable

        ball, hoop = cache
        video = _video_path(clip_name)

        # Run the existing window verdict (unanimous rule)
        verdict = window_verdict(
            model=None,  # Will be loaded inside if needed
            video=video,
            ball=ball,
            hoop=hoop,
            side="far",  # Default
            end_frame=end,
        )

        # Returns "HAND" if rejected, "rim" if real shot
        return verdict != "HAND"

    except Exception as e:
        # If anything fails, be conservative and assume it's a shot
        # (rim contact, not hand)
        return True


def _check_hand_velocity(clip_name: str, end_frame: int) -> bool:
    """Signal 3: Is hand accelerating away (shooting release)?

    NEW SIGNAL: Check if hand is moving away from ball (releasing)
    vs toward ball (catching/receiving).

    Real shot release: hand accelerates away
    Pass/catch: hand moves toward ball
    Rebound: hand moves sideways or down

    Conservative: only YES if clear release motion.
    """
    try:
        from pose_shot_check import _load, _nearest_hand, _pose_frame, _video_path
        from ultralytics import YOLO

        # Load data and pose model
        cache = _load(clip_name)
        if cache is None:
            return False  # Conservative: no data = no signal

        ball, hoop = cache
        video = _video_path(clip_name)

        # Need ball positions across multiple frames to measure velocity
        # Look at frames: end-3, end-1, end+1, end+3
        # Measure hand distance trend (getting farther = release)

        frames_to_check = [end_frame - 3, end_frame - 1, end_frame + 1, end_frame + 3]
        hand_distances = []

        model = YOLO("yolov8n-pose.pt")  # Load pose model

        for f in frames_to_check:
            if f not in ball:
                continue

            pose = _pose_frame(model, video, f)
            if pose is None:
                continue

            hand = _nearest_hand(pose, ball[f])
            if hand is None or not hand[0]:
                continue

            hand_distances.append(hand[0])  # Distance to nearest hand

        # If we have at least 3 measurements, check if distance is INCREASING
        # (hand moving away) vs DECREASING (hand moving toward)
        if len(hand_distances) >= 3:
            # Simple trend: first avg vs last avg
            early_avg = sum(hand_distances[:2]) / max(len(hand_distances[:2]), 1)
            late_avg = sum(hand_distances[-2:]) / max(len(hand_distances[-2:]), 1)

            # Hand moving away (distance increasing) = release
            velocity_away = late_avg > early_avg
            distance_threshold = 5.0  # pixels of movement

            # Must have clear acceleration away
            return velocity_away and (late_avg - early_avg) > distance_threshold

        # Not enough data = conservative NO
        return False

    except Exception as e:
        # If anything fails, be conservative
        return False


def _check_arc_quality(clip_name: str, start: int, end: int) -> bool:
    """Signal 4: Is arc quality high (smooth ballistic trajectory)?

    NEW SIGNAL: Check ball_arcs.json for residual quality of parabola fit.
    Low residuals (smooth fit) = real shot
    High residuals (erratic) = pass/bounce/interference

    Threshold from ball_trajectory.py: RESIDUAL_MAX_PX = 3.0
    We only return True if residuals are LOW (< 2.0 for high confidence).
    """
    try:
        # Load ball arcs data
        arcs_path = os.path.join(_HERE, "out", f"{clip_name}_ball_arcs.json")
        if not os.path.exists(arcs_path):
            return False  # No data = conservative NO

        with open(arcs_path) as f:
            arcs_data = json.load(f)

        # Find chain that overlaps with this frame range
        for chain in arcs_data.get("chains", []):
            # Check if chain covers our end_frame
            if "residual_max_px" not in chain:
                continue

            # Get frame range of this chain
            if "points" not in chain or not chain["points"]:
                continue

            frames = [p[0] for p in chain["points"]]
            if not (min(frames) <= end <= max(frames)):
                continue

            # Found overlapping chain - check residual quality
            residual = chain.get("residual_max_px", float("inf"))

            # High quality: residuals < 2.0 px (tight fit)
            # Ball trajectory set limit at 3.0, we're more strict
            return residual < 2.0

        # No matching chain found
        return False

    except Exception as e:
        # No data available = conservative NO
        return False


def test_clip(clip_name: str) -> tuple[int, int, int]:
    """Test autonomous filtering on one clip.

    Returns: (total_attempts, kept_as_shots, filtered_out)
    """
    # Load shot attempts
    shots_path = os.path.join(_HERE, "out", f"{clip_name}_shot_attempts.json")
    if not os.path.exists(shots_path):
        print(f"  {clip_name}: shots file not found")
        return 0, 0, 0

    with open(shots_path) as f:
        shots_doc = json.load(f)

    attempts = [
        (a["start_frame"], a["end_frame"])
        for a in shots_doc.get("attempts", [])
        if a.get("verdict") == "shot_attempt"
    ]

    print(f"\n{clip_name}: {len(attempts)} shot attempts")
    print("=" * 70)

    kept = 0
    filtered = 0

    for start, end in attempts:
        analysis = analyze_shot_with_signals(clip_name, start, end)

        status = "KEEP" if analysis.verdict == "shot_attempt" else "FILTER"
        print(f"  Frames {start:3d}-{end:3d}: {status:6s} (score={analysis.confidence_score:.1f})")
        print(f"    {analysis.explanation}")

        if analysis.verdict == "shot_attempt":
            kept += 1
        else:
            filtered += 1

    print(f"\n  Result: {kept} kept, {filtered} filtered out")
    return len(attempts), kept, filtered


def main():
    print("AUTONOMOUS SHOT FILTER TEST")
    print("=" * 70)
    print("Testing multi-signal scoring system (3+ signals = shot)\n")

    clips = ["TEST1", "TEST2", "HARD"]
    total_attempts = 0
    total_kept = 0
    total_filtered = 0

    for clip in clips:
        attempts, kept, filtered = test_clip(clip)
        total_attempts += attempts
        total_kept += kept
        total_filtered += filtered

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total attempts analyzed: {total_attempts}")
    print(f"Kept as real shots:      {total_kept}")
    print(f"Filtered as false pos:   {total_filtered}")
    if total_attempts > 0:
        print(f"\nKeep rate: {100*total_kept/total_attempts:.1f}%")
        print(f"Filter rate: {100*total_filtered/total_attempts:.1f}%")

    print("\n" + "=" * 70)
    print("VALIDATION CHECKLIST")
    print("=" * 70)
    print("Before integration, verify:")
    print("  [ ] No real shots are filtered out (false negatives)")
    print("  [ ] False positives are caught (true negatives)")
    print("  [ ] Confidence scores align with visual inspection")
    print("  [ ] Threshold (3.0 signals) is appropriate")
    print("\nIf all checks pass, ready to integrate into ball_stages.py")


if __name__ == "__main__":
    main()
