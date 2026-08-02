"""TEST 16: does an off-the-shelf POSE model separate real shots from the
three DJ-confirmed false positives, using no new labels at all?

WHY THIS TEST EXISTS. v3 claims three non-shots as shots (TEST 11 + TEST 12
follow-up, all DJ-confirmed): a rebound caught and dished out, a cross-court
pass, and a player simply holding the ball on an inbounds. TEST 12's follow-up
PROVED these cannot be separated by ball trajectory: a pass thrown by a human
arm and a shot thrown by a human arm are the same physics, and accel_y --
which looked like a clean separator at n=9 -- was destroyed by exactly that
case. What all three DO fail is the question a human answers instantly:
"is a person shooting AT the hoop here?" That question is about ARMS, and
COCO-pretrained pose models give wrists/elbows/shoulders for free.

WHAT IT MEASURES (raw quantities only -- NO rule, NO threshold, NO verdict):
at the arc's first frame (the claimed release) and its last frame (the claimed
arrival), where is the ball relative to the nearest person's HANDS, and is that
person's arm raised? Distances are normalised by that person's bbox height so
a distant player and a near player are comparable.

DISCIPLINE (the accel_y lesson, TEST_LOG 2026-07-23): this script prints the
table and stops. Any rule fitted to these 10 events afterwards is a
HYPOTHESIS, not a gate, and must survive a clip it was not built on (TEST 19)
before anyone calls it a fix.

Read-only. Writes one json of raw measurements, nothing else.

Run:  .venv/Scripts/python spikes/pose_shot_check.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

POSE_WEIGHTS = "yolo11x-pose.pt"   # COCO-pretrained, zero labelling required
IMGSZ = 1280                       # the project's proven optimum (DECISIONS 20)
KP_CONF = 0.3                      # below this a keypoint is a guess, not a reading
L_SHO, R_SHO, L_WRI, R_WRI = 5, 6, 9, 10

# Every near-rim judgement call that now has DJ ground truth. Spans are the
# arcs v3 ACTUALLY claimed (TEST 11 raw output), not the verified spans, so we
# measure what the classifier saw.
EVENTS = [
    # clip,   start, end,  hoop,   truth,  what it really is
    ("HARD",   351,  375, "near", "REAL", "verified jump shot"),
    ("HARD",  1177, 1214, "far",  "REAL", "verified jump shot"),
    ("TEST1",   58,   77, "far",  "REAL", "verified jump shot A"),
    ("TEST1",  166,  184, "far",  "REAL", "verified layup 1"),
    ("TEST1",  236,  250, "far",  "REAL", "verified layup 2"),
    ("TEST1",  314,  327, "far",  "REAL", "verified jump shot B"),
    ("TEST1",  571,  589, "near", "REAL", "verified layup 3"),
    ("HARD",   403,  415, "near", "FAKE", "rebound caught -> dished out"),
    ("HARD",  1352, 1377, "far",  "FAKE", "cross-court pass"),
    ("HARD",  2234, 2250, "far",  "FAKE", "player HOLDING ball, inbounds"),
]

BALL_LOG = {"HARD": "HARD_ball_spike_log_ball_finetuned_v3_gpu.json",
            "TEST1": "TEST1_ball_spike_log_ball_finetuned_v3.json",
            "TEST4": "TEST4_ball_spike_log_ball_finetuned_v3_gpu.json"}

# Clips with no ClipConfig entry (TEST4 is a raw holdout -- it deliberately has
# no roster/calibration, neither of which any measurement here needs).
VIDEO_PATHS = {"TEST4": r"C:\Users\djcha\New folder\Throw away repos"
                        r"\Basketball Analyer CV System Test\clips\Test4.mp4"}

WINDOW_FRAMES = 15      # 0.5s -- the pre-specified window variant (TEST 16)


def _video_path(clip):
    if clip in VIDEO_PATHS:
        return VIDEO_PATHS[clip]
    import clip_config
    return getattr(clip_config, f"{clip}_CLIP").video_path


def _load(clip):
    log = json.load(open(os.path.join(_HERE, "out", BALL_LOG[clip]), encoding="utf-8"))
    ball = {}
    for fr in log["frames"]:
        dets = [d for d in fr["detections"] if d["conf"] >= 0.10]
        if dets:
            b = max(dets, key=lambda d: d["conf"])["bbox"]
            ball[fr["frame_index"]] = (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))
    hoop_doc = json.load(open(os.path.join(_HERE, "out", f"{clip}_hoop_track.json"),
                              encoding="utf-8"))
    hoop = {f["frame_index"]: f for f in hoop_doc["frames"]}
    return ball, hoop


def _pose_frame(model, video, frame_index):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return []
    res = model.predict(img, imgsz=IMGSZ, conf=0.25, verbose=False)[0]
    people = []
    if res.keypoints is None:
        return people
    kxy = res.keypoints.xy.cpu().numpy()
    kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
    boxes = res.boxes.xyxy.cpu().numpy()
    for i in range(len(kxy)):
        h = float(boxes[i][3] - boxes[i][1])
        conf = kcf[i] if kcf is not None else np.ones(len(kxy[i]))
        people.append({"kp": kxy[i], "kpc": conf, "h": h})
    return people


def _nearest_hand(people, pt):
    """Closest wrist to a point.

    Returns (norm_dist, raw_dist, arm_raised, body_height). The body height is
    handed back so the RIM distance can be expressed in the same units -- the
    only way "did the ball end up at the rim or in someone's hands?" is a fair
    comparison rather than pixels against body-lengths.
    """
    best = None
    for p in people:
        for wri, sho in ((L_WRI, L_SHO), (R_WRI, R_SHO)):
            if p["kpc"][wri] < KP_CONF:
                continue
            w = p["kp"][wri]
            d = float(np.hypot(w[0] - pt[0], w[1] - pt[1]))
            nd = d / max(p["h"], 1e-6)
            # image y grows downward -> wrist ABOVE shoulder means smaller y
            raised = bool(p["kpc"][sho] >= KP_CONF and w[1] < p["kp"][sho][1])
            if best is None or nd < best[0]:
                best = (nd, d, raised, p["h"])
    return best if best else (float("nan"), float("nan"), None, float("nan"))


def ball_center_by_frame(frames):
    """frames: [{frame_index, detections}, ...] ALREADY conf-filtered. Same
    max-conf-detection-center pick as _load, exposed so callers with their
    own detections doc (ball_stages) don't need a second copy of this."""
    out = {}
    for fr in frames:
        dets = fr["detections"]
        if dets:
            b = max(dets, key=lambda d: d["conf"])["bbox"]
            out[fr["frame_index"]] = (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))
    return out


def window_majority(votes):
    """Pure decision rule, no I/O: majority of HAND/rim votes, or None if
    there were no votes at all (pose/ball unavailable in the window)."""
    if not votes:
        return None
    return "HAND" if votes.count("HAND") * 2 > len(votes) else "rim"


def window_unanimous(votes):
    """Pure decision rule, no I/O: ONLY calls HAND if every vote in the
    window says HAND. Chosen over window_majority (2026-07-31 real-pipeline
    experiment, spikes/player_signal_experiment.py) because a real made shot
    is often rebounded by a nearby player within the window -- majority
    over-triggers on that (lost TEST1's verified layup 3), while requiring
    unanimity only rejects a ball that NEVER touches the rim at all (a true
    catch/pass), matching HARD's 2 known false positives exactly. Same 8/9
    score as the single-frame rule but keeps the window's noise resistance."""
    if not votes:
        return None
    return "HAND" if all(v == "HAND" for v in votes) else "rim"


def window_verdict(model, video, ball, hoop, side, end_frame,
                    window_frames=WINDOW_FRAMES, step=3):
    """THE WINDOW rule (TEST 16/19, tuned to UNANIMOUS 2026-07-31): does the
    ball stay in a HAND for the ENTIRE 0.5s following the claimed arrival,
    rather than trusting the single end frame (which flips on 3/10 known
    events) or a plain majority (which over-rejects real shots a nearby
    player rebounds quickly). Returns "HAND", "rim", or None (no votes --
    ball/pose unavailable)."""
    votes = []
    for f in range(end_frame, end_frame + window_frames + 1, step):
        v = _ends_at_hand(model, video, ball, hoop, side, f)
        if v:
            votes.append(v[0])
    return window_unanimous(votes)


def _ends_at_hand(model, video, ball, hoop, side, frame):
    """The frozen TEST 16 rule at ONE frame: is the ball nearer a hand or the
    rim, in body-height units? Returns (verdict, hand_norm, rim_norm) or None."""
    f = None
    for d in range(0, 6):
        for c in (frame - d, frame + d):
            if c in ball:
                f = c
                break
        if f:
            break
    if f is None:
        return None
    arr = _nearest_hand(_pose_frame(model, video, f), ball[f])
    hp = hoop.get(f, {}).get(f"hoop_{side}_px")
    if hp is None or not np.isfinite(arr[0]):
        return None
    rim_n = float(np.hypot(ball[f][0] - hp[0], ball[f][1] - hp[1])) / max(arr[3], 1e-6)
    return ("HAND" if arr[0] < rim_n else "rim"), arr[0], rim_n


def main():
    from ultralytics import YOLO

    events = EVENTS
    if "--events" in sys.argv:
        spec = json.load(open(sys.argv[sys.argv.index("--events") + 1], encoding="utf-8"))
        events = [(e["clip"], e["start"], e["end"], e.get("hoop", "far"),
                   e.get("truth", "?"), e.get("desc", "")) for e in spec]

    model = YOLO(POSE_WEIGHTS)
    cache, rows = {}, []

    for clip, s, e, hoop_side, truth, desc in events:
        if clip not in cache:
            cache[clip] = _load(clip)
        ball, hoop = cache[clip]
        video = _video_path(clip)

        def at(frame):
            """Ball position at `frame`, or the nearest frame that has one."""
            for off in range(0, 6):
                for f in (frame - off, frame + off):
                    if f in ball:
                        return f, ball[f]
            return None, None

        fs, ball_s = at(s)
        fe, ball_e = at(e)
        if ball_s is None or ball_e is None:
            print(f"  !! {clip} {s}-{e}: no ball detection near the endpoints, skipped")
            continue

        rel = _nearest_hand(_pose_frame(model, video, fs), ball_s)
        arr = _nearest_hand(_pose_frame(model, video, fe), ball_e)

        hp = hoop.get(fe, {}).get(f"hoop_{hoop_side}_px")
        rim_d = float(np.hypot(ball_e[0] - hp[0], ball_e[1] - hp[1])) if hp else float("nan")
        # same body-height units as the hand columns, so the two are comparable
        rim_norm = rim_d / max(arr[3], 1e-6)

        # PRE-SPECIFIED window variant (TEST 16 verdict, written down BEFORE
        # this holdout ran): a caught or held ball STAYS in someone's hands,
        # a ball that reaches the rim does not linger in anyone's grip. So
        # take the majority verdict over the 0.5s FOLLOWING the arc, rather
        # than trusting one frame -- which is what flipped 3 of 10 events.
        votes = []
        for f in range(e, e + WINDOW_FRAMES + 1, 3):
            v = _ends_at_hand(model, video, ball, hoop, hoop_side, f)
            if v:
                votes.append(v[0])
        win = window_majority(votes)

        rows.append(dict(clip=clip, start=s, end=e, truth=truth, desc=desc,
                         release_hand_norm=rel[0], release_hand_px=rel[1],
                         release_arm_raised=rel[2],
                         arrival_hand_norm=arr[0], arrival_hand_px=arr[1],
                         arrival_rim_px=rim_d, arrival_rim_norm=rim_norm,
                         ends_at_hand_not_rim=bool(arr[0] < rim_norm),
                         window_verdict=win, window_votes=votes))

    print(f"\n{'='*104}")
    print("TEST 16 -- RAW POSE MEASUREMENTS (no rule applied, no verdict drawn)")
    print(f"{'='*104}")
    print(f"{'truth':5s} {'clip':5s} {'span':11s} {'rel.hand':>9s} {'armUp':>6s} "
          f"{'arr.hand':>9s} {'arr.rim':>8s} {'ENDS AT':>8s}  what it really is")
    print(f"{'-'*104}")
    for r in sorted(rows, key=lambda r: (r["truth"], r["clip"], r["start"])):
        print(f"{r['truth']:5s} {r['clip']:5s} {str(r['start'])+'-'+str(r['end']):11s} "
              f"{r['release_hand_norm']:9.2f} {str(r['release_arm_raised']):>6s} "
              f"{r['arrival_hand_norm']:9.2f} {r['arrival_rim_norm']:8.2f} "
              f"{('HAND' if r['ends_at_hand_not_rim'] else 'rim'):>8s}"
              f"{str(r.get('window_verdict')):>7s}  {r['desc']}")
    print(f"{'-'*104}")
    print("All distances in units of the nearest person's body height, so near and far")
    print("players compare fairly (0.3 = about a forearm; 3.0 = nowhere near anybody).")
    print("ENDS AT = whichever the ball finished closer to, its own hand or its own rim.")

    out = os.path.join(_HERE, "out", "pose_shot_check.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nraw -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
