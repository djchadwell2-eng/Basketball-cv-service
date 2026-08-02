"""Player-tracker plan item 4/TEST 12 follow-up: does a jersey-COLOR
consistency check catch the confirmed ID switch (TEST 12: TEST1 id=17,
f=196 white Milford -> f=221 green Little Miami) without needing team
rosters/OCR? If yes, this is the safeguard that could make a looser
match_thresh (TEST 6/12) actually adoptable.

Method: for every track that goes silent for >1 frame and then reappears
under the SAME id (a "reattach" -- exactly the moment TEST 12's switch
happened), compare the jersey color just before the gap to the jersey
color just after. Reference scale = simple unsupervised 2-cluster split
of a broad sample of this clip's own jersey crops (basketball has ~2
team colors; no roster/OCR needed, same "no hardcoded RGB, no new human
input" spirit as phase2/color_tiebreak.py, whose crop_color_signature
this reuses). A reattach whose before/after crops land in DIFFERENT
clusters (with margin, else abstain) is flagged SUSPECT.

Read-only: prints findings, writes nothing into any cache. Adoption is a
separate decision -- this is measurement only.

Run:  .venv/Scripts/python spikes/tracker_color_reattach_check.py TEST1 spikes/out/TEST1_tracks_mt09.json
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from color_tiebreak import crop_color_signature
from ocr_reader import jersey_crop

MARGIN_RATIO = 1.4          # same abstention discipline as color_tiebreak.COLOR_MARGIN_RATIO
MIN_GAP_FRAMES = 2          # a 1-frame hole is normal detector flicker, not a real "lost" event
SAMPLE_STRIDE = 5           # every 5th frame's every track, for building the 2 color clusters
REF_IOU = 0.5               # a track box this close to a Ref-class box IS that referee


def _in_any_region(bbox, regions):
    """True if the box's CENTRE falls inside any excluded screen region.

    Centre (not overlap) on purpose: a real player standing NEAR the scorebug
    corner still has their centre outside it, so this drops the boxes that sit
    ON the graphic without discarding legitimate players in that corner.
    """
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for (x1, y1, x2, y2) in regions)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def apply_junk_filter(frames, clip_name, ref_boxes=None):
    """TEST 15: drop detections that are not tracked athletes before any colour
    reasoning happens.

    TEST 13 found 4 of its 6 false alarms were the person detector firing on the
    ANIMATED SCOREBOARD GRAPHIC and 1 more on a REFEREE -- not jersey-colour
    noise at all, a separate detector bug. Both sources are already known to the
    project: the scorebug rectangle is the per-clip `exclude_regions` the
    calibration engine masks out of SIFT, and refs are a class the fine-tuned
    model already separates (TEST 4: 3/frame at conf>=0.4).

    Mutates nothing -- returns a new frames list plus a count of what went.
    """
    import clips_config
    regions = clips_config.CLIPS[clip_name].get("exclude_regions", [])
    ref_boxes = ref_boxes or {}

    # Referees are excluded WHOLE-TRACK, not detection-by-detection. Dropping
    # single detections punches holes in a referee's track wherever the Ref
    # detector happened to blink, and every hole becomes a fabricated "reattach
    # event" the tracker never actually experienced -- measured: doing it
    # per-detection RAISED the event count 104 -> 126 and left the referee
    # track flagged anyway, now full of artificial gaps.
    ref_hits: dict = {}
    ref_total: dict = {}
    for fr in frames:
        refs = ref_boxes.get(str(fr["frame_index"]), [])
        for t in fr["tracks"]:
            tid = t["track_id"]
            ref_total[tid] = ref_total.get(tid, 0) + 1
            if any(_iou(t["bbox"], rb) >= REF_IOU for rb in refs):
                ref_hits[tid] = ref_hits.get(tid, 0) + 1
    ref_tracks = {tid for tid, n in ref_total.items()
                  if ref_hits.get(tid, 0) / max(n, 1) >= 0.5}

    n_board = n_ref = 0
    out = []
    for fr in frames:
        kept = []
        for t in fr["tracks"]:
            if _in_any_region(t["bbox"], regions):
                n_board += 1
                continue
            if t["track_id"] in ref_tracks:
                n_ref += 1
                continue
            kept.append(t)
        out.append({"frame_index": fr["frame_index"], "tracks": kept})
    if ref_tracks:
        print(f"[reattach-check] referee tracks dropped whole: {sorted(ref_tracks)}")
    return out, n_board, n_ref


def _kmeans2(points, iters=25, seed=0):
    """Tiny k=2 k-means on 3D color points. No sklearn dependency needed
    for two clusters on a few hundred points."""
    rng = np.random.RandomState(seed)
    pts = np.array(points, dtype=float)
    c = pts[rng.choice(len(pts), 2, replace=False)]
    for _ in range(iters):
        d0 = np.linalg.norm(pts - c[0], axis=1)
        d1 = np.linalg.norm(pts - c[1], axis=1)
        assign = (d1 < d0).astype(int)
        new_c = np.array([pts[assign == k].mean(axis=0) if (assign == k).any() else c[k]
                          for k in (0, 1)])
        if np.allclose(new_c, c):
            break
        c = new_c
    return c


def _classify(sig, centroids, margin=MARGIN_RATIO):
    d = [float(np.linalg.norm(np.array(sig) - c)) for c in centroids]
    order = sorted(range(len(d)), key=lambda i: d[i])
    best, second = d[order[0]], d[order[1]]
    if second < margin * max(best, 1e-6):
        return None            # abstain -- not clearly one cluster over the other
    return order[0]


def main():
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    tracks_json = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(_HERE, "out", f"{clip_name}_tracks_mt09.json")

    import clip_config
    CLIP = getattr(clip_config, f"{clip_name}_CLIP")
    clip_config.ACTIVE_CLIP = CLIP
    import run_tracking

    doc = json.load(open(tracks_json, encoding="utf-8"))

    # TEST 15: junk-detection pre-filter (scoreboard graphic + referees).
    # --nofilter reproduces TEST 13's original, unfiltered numbers exactly.
    if "--nofilter" in sys.argv:
        print("[reattach-check] junk filter OFF (TEST 13 reproduction mode)")
    else:
        ref_path = os.path.join(_HERE, "out", f"{clip_name}_ref_boxes.json")
        ref_boxes = json.load(open(ref_path, encoding="utf-8")) if os.path.exists(ref_path) else {}
        if not ref_boxes:
            print(f"[reattach-check] NOTE: no {os.path.basename(ref_path)} -- "
                  f"scoreboard filter only, refs NOT filtered")
        doc["frames"], n_board, n_ref = apply_junk_filter(
            doc["frames"], clip_name, ref_boxes)
        print(f"[reattach-check] junk filter ON: dropped {n_board} scoreboard-region "
              f"detections + {n_ref} referee detections")

    span_start, span_len = doc["span_start"], doc["span_len"]
    subclip, fps, n = run_tracking.extract_subclip(CLIP.video_path, span_start, span_len)

    by_frame = {fr["frame_index"]: fr["tracks"] for fr in doc["frames"]}
    by_track: dict = {}
    for fr in doc["frames"]:
        for t in fr["tracks"]:
            by_track.setdefault(t["track_id"], []).append((fr["frame_index"], t["bbox"]))

    cap = cv2.VideoCapture(subclip)
    frame_cache: dict = {}

    def get_frame(f):
        if f not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - span_start)
            ok, img = cap.read()
            frame_cache[f] = img if ok else None
        return frame_cache[f]

    # --- build the 2 color clusters from a broad sample of this clip's crops ---
    sample_sigs = []
    for fr in doc["frames"][::SAMPLE_STRIDE]:
        img = get_frame(fr["frame_index"])
        if img is None:
            continue
        for t in fr["tracks"]:
            crop = jersey_crop(img, t["bbox"])
            if crop is not None and crop.size > 0:
                sample_sigs.append(crop_color_signature(crop))
    centroids = _kmeans2(sample_sigs)
    print(f"[reattach-check] built 2 color clusters from {len(sample_sigs)} sampled crops: "
          f"{tuple(round(v) for v in centroids[0])} vs {tuple(round(v) for v in centroids[1])}")

    # --- find every reattach event: a track_id with a gap, then check colors ---
    n_events = n_suspect = n_safe = n_abstain = 0
    for tid, pts in by_track.items():
        pts.sort(key=lambda p: p[0])
        for i in range(1, len(pts)):
            gap = pts[i][0] - pts[i - 1][0]
            if gap < MIN_GAP_FRAMES + 1:
                continue
            f_before, bbox_before = pts[i - 1]
            f_after, bbox_after = pts[i]
            img_b, img_a = get_frame(f_before), get_frame(f_after)
            if img_b is None or img_a is None:
                continue
            sig_b = crop_color_signature(jersey_crop(img_b, bbox_before))
            sig_a = crop_color_signature(jersey_crop(img_a, bbox_after))
            cb, ca = _classify(sig_b, centroids), _classify(sig_a, centroids)
            n_events += 1
            if cb is None or ca is None:
                verdict, n_abstain = "ABSTAIN", n_abstain + 1
            elif cb != ca:
                verdict, n_suspect = "SUSPECT", n_suspect + 1
            else:
                verdict, n_safe = "safe", n_safe + 1
            flag = "  <<<<<" if verdict == "SUSPECT" else ""
            print(f"  id={tid}  gap {f_before}->{f_after} ({gap}f, {gap/fps:.2f}s)  "
                  f"before_cluster={cb} after_cluster={ca}  {verdict}{flag}")

    print(f"\n[reattach-check] {n_events} reattach events: "
          f"{n_suspect} SUSPECT, {n_safe} safe, {n_abstain} abstained")


if __name__ == "__main__":
    main()
