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
