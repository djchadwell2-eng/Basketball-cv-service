"""Does a candidate tracker MERGE two people into one id? Answered with NO LABELS.

WHY THIS EXISTS. Every tracker experiment so far has been scored on
FRAGMENTATION alone -- "122 ids -> 117 ids, better". That is only half the
ledger. A tracker can always cut the id count by gluing two girls into one
track, and that failure is far worse than a fragment: a fragment costs a click,
a merge puts one player's floor time on another and nothing downstream can tell.

The plan for the player tracker called ID-switch ground truth the "HIGHEST VALUE
ITEM" and assumed it needed a human labelling session. IT DOES NOT.

THE IDEA. Both trackers run over the SAME frames from the SAME detector, so
their boxes line up almost exactly. Match candidate boxes to committed boxes by
IoU per frame. Then ask a question that needs no labels at all:

    did ONE candidate track absorb TWO committed tracks that were alive AT THE
    SAME TIME, in DIFFERENT PLACES?

Two boxes in one frame are two bodies -- that is logic, not similarity. If a
single candidate id covers both, it merged two people. No human ever has to say
who they were.

WHAT IT CANNOT DO, said plainly. The metric is ONE-SIDED: the committed tracker
is the reference, so it scores 0 by construction. It answers "is this variant
riskier than what we ship?", NOT "how good is the baseline?". Scoring the
baseline needs a neutral over-fragmented reference, and at these thresholds that
returned 0 for everyone -- it needs threshold work first. Do not read a 0 for
the committed tracker as evidence it never switches.

THE DOUBLE-DETECTION TRAP. Two committed ids sitting on ONE body (the detector
firing twice) would look exactly like a merge. They are excluded by requiring
the two committed tracks to be genuinely APART -- median IoU below
MAX_OVERLAP_IOU across at least MIN_SHARED_FRAMES shared frames. On TEST1 that
filter took a raw count of 7 down to 2.

Usage:
    .venv/Scripts/python.exe spikes/tracker_switch_metric.py COMMITTED.json CANDIDATE.json
    .venv/Scripts/python.exe spikes/tracker_switch_metric.py --clip HARD CANDIDATE.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# A candidate box must overlap a committed box this much to be "the same body".
# Same detector, same frames, so a real correspondence sits near 1.0.
MATCH_IOU = 0.5
# Two committed tracks are genuinely two BODIES only if they stay this far
# apart. Above it they are one body detected twice.
MAX_OVERLAP_IOU = 0.2
# ...measured over at least this many frames where both are alive, so a single
# ambiguous frame cannot create or hide a merge.
MIN_SHARED_FRAMES = 10


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _by_frame(doc):
    return {fr["frame_index"]: fr["tracks"] for fr in doc["frames"]}


def measure(committed, candidate):
    """-> dict of findings. Pure: takes two loaded tracker documents."""
    cm, cd = _by_frame(committed), _by_frame(candidate)
    frames = sorted(set(cm) & set(cd))

    # candidate track -> committed track -> how many frames they coincided
    absorbed = defaultdict(lambda: defaultdict(int))
    # committed pair -> IoUs on frames where BOTH are alive (the apartness test)
    pair_ious = defaultdict(list)

    for f in frames:
        c_boxes = [(t["track_id"], t["bbox"]) for t in cm[f]]
        for (cand_id, cand_bb) in [(t["track_id"], t["bbox"]) for t in cd[f]]:
            best, best_iou = None, MATCH_IOU
            for (com_id, com_bb) in c_boxes:
                v = iou(cand_bb, com_bb)
                if v > best_iou:
                    best, best_iou = com_id, v
            if best is not None:
                absorbed[cand_id][best] += 1
        # apartness, measured on the COMMITTED side only
        for i in range(len(c_boxes)):
            for j in range(i + 1, len(c_boxes)):
                a, b = c_boxes[i], c_boxes[j]
                key = (min(a[0], b[0]), max(a[0], b[0]))
                pair_ious[key].append(iou(a[1], b[1]))

    def are_two_bodies(x, y):
        """Two committed ids are two BODIES, not one detected twice?"""
        ious = pair_ious.get((min(x, y), max(x, y)))
        if not ious or len(ious) < MIN_SHARED_FRAMES:
            return False              # never co-alive long enough to judge
        ious.sort()
        return ious[len(ious) // 2] < MAX_OVERLAP_IOU

    merges = []
    for cand_id, hits in absorbed.items():
        owned = [t for t, n in hits.items() if n >= MIN_SHARED_FRAMES]
        for i in range(len(owned)):
            for j in range(i + 1, len(owned)):
                if are_two_bodies(owned[i], owned[j]):
                    merges.append({
                        "candidate_track": cand_id,
                        "committed_tracks": [owned[i], owned[j]],
                        "frames": hits[owned[i]] + hits[owned[j]],
                    })
    merged_frames = sum(m["frames"] for m in merges)

    def stats(doc):
        n = defaultdict(int)
        for fr in doc["frames"]:
            for t in fr["tracks"]:
                n[t["track_id"]] += 1
        return len(n), (sum(n.values()) / len(n) if n else 0.0)

    ci, cl = stats(committed)
    di, dl = stats(candidate)
    return {"frames_compared": len(frames),
            "committed_ids": ci, "committed_mean_life": round(cl, 1),
            "candidate_ids": di, "candidate_mean_life": round(dl, 1),
            "id_change_pct": round(100.0 * (di - ci) / ci, 1) if ci else None,
            "merges": merges, "n_merges": len(merges),
            "merged_frames": merged_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--clip", help="use this clip's committed cache as the reference")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.clip:
        import clip_config
        cfg = clip_config.get_clip(args.clip)
        committed_path, candidate_path = cfg.tracks_cache_path, args.paths[0]
    else:
        committed_path, candidate_path = args.paths[0], args.paths[1]

    committed = json.load(open(committed_path, encoding="utf-8"))
    candidate = json.load(open(candidate_path, encoding="utf-8"))
    r = measure(committed, candidate)

    print(f"committed : {os.path.basename(committed_path)}")
    print(f"candidate : {os.path.basename(candidate_path)}")
    print(f"frames compared: {r['frames_compared']}\n")
    print(f"  {'':<12}{'ids':>6}{'mean life':>12}")
    print(f"  {'committed':<12}{r['committed_ids']:>6}{r['committed_mean_life']:>12}")
    print(f"  {'candidate':<12}{r['candidate_ids']:>6}{r['candidate_mean_life']:>12}"
          f"   ({r['id_change_pct']:+}% ids)")
    print(f"\n  MERGES (one candidate id covering two co-alive, separated bodies): "
          f"{r['n_merges']}")
    for m in r["merges"]:
        print(f"     candidate t{m['candidate_track']} absorbed committed "
              f"{m['committed_tracks']}  over {m['frames']} frames")
    print(f"  merged frames: {r['merged_frames']} "
          f"({r['merged_frames'] / args.fps:.1f} s of film)")
    print("\n  NOTE: one-sided. The committed tracker is the reference and scores 0 "
          "by construction;\n  this answers 'is the candidate riskier than what we "
          "ship', not 'how good is the baseline'.")


if __name__ == "__main__":
    main()
