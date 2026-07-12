"""STAGE 7 -- retroactive stat merge (the payoff of a second-signal confirm).

When OCR AGREES with a candidate's position hypothesis, that identity is the
player -- and its provisional (candidate-stamped) span can now be credited.
This stage re-stamps those events `confirmed_retroactive` (kept distinguishable
from live `confirmed` forever) and writes a NEW artifact; stage5's raw
player_events file is never mutated.

SAFETY SHAPE (the reason this file is small and boring):
  * The ONLY trigger is an AGREE outcome -- a record that exists solely because
    promote_via_second_signal() routed through the set_confirmed gate. This
    stage consumes those records and nothing else: it takes NO position input,
    so merging on reappearance/continuity is impossible by construction.
  * Only candidate events restamp. LOST gaps attributed nothing and stay that
    way (we never invent presence). Unknown events never restamp.
  * CONTRADICTION check: if the same jersey number already has confirmed (live
    or retro) presence on overlapping frames in the same window, one player
    would be in two places -- that is evidence of an upstream error, so the
    merge is REFUSED and flagged loudly for review instead.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)

from clip_config import ACTIVE_CLIP as CLIP

EVENTS_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_player_events.json")
OCR_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_ocr_confirms.json")
OUT_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_player_events_merged.json")


def merge_events(events_doc: dict, ocr_doc: dict,
                 human_resolutions: list | None = None) -> dict:
    """Pure merge: (stage5 events, stage6 outcomes, human queue resolutions)
    -> NEW merged document. Triggers are ONLY (a) gate-emitted OCR agree
    records and (b) explicit human resolutions -- both pass the same
    contradiction check; neither is ever position-derived."""
    events = [dict(e) for e in events_doc["player_events"]]     # never mutate input

    by_ident: dict = defaultdict(list)
    for e in events:
        by_ident[(e["window"], e["identity_id"])].append(e)

    # Frames already credited to a number per window (live confirms first;
    # grows as merges land, so later merges collide against earlier ones too).
    numbers = {(r["window"], r["identity_id"]): r["roster_number"]
               for r in ocr_doc.get("identities", [])}
    credited: dict = defaultdict(set)
    for e in events:
        if e["identity_state"] == "confirmed":
            n = numbers.get((e["window"], e["identity_id"]))
            if n is not None:
                credited[(e["window"], n)].add(e["frame"])

    merges, contradictions, conflicts, rejected = [], [], [], []
    merged_number: dict = {}                     # key -> number already credited

    def _restamp(key, n, states, source, extra):
        """Shared restamp path for OCR and human triggers: same contradiction
        check, same credited-set bookkeeping. Returns True if merged."""
        rows_x = by_ident.get(key)
        if not rows_x:
            raise SystemExit(
                f"MERGE ABORTED: {source} trigger references identity {key} with "
                f"no player_events -- artifacts are stale or replays diverged. "
                f"Regenerate from the same caches/windows before merging.")
        take = [e for e in rows_x if e["identity_state"] in states]
        frames = {e["frame"] for e in take}
        clash = frames & credited[(key[0], n)]
        if clash:
            contradictions.append({
                "window": key[0], "identity_id": key[1], "number": n,
                "source": source,
                "n_overlap": len(clash), "overlap_frames": sorted(clash)[:10],
                **extra,
                "why": "same number already has confirmed presence on these "
                       "frames -- one player cannot be in two places; upstream "
                       "seed/relink/label/OCR error suspected. NOT merged; review."})
            return False
        for e in take:
            e["identity_state"] = "confirmed_retroactive"
            e["merge"] = {"number": n, "source": source, **extra}
        credited[(key[0], n)] |= frames
        merged_number[key] = n
        merges.append({"window": key[0], "identity_id": key[1], "number": n,
                       "source": source, "frames_restamped": len(frames)})
        return True

    for row in ocr_doc["outcomes"]["agree"]:
        _restamp((row["window"], row["identity_id"]), row["read_number"],
                 ("candidate",), "ocr",
                 {"read_confidence": row.get("read_confidence"),
                  "read_frame": row.get("read_frame")})

    # Human queue resolutions: the reviewer saw the identity's WHOLE span, so
    # both its candidate and its unknown frames are vouched. Same check.
    for res in (human_resolutions or []):
        key = (res["window"], res["identity_id"])
        n = res["number"]
        if n == "reject":                        # crowd / not-a-player: recorded,
            rejected.append({"window": key[0],   # never credited
                             "identity_id": key[1]})
            continue
        if key in merged_number and merged_number[key] != n:
            conflicts.append({"window": key[0], "identity_id": key[1],
                              "ocr_number": merged_number[key],
                              "human_number": n,
                              "why": "OCR and the human reviewer disagree -- "
                                     "NEVER silently resolved; re-review."})
            continue
        if key in merged_number:                 # same number: already credited
            continue
        _restamp(key, n, ("candidate", "unknown"), "human", {})

    counts: dict = defaultdict(int)
    for e in events:
        counts[e["identity_state"]] += 1
    box: dict = defaultdict(lambda: {"confirmed": 0, "confirmed_retroactive": 0})
    for e in events:
        if e["identity_state"] in ("confirmed", "confirmed_retroactive"):
            box[(e["window"], e["identity_id"])][e["identity_state"]] += 1

    return {"clip": events_doc.get("clip"),
            "player_events": events,
            "merges": merges,
            "contradictions": contradictions,
            "conflicts": conflicts,
            "rejected_by_review": rejected,
            "state_counts": dict(counts),
            "box_score_with_retro": [
                {"window": w, "identity_id": i, **c}
                for (w, i), c in sorted(box.items())]}


DECISIONS_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_decisions.json")


def load_queue_resolutions(path, roster_numbers, current_boundaries):
    """Human queue resolutions from the decisions file. STALE resolutions
    (made against different window boundaries -> identity ids shifted) are
    REFUSED loudly; off-roster numbers are refused per item."""
    if not os.path.exists(path):
        return []
    dec = json.load(open(path, encoding="utf-8"))
    items = dec.get("queue_resolutions", [])
    if not items:
        return []
    made_against = dec.get("window_boundaries")
    if (made_against is not None and current_boundaries is not None
            and list(made_against) != list(current_boundaries)):
        raise SystemExit(
            "STALE QUEUE RESOLUTIONS -- refusing to apply.\n"
            f"  resolutions were made against windows {made_against}\n"
            f"  the current windows are              {current_boundaries}\n"
            "Identity ids shift with windows; regenerate the review bundle "
            "and re-resolve.")
    out = []
    for r in items:
        if r["number"] != "reject" and r["number"] not in roster_numbers:
            print(f"  WARNING: queue resolution w{r['window']} "
                  f"id{r['identity_id']} -> {r['number']!r} is OFF-ROSTER -- "
                  f"REFUSED (fix the label or the roster)")
            continue
        out.append(r)
    if out:
        print(f"  queue resolutions: {len(out)} human decision(s) loaded from "
              f"{os.path.basename(path)}")
    return out


def main():
    events_doc = json.load(open(EVENTS_JSON, encoding="utf-8"))
    ocr_doc = json.load(open(OCR_JSON, encoding="utf-8"))
    if "identities" not in ocr_doc:
        raise SystemExit(f"{OCR_JSON} has no identities registry -- re-run "
                         f"stage6 (older format) before merging.")
    human = load_queue_resolutions(DECISIONS_JSON, CLIP.roster_numbers(),
                                   ocr_doc.get("window_boundaries"))
    doc = merge_events(events_doc, ocr_doc, human_resolutions=human)

    print(f"clip={doc['clip']}  events={len(doc['player_events'])}")
    print(f"\nMERGES applied (OCR-agree + human-resolution ONLY): "
          f"{len(doc['merges'])}")
    for m in doc["merges"]:
        print(f"  window {m['window']} identity {m['identity_id']}: "
              f"#{m['number']} -> {m['frames_restamped']} frames re-credited "
              f"as confirmed_retroactive  [{m['source']}]")
    print(f"\nCONTRADICTIONS (refused merges, flagged for review): "
          f"{len(doc['contradictions'])}")
    for c in doc["contradictions"]:
        print(f"  window {c['window']} identity {c['identity_id']} #{c['number']} "
              f"[{c.get('source', 'ocr')}]: {c['n_overlap']} overlapping frames "
              f"-- {c['why']}")
    if doc["conflicts"]:
        print(f"\nOCR-vs-HUMAN CONFLICTS (never silently resolved): "
              f"{len(doc['conflicts'])}")
        for c in doc["conflicts"]:
            print(f"  window {c['window']} identity {c['identity_id']}: OCR read "
                  f"#{c['ocr_number']}, reviewer said #{c['human_number']} -> re-review")
    if doc["rejected_by_review"]:
        print(f"\nrejected by reviewer (not players, never credited): "
              f"{len(doc['rejected_by_review'])}")
    print(f"\nevent states after merge: {doc['state_counts']}")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"\nsaved merged player_events -> {OUT_JSON}")


if __name__ == "__main__":
    main()
