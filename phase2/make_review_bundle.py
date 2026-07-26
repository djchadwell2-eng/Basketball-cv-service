"""Review bundle v1 -- the coach's click-seeding page (no server, one file).

Generates phase2/out/{clip}_review.html: every DISTINCT track that was seeded
on-court at any window start, sorted by floor time, each with its 3 largest
jersey crops (the same torso patch OCR reads) and one button per roster number.
The human labels what they can and clicks Download -- the resulting
{clip}_decisions.json goes into phase2/out/, where roster.seed_number_for()
merges it on the next run. A human click is a seed: same gate, same rules.

v1 scope: TRACK labeling only. Mid-window candidate resolution (accept/reject
queue items) is v2 -- it needs a post-hoc human-confirm mechanism.
"""

from __future__ import annotations

import base64
import html
import json
import os
import sys
from collections import defaultdict

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "phase1"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

import oncourt
import purity
from clip_config import ACTIVE_CLIP as CLIP
from ocr_reader import jersey_crop

OUT_HTML = os.path.join(_HERE, "out", f"{CLIP.name}_review.html")
# 6, not 3: a coach cannot see an ID SWAP from three crops. Spread across the
# whole track life, a swap shows up as the body simply changing partway along
# the strip -- which is how DJ caught two of them on TEST2 by eye, and neither
# purity.py nor identity_report.py can detect that class at all.
CROPS_PER_TRACK = 6
MIN_CROP_BOX_H = 60          # smaller than OCR's 90: humans read more than OCR
CROP_SPACING = 5             # frames apart, so the crops aren't near-duplicates
FULL_H = 190                 # whole player -- a coach may recognise the athlete
NUM_H = 150                  # jersey patch, blown up bigger than the body shot
                             # so the number itself is verifiable


def _frames_at(video, need):
    need = sorted(set(need))
    cap = cv2.VideoCapture(video)
    out, it = {}, iter(need)
    nxt, idx = next(it, None), 0
    while nxt is not None and cap.grab():
        if idx == nxt:
            ok, fr = cap.retrieve()
            if ok:
                out[idx] = fr
            nxt = next(it, None)
        idx += 1
    cap.release()
    return out


def _pair(frame, bbox, f):
    """One review cell: the WHOLE player beside a blown-up crop of just the
    number. DJ's ask -- the full body so a coach can recognise the athlete, the
    zoom so they can verify the number rather than trust it. Frame index shown
    so anything suspicious can be found in the footage."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = frame.shape[:2]
    body = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    num = jersey_crop(frame, bbox)
    return (f'<figure class="cell">'
            f'<img class="body" src="data:image/jpeg;base64,'
            f'{_b64_jpg(body, FULL_H)}" alt="whole player at frame {f}">'
            f'<img class="num" src="data:image/jpeg;base64,'
            f'{_b64_jpg(num, NUM_H)}" alt="jersey number at frame {f}">'
            f'<figcaption>f{f}</figcaption></figure>')


def _b64_jpg(img, height=110):
    s = height / img.shape[0]
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), height),
                     interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def main():
    tdoc = json.load(open(CLIP.tracks_cache_path, encoding="utf-8"))
    odoc = oncourt.load_checked(CLIP)
    fps = tdoc["fps"]
    span_start = tdoc["span_start"]
    import possessions
    boundaries, _wlabel = possessions.load_windows(CLIP, verbose=False)
    on_by_frame = {fr["frame_index"]: {int(t) for t, i in fr["tracks"].items() if i["on"]}
                   for fr in odoc["frames"]}

    # distinct tracks seeded on-court at any window start (shared helper, so
    # the bundle and the purity check agree on the labelable set)
    seed_tracks = oncourt.seed_frame_tracks(tdoc, odoc, boundaries=boundaries)
    pur = purity.load(CLIP)      # {} until cache_purity has been run

    # --- queue-resolution inputs (post-run artifacts; section skipped if absent)
    def _load(name):
        p = os.path.join(_HERE, "out", f"{CLIP.name}_{name}.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    merged = _load("player_events_merged")
    queue_doc = _load("review_queue")
    ocr_doc = _load("ocr_confirms")
    existing = _load("decisions") or {}
    bbox_at = {(fr["frame_index"], t["track_id"]): t["bbox"]
               for fr in tdoc["frames"] for t in fr["tracks"]}

    queue_entries = []
    if merged and queue_doc and ocr_doc and "identities" in ocr_doc:
        reg = {(r["window"], r["track_id"]): r for r in ocr_doc["identities"]}
        by_ident = defaultdict(list)
        for e in merged["player_events"]:
            by_ident[(e["window"], e["identity_id"])].append(
                (e["frame"], e["track_id"], e["identity_state"]))
        seen_ids = set()
        for q in queue_doc["items"]:
            r = reg.get((q["window"], q["track"]))
            if r is None or r["final_state"] not in ("candidate", "unknown"):
                continue                     # OCR already confirmed it, or unmapped
            key = (q["window"], r["identity_id"])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            evs = sorted(by_ident.get(key, []))
            queue_entries.append({"window": key[0], "identity_id": key[1],
                                  "state": q["state"], "why": q["why"],
                                  "hyp": r.get("roster_number"),
                                  "events": [(f, t) for (f, t, _s) in evs]})

    # floor time + candidate crops per track (largest boxes, spaced apart)
    floor: dict[int, int] = defaultdict(int)
    boxes: dict[int, list] = defaultdict(list)          # tid -> [(h, f, bbox)]
    for fr in tdoc["frames"]:
        f = fr["frame_index"]
        for t in fr["tracks"]:
            tid = t["track_id"]
            if tid not in seed_tracks:
                continue
            if tid in on_by_frame.get(f, set()):
                floor[tid] += 1
            h = t["bbox"][3] - t["bbox"][1]
            if h >= MIN_CROP_BOX_H:
                boxes[tid].append((h, f, t["bbox"]))

    # SPREAD crops across the track's life (early/mid/late, biggest of each
    # third) -- so a mid-track jersey change (splice) is always visible to a
    # human, not only catchable by luck.
    picks: dict[int, list] = {}
    need = set()
    for tid, bs in boxes.items():                       # bs is in frame order
        k = max(1, len(bs) // CROPS_PER_TRACK)
        sel = []
        for i in range(CROPS_PER_TRACK):
            part = bs[i * k:(i + 1) * k] if i < CROPS_PER_TRACK - 1 else bs[i * k:]
            if not part:
                continue
            h, f, bb = max(part, key=lambda b: b[0])
            if all(abs(f - g) >= CROP_SPACING for (_h2, g, _b2) in sel):
                sel.append((h, f, bb))
        picks[tid] = sel
        need |= {f for (_h, f, _b) in sel}

    # queue-entry crops: early/mid/late of the identity's WHOLE span (the
    # reviewer's click vouches for everything that gets credited)
    qpicks = {}
    for q in queue_entries:
        rows_q = [(f, t, bbox_at.get((f, t))) for (f, t) in q["events"]]
        rows_q = [(f, t, bb) for (f, t, bb) in rows_q
                  if bb and bb[3] - bb[1] >= MIN_CROP_BOX_H]
        k = max(1, len(rows_q) // CROPS_PER_TRACK)
        sel = []
        for i in range(CROPS_PER_TRACK):
            part = (rows_q[i * k:(i + 1) * k]
                    if i < CROPS_PER_TRACK - 1 else rows_q[i * k:])
            if not part:
                continue
            f, t, bb = max(part, key=lambda r: r[2][3] - r[2][1])
            if all(abs(f - g) >= CROP_SPACING for (g, _b2) in sel):
                sel.append((f, bb))
        qpicks[(q["window"], q["identity_id"])] = sel
        need |= {f for (f, _b) in sel}
    imgs = _frames_at(CLIP.video_path, need)

    # --- build rows, longest floor time first --------------------------------
    order = sorted(seed_tracks, key=lambda t: -floor[t])
    team_btns = ""
    for team in CLIP.teams:
        nums = "".join(f'<button data-n="{n}">{n}</button>'
                       for n in sorted(team.numbers))
        team_btns += (f'<span class="team">{html.escape(team.name)}:</span>'
                      f'{nums}')
    # 'bench' is deliberately its own button, not folded into 'not a player':
    # a ref is correctly on-court and simply is not a player, whereas a seated
    # sub being on-court at all is an on-court-classifier miss. Counting them
    # apart turns the second one into a bug signal instead of a chore.
    team_btns += ('<button data-n="ref" class="alt">REF / official</button>'
                  '<button data-n="bench" class="alt">ON THE BENCH / not in play</button>'
                  '<button data-n="" class="alt">unsure</button>')

    rows = []
    n_quarantined = 0
    for tid in order:
        crops = "".join(_pair(imgs[f], bb, f)
                        for (h, f, bb) in sorted(picks.get(tid, []),
                                                 key=lambda r: r[1])
                        if f in imgs)
        wins = ",".join(str(w) for w in sorted(seed_tracks[tid]))
        pv = pur.get(tid)
        if pv and pv["verdict"] == "spliced":
            n_quarantined += 1
            nums = ", ".join(f"#{n}" for n in sorted(pv["numbers"]))
            controls = (f'<div class="spliced">&#9888; SPLICED — the computer '
                        f'read TWO different numbers on this track ({nums}): '
                        f'it jumped between players mid-play. Quarantined; '
                        f'not labelable.</div>')
        else:
            controls = f'<div class="btns">{team_btns}</div>'
        rows.append(
            f'<div class="row" data-tid="{tid}">'
            f'<div class="crops">{crops or "<i>no crop &ge; 60px</i>"}</div>'
            f'<div class="meta">t{tid} &middot; {floor[tid] / fps:.1f}s on floor '
            f'&middot; seed windows {wins}</div>'
            f'{controls}</div>')

    # --- queue-resolution rows (Part 2) ---------------------------------------
    qbtns = ""
    for team in CLIP.teams:
        nums = "".join(f'<button data-n="{n}">{n}</button>'
                       for n in sorted(team.numbers))
        qbtns += (f'<span class="team">{html.escape(team.name)}:</span>{nums}')
    qbtns += ('<button data-n="reject" class="alt">reject — not a player</button>'
              '<button data-n="" class="alt">unsure</button>')

    qrows = []
    for q in queue_entries:
        key = (q["window"], q["identity_id"])
        crops = "".join(
            f'<img src="data:image/jpeg;base64,{_b64_jpg(jersey_crop(imgs[f], bb))}" '
            f'title="frame {f}">'
            for (f, bb) in qpicks.get(key, []) if f in imgs)
        hyp = (f' &middot; position guess: #{q["hyp"]}'
               if q["hyp"] is not None else "")
        qrows.append(
            f'<div class="row qrow" data-win="{q["window"]}" data-id="{q["identity_id"]}">'
            f'<div class="crops">{crops or "<i>no crop &ge; 60px</i>"}</div>'
            f'<div class="meta">possession {q["window"]} &middot; {q["state"]} '
            f'&middot; {len(q["events"]) / fps:.1f}s of unclaimed time{hyp}<br>'
            f'<small>{html.escape(q["why"])}</small></div>'
            f'<div class="btns">{qbtns}</div></div>')

    preset_tracks = json.dumps(existing.get("track_labels", {}))
    # Queue resolutions are only valid against the boundaries they were made
    # for (identity ids shift when windows change) -- carry them through only
    # if the boundaries match; stale ones would pre-select the WRONG rows.
    if existing.get("window_boundaries") == boundaries:
        preset_q = json.dumps({f"{r['window']}|{r['identity_id']}": r["number"]
                               for r in existing.get("queue_resolutions", [])})
    else:
        preset_q = "{}"

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{CLIP.name} — label the players</title><style>
body{{background:#181818;color:#eee;font:14px/1.5 system-ui,sans-serif;margin:20px}}
h1{{font-size:20px}} .row{{border-bottom:1px solid #333;padding:12px 0}}
.crops{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}}
.cell{{margin:0;text-align:center}}
.cell img{{display:block;border:1px solid #444;background:#000}}
.cell .body{{height:190px}}
.cell .num{{height:150px;margin-top:4px;border-color:#c86}}
.cell figcaption{{font:11px monospace;color:#888;margin-top:2px}}
.meta{{color:#9c9;margin:4px 0}} .team{{color:#999;margin:0 6px 0 14px}}
button{{margin:2px;padding:4px 10px;background:#2a2a2a;color:#eee;
border:1px solid #555;border-radius:4px;cursor:pointer}}
button.sel{{background:#2e7d32;border-color:#7c4}} button.alt{{color:#caa}}
.spliced{{color:#f77;border:1px solid #a33;padding:6px 10px;border-radius:4px;
display:inline-block;margin-top:4px}}
.qrow .meta{{color:#fc7}}
#bar{{position:sticky;top:0;background:#181818;padding:10px 0;border-bottom:2px solid #444}}
#dl{{background:#1565c0;font-size:15px;padding:8px 16px}}
</style></head><body>
<div id="bar"><button id="dl">Download {CLIP.name}_decisions.json</button>
<span id="ct">0 labeled</span></div>
<h1>Part 1 — {CLIP.name}: click the jersey number for each player</h1>
<p><b>Each cell is the same body at one moment: whole player on top, the
number blown up underneath.</b> The cells run left to right in time, so if the
tracker jumped from one girl to another partway through, you will see the body
change along the strip -- if it does, leave it blank and tell me the track
number. Label what YOU are sure of;
leave the rest unsure — an unsure player stays honestly unnamed (never guessed).
Your earlier answers are pre-selected. When done: Download, move the file into
<b>phase2/out/</b>, and tell Claude to re-run.</p>
{"".join(rows)}
<h1>Part 2 — the review queue: unclaimed time, who is it?</h1>
<p>Each row below is floor time the system REFUSED to guess about (a player it
lost and re-found, or never identified). The photos span that whole stretch of
time — your click credits ALL of it to that player, so only answer when the
photos clearly show one player. Wrong-looking or non-player rows: reject.</p>
{"".join(qrows) or "<p><i>(no unresolved queue items — nothing to do here)</i></p>"}
<script>
const presetTracks = {preset_tracks};
const presetQ = {preset_q};
// Start from the presets so labels/resolutions for tracks NOT shown on this
// page survive a re-download (the shown-track set changes across reruns).
const dec = Object.assign({{}}, presetTracks);
const qdec = Object.assign({{}}, presetQ);
const ct = document.getElementById('ct');
function refresh() {{
  ct.textContent = (Object.values(dec).filter(x => x !== null).length +
                    Object.values(qdec).filter(x => x !== null).length) + ' labeled';
}}
document.querySelectorAll('.row').forEach(row => {{
  const isQ = row.classList.contains('qrow');
  row.querySelectorAll('button[data-n]').forEach(b => b.onclick = () => {{
    const v = b.dataset.n;
    const val = v === '' ? null : (/^[0-9]+$/.test(v) ? parseInt(v) : v);
    if (isQ) qdec[row.dataset.win + '|' + row.dataset.id] = val;
    else dec[row.dataset.tid] = val;
    row.querySelectorAll('button').forEach(x => x.classList.remove('sel'));
    b.classList.add('sel');
    refresh();
  }});
}});
document.querySelectorAll('.row').forEach(row => {{
  const isQ = row.classList.contains('qrow');
  const k = isQ ? row.dataset.win + '|' + row.dataset.id : row.dataset.tid;
  const pv = isQ ? presetQ[k] : presetTracks[k];
  if (pv === undefined) return;
  const want = pv === null ? '' : String(pv);
  row.querySelectorAll('button[data-n]').forEach(b => {{
    if (b.dataset.n === want) b.click();
  }});
}});
document.getElementById('dl').onclick = () => {{
  const qres = Object.entries(qdec).filter(([k, v]) => v !== null).map(([k, v]) => {{
    const p = k.split('|');
    return {{window: parseInt(p[0]), identity_id: parseInt(p[1]), number: v}};
  }});
  const doc = {{clip: '{CLIP.name}', track_labels: dec,
               queue_resolutions: qres,
               window_boundaries: {json.dumps(boundaries)}}};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 2)],
                                        {{type: 'application/json'}}));
  a.download = '{CLIP.name}_decisions.json';
  a.click();
}};
</script></body></html>"""

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[review bundle] {len(order)} tracks + {len(queue_entries)} queue "
          f"items ({sum(len(v) for v in picks.values()) + sum(len(v) for v in qpicks.values())} "
          f"crops; {n_quarantined} quarantined as spliced) -> {OUT_HTML}")
    print(f"  top floor times: " + ", ".join(
        f"t{t}={floor[t] / fps:.1f}s" for t in order[:6]))


if __name__ == "__main__":
    main()
