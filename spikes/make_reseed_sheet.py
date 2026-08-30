"""make_reseed_sheet.py -- name the girls the system tracked but could not read.

THE PROBLEM THIS IS FOR. A finished game leaves ~2,029 identities the pipeline
confirmed as real, continuous bodies on the floor and could not put a name to
(MEASURED on DJ's game). The review queue lists 23,288 items as a flat wall of
"window 41, track 6000273, candidate" -- true, and useless: nothing tells a coach
which of them is worth a second of his time, and a single frame number is not
enough to recognise a girl by.

WHAT THIS BUILDS INSTEAD, in one self-contained page:

  ONE CARD PER IDENTITY, not one per crop. A card carries SEVERAL pictures of
  the same girl taken across her whole time on screen, because the reason the
  machine could not read her is almost always ANGLE -- MEASURED, 30% of players
  had all ten read attempts inside a single second, so the model saw one pose
  ten times. A human given six different moments can recognise someone the model
  never could.

  RANKED BY WHAT SHE IS WORTH, in seconds of floor time. Nothing is hidden --
  every identity is on the page, in order -- but the top card is worth 97
  seconds and the median one 3.8, and a coach should be able to see that and
  decide for himself where to stop.

  A RUNNING TOTAL of how much of the tracked floor time the names so far
  actually cover, so stopping is an informed choice rather than a guess.

Output feeds straight back in as seed labels.

    .venv/Scripts/python.exe spikes/make_reseed_sheet.py <CLIP> [--from N] [--to N]
"""

from __future__ import annotations

import base64
import json
import os
import sys
from collections import defaultdict

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

SLICE_FRAMES = 17112        # a ten-way split of a 171,120-frame game
CROPS_PER_CARD = 6          # six moments is enough to recognise someone
CARD_PAD = 0.35             # show context around the torso, not a bare patch


def _events(clip, results_dir):
    p = os.path.join(results_dir, f"{clip}_player_events_merged.json")
    if not os.path.exists(p):
        p = os.path.join(_ROOT, "phase2", "out", f"{clip}_player_events.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["player_events"]


def _ocr_notes(clip, results_dir):
    """What the reader already tried on each identity, so a card can say
    'the machine had 10 goes and got nothing' instead of staying silent."""
    p = os.path.join(results_dir, f"{clip}_ocr_confirms.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        doc = json.load(fh)
    notes = {}
    for bucket, rows in (doc.get("outcomes") or {}).items():
        for r in rows:
            if isinstance(r, dict) and "window" in r:
                notes[(r["window"], r.get("identity_id"))] = bucket
    return notes


def gather(clip, tracks_path, frm=None, to=None, results_dir=None):
    """(cards, totals). A card is one identity with its best few moments."""
    results_dir = results_dir or os.path.join(_ROOT, "results", clip)
    with open(tracks_path, encoding="utf-8") as fh:
        tdoc = json.load(fh)
    boxes = {fr["frame_index"]: {t["track_id"]: t["bbox"] for t in fr["tracks"]}
             for fr in tdoc["frames"]}
    lo = min(boxes) if frm is None else frm
    hi = max(boxes) if to is None else to

    ev = _events(clip, results_dir)
    seen = defaultdict(list)                 # (win, id) -> [(frame, track_id)]
    state_of = {}
    total_frames = 0
    for e in ev:
        f = e["frame"]
        total_frames += 1
        if not (lo <= f <= hi):
            continue
        key = (e["window"], e["identity_id"])
        state_of[key] = e.get("identity_state")
        seen[key].append((f, e.get("track_id")))

    notes = _ocr_notes(clip, results_dir)
    cards = []
    for key, rows in seen.items():
        if state_of.get(key) != "confirmed":
            continue                         # candidates are a different problem
        rows.sort()
        n = len(rows)
        # crops spread across her WHOLE time, for the same reason the reader's
        # attempts are: six pictures of one second is one picture.
        step = max(1, n // CROPS_PER_CARD)
        picks = []
        for (f, tid) in rows[::step][:CROPS_PER_CARD]:
            bb = boxes.get(f, {}).get(tid)
            if bb is None and tid is not None:
                # THE MERGE RENAMES EVERY TRACK. merge_streamed offsets ids by
                # (slice + 1) * 1,000,000 so two slices' "player 4" stay
                # strangers, so an id in player_events does not exist in the
                # SLICE file it came from. Undo it for whichever slice this
                # frame belongs to; harmless if the cache is already merged.
                bb = boxes.get(f, {}).get(tid - (f // SLICE_FRAMES + 1) * 1_000_000)
            if bb:
                picks.append((f, bb))
        if not picks:
            continue
        cards.append({"window": key[0], "identity": key[1],
                      "track": rows[0][1], "frames": n,
                      "seconds": round(n / 30.0, 1),
                      "first": rows[0][0], "last": rows[-1][0],
                      "ocr": notes.get(key, "not attempted"), "picks": picks})
    cards.sort(key=lambda c: -c["frames"])
    return cards, {"events_in_range": sum(len(v) for v in seen.values()),
                   "events_whole_game": total_frames}


def _thumb(img, bb, h=118):
    x1, y1, x2, y2 = [int(v) for v in bb]
    w, ht = x2 - x1, y2 - y1
    px, py = int(w * CARD_PAD), int(ht * CARD_PAD * 0.4)
    H, W = img.shape[:2]
    a, b = max(0, x1 - px), max(0, y1 - py)
    c, d = min(W, x2 + px), min(H, y2 + py)
    crop = img[b:d, a:c]
    if crop.size == 0:
        return None
    s = h / crop.shape[0]
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), h))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 74])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def build(clip, tracks_path, frm=None, to=None, limit=400):
    import clip_registry
    import fast_frames
    doc = clip_registry.load(clip)
    cards, totals = gather(clip, tracks_path, frm, to)
    cards = cards[:limit]
    need = sorted({f for c in cards for (f, _bb) in c["picks"]})
    print(f"[reseed] {len(cards)} identities, {len(need)} frames to read", flush=True)
    frames = fast_frames.read_frames(doc["video_path"], need)
    for c in cards:
        c["shots"] = []
        for (f, bb) in c["picks"]:
            img = frames.get(f)
            if img is None:
                continue
            j = _thumb(img, bb)
            if j:
                c["shots"].append({"f": f, "t": round(f / 30.0, 1), "jpg": j})
        c.pop("picks")
    cards = [c for c in cards if c["shots"]]

    # A NUMBER IS NOT AN IDENTITY WHEN THERE ARE TWO TEAMS. Both rosters here
    # carry a #24, so keying a pick by the number alone lit up BOTH buttons and
    # would have written an ambiguous label -- the same collision color_tiebreak
    # exists to resolve. Every entry gets a team-qualified id.
    roster = []
    for ti, t in enumerate(doc.get("teams", [])):
        names = t.get("player_names") or {}
        for num in sorted(t.get("numbers", [])):
            roster.append({"id": f"{ti}:{num}", "n": num, "ti": ti,
                           "team": t.get("name", "") or f"team {ti + 1}",
                           "colour": t.get("jersey_color", ""),
                           "name": names.get(str(num), "")})
    total_secs = round(sum(c["seconds"] for c in cards), 1)
    data = {"clip": clip, "cards": cards, "roster": roster,
            "teams": [t.get("name", "") or f"team {i+1}"
                      for i, t in enumerate(doc.get("teams", []))],
            "total_seconds": total_secs,
            "tracked_seconds_whole_game": round(totals["events_whole_game"] / 30.0, 1)}
    out = os.path.join(_ROOT, "results", clip, f"{clip}_reseed_sheet.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(_PAGE.replace("__DATA__", json.dumps(data)).replace("__CLIP__", clip))
    print(f"[reseed] {len(cards)} cards, {total_secs/60:.1f} min of floor time -> {out}")
    return out


_PAGE = r"""<!doctype html><meta charset="utf-8"><title>__CLIP__ -- name the players</title>
<style>
 :root{--bg:#0e1211;--card:#161d1c;--line:#26302f;--ink:#e7edeb;--dim:#93a29f;--hot:#e8912f;--ok:#55c495}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
 header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
        padding:12px 18px;z-index:9}
 h1{margin:0 0 8px;font-size:18px}
 .bar{height:16px;background:#1d2625;border-radius:8px;overflow:hidden;border:1px solid var(--line)}
 .fill{height:100%;background:linear-gradient(90deg,var(--ok),var(--hot));width:0%;
       transition:width .25s}
 .stat{display:flex;gap:22px;flex-wrap:wrap;font-size:13px;color:var(--dim);margin-top:7px}
 .stat b{color:var(--ink)}
 main{padding:16px 18px 80px;display:flex;flex-direction:column;gap:12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;
       display:grid;grid-template-columns:190px 1fr;gap:14px}
 .card.done{border-color:var(--ok)}
 .meta{font-size:13px;color:var(--dim)}
 .meta .big{font-size:22px;color:var(--ink);font-weight:650;
            font-variant-numeric:tabular-nums}
 .meta .pct{color:var(--hot);font-weight:600}
 .shots{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-start}
 .shot{position:relative}
 .shot img{height:118px;display:block;border-radius:4px;border:1px solid #2c3635}
 .shot span{position:absolute;left:3px;bottom:3px;background:#000a;color:#fff;
            font-size:10px;padding:1px 4px;border-radius:3px}
 .nums{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}
 .team{margin-top:9px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
       color:var(--dim)}
 button.np{border-color:#5a4a3a;color:#e0c9a8}
 button{font:inherit;padding:5px 9px;border-radius:5px;border:1px solid #3a4948;
        background:#1e2827;color:var(--ink);cursor:pointer;font-size:13px}
 button:hover{border-color:var(--hot)}
 button.pick{background:var(--ok);border-color:var(--ok);color:#07130f;font-weight:700}
 button.skip{color:var(--dim)}
 #sentinel{padding:20px;text-align:center;color:var(--dim);font-size:13px}
 .dl{position:fixed;right:18px;bottom:18px;background:var(--hot);border-color:var(--hot);
     color:#111;font-weight:700;padding:11px 18px;font-size:15px;border-radius:8px}
</style>
<header>
 <h1>__CLIP__ &mdash; who is this?</h1>
 <div class="bar"><div class="fill" id="fill"></div></div>
 <div class="stat">
   <span>named <b id="ndone">0</b> of <b id="ntot">0</b></span>
   <span>marked not-a-player <b id="nonp">0</b></span>
   <span>floor time covered <b id="cov">0.0</b> min of <b id="covtot">0.0</b>
         (<b id="covpct">0%</b>)</span>
   <span>of everything the system tracked: <b id="seen">0%</b></span>
   <span id="left" class="dim"></span>
 </div>
</header>
<main id="main"></main>
<button class="dl" onclick="dl()">Download names</button>
<script>
const D = __DATA__;
const picks = {};
// NOT A PLAYER is a real answer, not a skip. DJ, using this on his own game:
// "majority of the longest unknowns are coaches and non-players" -- of course
// they are, a coach stands still on the sideline for minutes and so ranks top
// by floor time, while a player is fragmented every time she is occluded. These
// feed roster.load_ref_tracks, the path this project already uses to stop a
// referee ever becoming a player, and they are the SAME click as naming.
const NOTP = [["x:ref","referee"],["x:coach","coach / bench"],
              ["x:crowd","crowd / not on court"],["x:two","two players merged"]];
const key = c => c.window + ":" + c.identity;

const PAGE = 30;
let shown = 0;
function cardHTML(c, i){
    const share = D.total_seconds ? c.seconds / D.total_seconds : 0;
    return '<div class="card' + (picks[key(c)] ? ' done' : '') + '" id="c' + i + '">' +
      '<div class="meta">' +
        '<div class="big">' + c.seconds.toFixed(1) + 's</div>' +
        '<div class="pct">' + (share*100).toFixed(1) + '% of tracked time</div>' +
        '<div>' + c.frames + ' frames seen</div>' +
        '<div>window ' + c.window + ' &middot; ' +
             (c.first/30/60).toFixed(1) + '&ndash;' + (c.last/30/60).toFixed(1) + ' min</div>' +
        '<div style="margin-top:6px">reader: ' + c.ocr.replace(/_/g,' ') + '</div>' +
        D.teams.map((tm, ti) =>
          '<div class="team">' + tm + '</div><div class="nums">' +
          D.roster.filter(r => r.ti === ti).map(r =>
            '<button class="' + (picks[key(c)]===r.id ? 'pick':'') + '" ' +
            'onclick="pick(\'' + key(c) + '\',\'' + r.id + '\')" title="' +
            (r.name||'') + ' &mdash; ' + r.team + '">' + r.n +
            (r.name ? ' ' + r.name.slice(0,9) : '') + '</button>').join('') +
          '</div>').join('') +
        '<div class="nums" style="margin-top:8px">' +
          NOTP.map(x =>
            '<button class="np' + (picks[key(c)]===x[0] ? ' pick':'') + '" ' +
            'onclick="pick(\'' + key(c) + '\',\'' + x[0] + '\')">' + x[1] +
            '</button>').join('') +
          '<button class="skip" onclick="pick(\'' + key(c) + '\',null)">clear</button>' +
        '</div>' +
      '</div>' +
      '<div class="shots">' + c.shots.map(s =>
        '<div class="shot"><img loading="lazy" src="data:image/jpeg;base64,' + s.jpg + '">' +
        '<span>' + (s.t/60).toFixed(1) + 'm</span></div>').join('') + '</div>' +
      '</div>';
}
function more(){
  const m = document.getElementById('main');
  const upto = Math.min(shown + PAGE, D.cards.length);
  let html = '';
  for (let i = shown; i < upto; i++) html += cardHTML(D.cards[i], i);
  document.getElementById('sentinel').insertAdjacentHTML('beforebegin', html);
  shown = upto;
  document.getElementById('sentinel').textContent =
      shown < D.cards.length
        ? 'loading ' + (shown+1) + '-' + Math.min(shown+PAGE, D.cards.length) +
          ' of ' + D.cards.length + ' ...'
        : 'all ' + D.cards.length + ' shown';
}
function render(){
  document.getElementById('main').innerHTML = '<div id="sentinel"></div>';
  shown = 0;
  more();
  new IntersectionObserver(es => {
    if (es[0].isIntersecting && shown < D.cards.length) more();
  }).observe(document.getElementById('sentinel'));
  stats();
}
function pick(k, id){
  if (id === null) delete picks[k]; else picks[k] = id;
  // repaint ONLY this card. Re-rendering 2,029 cards (12,000 images) on every
  // click freezes the browser, and a naming tool that stutters is a naming tool
  // nobody finishes.
  const i = D.cards.findIndex(c => key(c) === k);
  const el = document.getElementById('c' + i);
  if (el) { el.outerHTML = cardHTML(D.cards[i], i); }
  stats();
}
function stats(){
  let done = 0, secs = 0, nonp = 0;
  D.cards.forEach(c => { const p = picks[key(c)];
    if (p && !p.startsWith('x:')) { done++; secs += c.seconds; }
    if (p && p.startsWith('x:')) nonp++; });
  const pct = D.total_seconds ? secs / D.total_seconds : 0;
  document.getElementById('ndone').textContent = done;
  document.getElementById('nonp').textContent = nonp;
  document.getElementById('ntot').textContent = D.cards.length;
  document.getElementById('cov').textContent = (secs/60).toFixed(1);
  document.getElementById('covtot').textContent = (D.total_seconds/60).toFixed(1);
  document.getElementById('covpct').textContent = (pct*100).toFixed(1) + '%';
  document.getElementById('seen').textContent =
      (D.tracked_seconds_whole_game ? secs/D.tracked_seconds_whole_game*100 : 0).toFixed(1) + '%';
  document.getElementById('fill').style.width = (pct*100) + '%';
  // what the NEXT click is worth, so stopping is an informed choice
  const next = D.cards.find(c => !picks[key(c)]);
  document.getElementById('left').textContent = next
      ? 'next card is worth ' + next.seconds.toFixed(1) + 's'
      : 'every card named';
}
function dl(){
  const out = {clip: D.clip, seed_labels: {}, ref_tracks: [], spliced_tracks: [],
               by_identity: {}};
  D.cards.forEach(c => { const p = picks[key(c)]; if (!p) return;
    if (p === 'x:two') { out.spliced_tracks.push(c.track); }
    else if (p.startsWith('x:')) { out.ref_tracks.push(c.track); }
    else { const r = D.roster.find(z => z.id === p);
           out.seed_labels[c.track] = r.n;
           out.by_identity[key(c)] = {number: r.n, team: r.team,
                                      seconds: c.seconds}; }
    out.by_identity[key(c)] = out.by_identity[key(c)] ||
        {not_a_player: p.slice(2), seconds: c.seconds}; });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  a.download = D.clip + '_names.json'; a.click();
}
render();
</script>
"""


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_reseed_sheet.py <CLIP> [tracks.json] [--limit N]")
    clip = sys.argv[1]
    tracks = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") \
        else os.path.join(_ROOT, "phase2", "out", f"{clip}_tracks_raw.json")
    limit = 400
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    build(clip, tracks, limit=limit)


if __name__ == "__main__":
    main()
