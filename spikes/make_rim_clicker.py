"""Build a browser page for marking the TWO RIMS of a game -- the last human
input the shot layer needs.

WHY THIS EXISTS. The shot layer cannot run without ClipConfig.hoop_anchors: two
rim pixels, each tied to a calibration keyframe, which hoop_anchor.py then
carries through the whole camera pan. Every clip that has them got them typed
into clip_config.py by hand. A game set up in the browser has no route to them
at all -- clip_registry stores the field, nothing writes it -- so a coach's
uploaded game can never have shots, and nothing says so out loud.

NO CALIBRATION LANDMARK CAN SUBSTITUTE. Every landmark in the court model is a
point ON THE FLOOR, and a floor homography only holds on the floor plane. The
rim is ten feet up. It has to be pointed at.

Same shape as make_landmark_clicker.py, for the same reason: this venv has
opencv-python-headless shadowing the GUI build, so cv2 click windows do not
open. Frames are embedded as base64 JPEGs, so the page is one file with no
server and no external requests.

Usage:
    .venv/Scripts/python.exe spikes/make_rim_clicker.py <CLIP>
    -> spikes/out/<CLIP>_rim_clicker.html
       click, Download, then:
    .venv/Scripts/python.exe spikes/make_rim_clicker.py <CLIP> --save <json>
"""

from __future__ import annotations

import base64
import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


def _frames(doc):
    """The clip's calibration keyframes, full resolution.

    Deliberately ALL of them: the camera pans, so one basket is often out of
    shot in any given keyframe, and the rim must be marked on a frame where it
    is actually visible. Which keyframe carries which rim is the coach's call,
    not a guess made here -- hoop_anchor takes the keyframe as part of the
    anchor precisely so either end can come from wherever it looks clearest.
    """
    kfs = doc.get("keyframes") or []
    if not kfs:
        raise SystemExit(f"{doc.get('name')}: no keyframes -- calibrate the clip first")
    video = doc["video_path"]
    # PLUS A SWEEP ACROSS THE GAME, because the keyframes are not enough.
    # Calibration lands wherever the court marks were clearest, which is no
    # guarantee either basket is in shot: on DJ's own game the RIGHT-hand basket
    # appears in none of the five keyframes, so the shot layer was unreachable.
    # hoop_anchor now carries a rim from any frame (SIFT-matched to its nearest
    # keyframe, same maths it runs per frame anyway), so the page can offer the
    # whole game and let the coach pick a view where the rim is actually there.
    # Keyframes stay FIRST -- they need no match and are the surest option.
    _cap = cv2.VideoCapture(video)
    _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    _cap.release()
    sweep = list(range(0, _total, max(1, _total // 40))) if _total else []
    kfs = list(dict.fromkeys(list(kfs) + sweep))
    if not os.path.exists(video):
        raise SystemExit(f"video not found: {video}")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for f in kfs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, img = cap.read()
        if not ok:
            print(f"  WARNING keyframe {f} unreadable -- skipped")
            continue
        if landed != int(f):
            # Same rule as fast_frames: never hand over a frame we cannot prove
            # is the frame. A rim clicked on the wrong picture is a silently
            # wrong rim for the whole game.
            print(f"  WARNING keyframe {f}: seek landed on {landed} -- skipped")
            continue
        ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok2:
            continue
        out.append({"frame": int(f), "t": round(int(f) / fps, 1),
                    "w": img.shape[1], "h": img.shape[0],
                    "jpg": base64.b64encode(buf.tobytes()).decode("ascii")})
    cap.release()
    if not out:
        raise SystemExit("no keyframes could be read")
    return out


_PAGE = """<!doctype html><meta charset="utf-8"><title>__CLIP__ -- mark the rims</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#111413;color:#e8edea}
 header{padding:14px 18px;border-bottom:1px solid #2b3634}
 h1{margin:0 0 4px;font-size:19px}
 p{margin:0;color:#93a29f}
 .bar{display:flex;gap:8px;padding:12px 18px;align-items:center;flex-wrap:wrap}
 button{font:inherit;padding:7px 13px;border-radius:5px;border:1px solid #3a4948;
        background:#1b2423;color:#e8edea;cursor:pointer}
 button.on{background:#e8912f;border-color:#e8912f;color:#111413;font-weight:600}
 button:disabled{opacity:.4;cursor:default}
 #wrap{position:relative;display:inline-block;margin:0 18px 18px}
 img{max-width:calc(100vw - 40px);display:block;cursor:crosshair}
 canvas{position:absolute;left:0;top:0;pointer-events:none}
 .done{color:#55c495}.todo{color:#e8746a}
 code{background:#1b2423;padding:1px 5px;border-radius:3px}
</style>
<header>
 <h1>__CLIP__ &mdash; mark the two rims</h1>
 <p>Pick the basket you are marking, scrub to ANY frame where you can see it, and
    click the <b>centre of the hoop</b> (the ring itself, not the backboard or the net).
    Click again to move it. Then press Download.</p>
</header>
<div class="bar">
 <b>Basket:</b>
 <button id="bnear" class="on" onclick="pick('near')">NEAR / left</button>
 <button id="bfar" onclick="pick('far')">FAR / right</button>
 <span style="width:16px"></span>
 <b>Keyframe:</b><span id="kfs"></span>
</div>
<div class="bar">
 <span id="status"></span>
 <span style="flex:1"></span>
 <button onclick="clr()">Clear this basket</button>
 <button id="dl" onclick="dl()" disabled>Download</button>
</div>
<div id="jsonwrap" style="display:none;padding:0 16px 10px">
 <p style="margin:0 0 5px">Both rims marked. Download above, or just COPY this and
  paste it back &mdash; copying cannot be blocked by the browser.</p>
 <textarea id="json" rows="8" readonly style="width:100%;box-sizing:border-box;
  font:12px ui-monospace,monospace;background:#0f1413;color:#cfe3df;
  border:1px solid #3a4948;border-radius:5px;padding:8px"></textarea>
 <button onclick="copyjson()">Copy</button><span id="copied" style="color:#55c495"></span>
</div>
<div id="wrap"><img id="img"><canvas id="cv"></canvas></div>
<script>
const D = __DATA__;
let si = 0, side = 'near';
const marks = {};                       // side -> {frame, x, y}
const img = document.getElementById('img'), cv = document.getElementById('cv');
const shot = () => D.shots[si];

function kfbar(){
  const el = document.getElementById('kfs'); el.innerHTML = '';
  D.shots.forEach((s, i) => {
    const b = document.createElement('button');
    b.textContent = s.frame + '  (' + s.t + 's)';
    if (i === si) b.className = 'on';
    b.onclick = () => { si = i; load(); };
    el.appendChild(b);
  });
}
function load(){ img.src = 'data:image/jpeg;base64,' + shot().jpg; kfbar(); }
img.onload = () => { cv.width = img.clientWidth; cv.height = img.clientHeight; draw(); };
window.addEventListener('resize', () => { cv.width = img.clientWidth;
                                          cv.height = img.clientHeight; draw(); });

img.onclick = e => {
  const r = img.getBoundingClientRect(), sc = shot().w / r.width;
  marks[side] = {frame: shot().frame,
                 x: +((e.clientX - r.left) * sc).toFixed(1),
                 y: +((e.clientY - r.top) * sc).toFixed(1)};
  draw();
};
function draw(){
  const g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  const r = img.getBoundingClientRect(), sc = r.width / shot().w;
  for (const s of ['near','far']){
    const m = marks[s];
    if (!m || m.frame !== shot().frame) continue;
    const x = m.x * sc, y = m.y * sc;
    g.strokeStyle = s === side ? '#e8912f' : '#55c495';
    g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, 13, 0, 6.284); g.stroke();
    g.beginPath(); g.moveTo(x-20,y); g.lineTo(x+20,y);
    g.moveTo(x,y-20); g.lineTo(x,y+20); g.stroke();
    g.fillStyle = g.strokeStyle; g.font = 'bold 13px system-ui';
    g.fillText(s.toUpperCase(), x + 17, y - 15);
  }
  status();
}
function status(){
  const bits = ['near','far'].map(s => marks[s]
      ? `<span class="done">&#10003; ${s.toUpperCase()} rim on keyframe ${marks[s].frame}
         at ${marks[s].x}, ${marks[s].y}</span>`
      : `<span class="todo">&#10007; ${s.toUpperCase()} rim not marked</span>`);
  document.getElementById('status').innerHTML = bits.join(' &nbsp;&middot;&nbsp; ');
  document.getElementById('dl').disabled = !(marks.near && marks.far);
  showjson();
}
function pick(s){
  side = s;
  document.getElementById('bnear').className = s === 'near' ? 'on' : '';
  document.getElementById('bfar').className  = s === 'far'  ? 'on' : '';
  // jump to the keyframe this basket is already marked on, so a second click
  // corrects the existing mark instead of quietly making a different one
  if (marks[s]) { const i = D.shots.findIndex(x => x.frame === marks[s].frame);
                  if (i >= 0 && i !== si) { si = i; load(); return; } }
  draw();
}
function clr(){ delete marks[side]; draw(); }
function payload(){
  return JSON.stringify({clip: D.clip,
      hoop_anchors: {near: [marks.near.frame, [marks.near.x, marks.near.y]],
                     far:  [marks.far.frame,  [marks.far.x,  marks.far.y]]}}, null, 2);
}
function dl(){
  // APPENDED TO THE DOCUMENT, then removed. A detached anchor's click() is
  // ignored by some browsers, which looks exactly like a broken button.
  const a = document.createElement('a');
  const url = URL.createObjectURL(new Blob([payload()], {type: 'application/json'}));
  a.href = url; a.download = D.clip + '_rims.json';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function copyjson(){
  const t = document.getElementById('json');
  t.select(); t.setSelectionRange(0, 99999);
  try { document.execCommand('copy'); } catch (e) {}
  if (navigator.clipboard) navigator.clipboard.writeText(t.value).catch(()=>{});
  document.getElementById('copied').textContent = ' copied';
  setTimeout(()=>{document.getElementById('copied').textContent='';}, 1500);
}
function showjson(){
  const w = document.getElementById('jsonwrap');
  if (!w) return;                      // never let a missing box break the page
  const ok = marks.near && marks.far;
  w.style.display = ok ? 'block' : 'none';
  if (ok) document.getElementById('json').value = payload();
}
pick('near'); load();
</script>
"""


def build(clip: str) -> str:
    import clip_registry
    doc = clip_registry.load(clip)
    if doc is None:
        raise SystemExit(f"no clips/{clip}.json")
    shots = _frames(doc)
    out = os.path.join(_HERE, "out", f"{clip}_rim_clicker.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    page = (_PAGE.replace("__DATA__", json.dumps({"clip": clip, "shots": shots}))
                 .replace("__CLIP__", clip))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"[rim_clicker] {clip}: {len(shots)} frame(s) to choose from (keyframes first, then a sweep across the game) -> {out}")
    print(f"[rim_clicker] open it, mark both rims, Download, then:")
    print(f"    .venv/Scripts/python.exe spikes/make_rim_clicker.py {clip} "
          f"--save <the downloaded {clip}_rims.json>")
    return out


def save(clip: str, path: str) -> dict:
    """Write the clicked rims into the clip's registry document.

    Also sets the ball span to the tracking span, because a rim with no span is
    still a clip with no shot layer -- and a half-configured clip that looks
    configured is exactly the silent failure this project keeps refusing.
    """
    import clip_registry
    marked = json.load(open(path, encoding="utf-8"))
    hoops = marked.get("hoop_anchors") or {}
    if set(hoops) != {"far", "near"}:
        raise SystemExit(f"{path}: expected both 'far' and 'near', got {sorted(hoops)}")
    doc = clip_registry.load(clip)
    if doc is None:
        raise SystemExit(f"no clips/{clip}.json")
    span_start = doc.get("tracking_span_start", 0)
    span_len = doc.get("tracking_span_len")
    if not span_len:
        raise SystemExit(f"{clip}: no tracking span yet -- set one before the ball layer")
    # A rim no longer has to sit on a keyframe: hoop_anchor SIFT-matches any
    # marked frame to its nearest keyframe and REFUSES a weak match, so the
    # check that used to live here is now made where the geometry is (and where
    # it can see the inlier count). Marking on a keyframe is still the surest
    # option and is noted below.
    kfs = set(doc.get("keyframes") or [])
    for side, (kf, _px) in hoops.items():
        if kf not in kfs:
            print(f"[rim_clicker] NOTE: {side} rim is on frame {kf}, not a "
                  f"keyframe. hoop_anchor will carry it by SIFT-matching that "
                  f"frame, and will refuse it out loud if the match is weak.")
    doc = clip_registry.update(clip, hoop_anchors=hoops,
                               ball_span_start=span_start, ball_span_len=span_len)
    print(f"[rim_clicker] {clip}: rims saved, ball span {span_start}..+{span_len}")
    for side, (kf, px) in hoops.items():
        print(f"    {side:>4} rim: keyframe {kf}, pixel {tuple(px)}")
    return doc


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_rim_clicker.py <CLIP> [--save <rims.json>]")
    clip = sys.argv[1]
    if "--save" in sys.argv:
        save(clip, sys.argv[sys.argv.index("--save") + 1])
    else:
        build(clip)


if __name__ == "__main__":
    main()
