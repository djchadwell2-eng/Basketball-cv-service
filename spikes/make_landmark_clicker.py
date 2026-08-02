"""Build a self-contained HTML page for clicking court landmarks in a BROWSER.

WHY NOT AN OPENCV WINDOW: this venv has opencv-python-headless installed
alongside opencv-python, and the headless build shadows the GUI one -- so
cv2.namedWindow raises "The function is not implemented" and any cv2-based
clicker dies on launch. Uninstalling headless risks breaking ultralytics and the
rest of a working pipeline for the sake of a click box, which is a bad trade.

So this follows the pattern the project ALREADY uses to collect clicks from DJ:
phase2/make_review_bundle.py writes an HTML file, he works in the browser, and
downloads a JSON. Same shape here.

Frames are the 4-frame CHAIN plan (spikes/out/FULLGAME_chain_plan.json), verified
frame-accurate at FULL RESOLUTION with the scorebug masked on 2026-07-30
(spikes/verify_chain_fullres.py -- every adjacent pair ratio >= 0.6, the
project's own weak-pair bar). This replaces the earlier 5-frame COVERAGE set
(200/16000/65800/79200/169000), whose calibration FAILED at 15.45 ft because two
of those frames didn't chain together (TEST 36) -- chainability, not coverage, is
what calibration needs. Embedded as base64 JPEGs so the page is one file with no
server and no external requests.

Output: spikes/out/{TAG}_chain_clicker.html
        -> click, then Download -> spikes/out/{TAG}_chain_landmarks.json
        (deliberately DIFFERENT filenames from FULLGAME_landmarks.json, which
        holds DJ's 63 good clicks on the old frame set -- do not overwrite those)

GENERALIZED 2026-07-30 to any game: the frame list is read from the chain plan
that spikes/verify_chain_fullres.py has ALREADY PASSED at full resolution --
never a hand-typed list, so a set of frames physically cannot reach DJ unless
its links were verified first. That ordering is the whole lesson of TEST 36.

Usage:  .venv/Scripts/python.exe spikes/make_landmark_clicker.py [video_name]
"""

from __future__ import annotations

import base64
import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

VIDEO_NAME = sys.argv[1] if len(sys.argv) > 1 else "Full_Game.mp4"
VIDEO = os.path.join(_ROOT, VIDEO_NAME)
_TAG = os.path.splitext(VIDEO_NAME)[0].upper()
if _TAG == "FULL_GAME":
    _TAG = "FULLGAME"                  # keep the original game's existing filenames
OUT_HTML = os.path.join(_HERE, "out", f"{_TAG}_chain_clicker.html")
VERIFY_JSON = os.path.join(_HERE, "out",
                           f"{os.path.splitext(VIDEO_NAME)[0].upper()}_chain_verify_fullres.json")

# (tag, court_x, court_y, plain-English description). The court dimension is
# SOLVED from these clicks later (spikes/court_detect.py), never assumed -- these
# x/y are only used to draw the little diagram showing which point is wanted.
# ORDERING AND THE "essential" FLAG COME FROM A REAL FAILURE (2026-07-29). The
# first session offered only these 9, all free-throw lines and the centre. The
# solve produced a GOOD court (0.32 ft) but could not tell an 84 ft floor from a
# 94 ft one -- 0.32 vs 0.33 ft, a dead heat -- because nothing was marked at a
# BASELINE, leaving no anchor to measure the floor's LENGTH against. court_detect
# refused rather than guess, correctly: assuming 84 for TEST2's real 94 ft floor
# is exactly what made it read 0.94 ft and look "completely wrong".
# The only baseline options offered were the far COURT CORNERS, which the crowd
# usually hides. The LANE-BASE points -- where the painted key meets the baseline
# -- are large, painted and nearly always visible, and TEST1/TEST2 both lean on
# them. They now come FIRST and are flagged essential.
# Also: DJ, 2026-07-29 -- "I dont mind more clicks if nessessary as long as there
# only 5-8 frames." So the list is now the FULL landmark set; frames are the
# scarce resource, clicks are not.
LANDMARKS = [
    ("L_lane_base_near",  0.0, 19.0, "LEFT baseline meets the NEAR edge of the painted key", True),
    ("L_lane_base_far",   0.0, 31.0, "LEFT baseline meets the FAR edge of the painted key", True),
    ("R_lane_base_near", 84.0, 19.0, "RIGHT baseline meets the NEAR edge of the painted key", True),
    ("R_lane_base_far",  84.0, 31.0, "RIGHT baseline meets the FAR edge of the painted key", True),
    ("center_logo",      42.0, 25.0, "EXACT CENTRE of the centre circle", False),
    ("center_near",      42.0,  0.0, "HALF-COURT line meets the NEAR sideline", False),
    ("center_far",       42.0, 50.0, "HALF-COURT line meets the FAR sideline", False),
    ("L_FT_near",        19.0, 19.0, "LEFT free-throw line meets the NEAR edge of the lane", False),
    ("L_FT_far",         19.0, 31.0, "LEFT free-throw line meets the FAR edge of the lane", False),
    ("R_FT_near",        65.0, 19.0, "RIGHT free-throw line meets the NEAR edge of the lane", False),
    ("R_FT_far",         65.0, 31.0, "RIGHT free-throw line meets the FAR edge of the lane", False),
    ("LB_side_near",      0.0,  0.0, "LEFT baseline meets the NEAR sideline (near corner, left end)", False),
    ("LB_side_far",       0.0, 50.0, "LEFT baseline meets the FAR sideline (far corner, left end)", False),
    ("RB_side_near",     84.0,  0.0, "RIGHT baseline meets the NEAR sideline (near corner, right end)", False),
    ("RB_side_far",      84.0, 50.0, "RIGHT baseline meets the FAR sideline (far corner, right end)", False),
    ("circle_left",      36.0, 25.0, "LEFT edge of the centre circle (9 o'clock)", False),
    ("circle_right",     48.0, 25.0, "RIGHT edge of the centre circle (3 o'clock)", False),
    ("circle_bottom",    42.0, 19.0, "NEAR edge of the centre circle (closest to the camera)", False),
    ("circle_top",       42.0, 31.0, "FAR edge of the centre circle (furthest from the camera)", False),
    ("L_arc_top",        25.0, 25.0, "TOP of the LEFT 3-point arc (furthest point from that basket)", False),
    ("R_arc_top",        59.0, 25.0, "TOP of the RIGHT 3-point arc", False),
]


def frames_to_use():
    """The VERIFIED chain, read from verify_chain_fullres.py's own output.

    REFUSES to build a page from a chain that did not pass. Asking DJ to click
    an unverified frame set is precisely what cost him a whole session in
    TEST 36, so the gate lives in code here rather than in someone's memory.
    Third slot is unused (no "coverage" concept for a chain); kept at 0 for
    the UI.
    """
    if not os.path.exists(VERIFY_JSON):
        raise SystemExit(
            f"REFUSING TO BUILD A CLICKER -- no full-resolution verification on "
            f"file for {VIDEO_NAME}.\n  Expected: {VERIFY_JSON}\n  Run first: "
            f".venv/Scripts/python.exe spikes/verify_chain_fullres.py {VIDEO_NAME}")
    doc = json.load(open(VERIFY_JSON, encoding="utf-8"))
    if doc.get("weak"):
        raise SystemExit(
            f"REFUSING TO BUILD A CLICKER -- {VIDEO_NAME}'s chain has "
            f"{len(doc['weak'])} WEAK link(s): {doc['weak']}\n  Verdict on file: "
            f"{doc.get('verdict')}\n  Bridge the gap first "
            f"(spikes/bridge_gap_fullres.py), then re-verify.")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"  chain VERIFIED at full resolution: {doc['verdict']}")
    return [(f, f / fps, 0) for f in doc["candidate"]]


def main():
    if not os.path.exists(VIDEO):
        raise SystemExit(f"missing {VIDEO}")
    want = frames_to_use()
    cap = cv2.VideoCapture(VIDEO)
    shots = []
    for (f, t, cov) in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            print(f"  WARNING could not read frame {f} -- skipped")
            continue
        ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok2:
            continue
        shots.append({"frame": int(f), "t": float(t), "cov": int(cov),
                      "w": img.shape[1], "h": img.shape[0],
                      "b64": base64.b64encode(buf).decode("ascii")})
        print(f"  embedded frame {f} ({len(buf)//1024} KB)", flush=True)
    cap.release()
    if not shots:
        raise SystemExit("no frames could be read")

    prior = {}
    lm_json = os.path.join(_HERE, "out", f"{_TAG}_chain_landmarks.json")
    if os.path.exists(lm_json):
        prior = json.load(open(lm_json, encoding="utf-8"))
        n = sum(len(v) for v in prior.values())
        print(f"  pre-loading {n} marks already clicked across "
              f"{len(prior)} frame(s) -- they will not need redoing")
    payload = json.dumps({"shots": shots, "landmarks": LANDMARKS,
                          "prior": prior, "tag": _TAG})
    html = _TEMPLATE.replace("__PAYLOAD__", payload)
    _check_js(html)                      # refuse to write a page that cannot run
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\nwrote {OUT_HTML}  ({os.path.getsize(OUT_HTML)//1024} KB)")


def _check_js(html):
    """Syntax-check the page's JavaScript with node before writing it.

    WHY THIS EXISTS: a broken page was handed over TWICE. A JS SyntaxError kills
    the entire script block, so the symptoms are silent -- no tabs, no list, a
    broken image, and no error message, because the on-page error handler lives
    in the block that failed to parse. Static string checks did not catch it;
    only a real parser does. If node is unavailable the check is SKIPPED LOUDLY
    rather than silently passing.
    """
    import shutil
    import subprocess
    import tempfile
    node = shutil.which("node")
    if not node:
        print("  WARNING: node not found -- JS was NOT syntax-checked")
        return
    i, j = html.index("<script>") + len("<script>"), html.rindex("</script>")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(html[i:j])
        tmp = fh.name
    try:
        r = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("REFUSING TO WRITE -- the page's JavaScript does not "
                             "parse:\n" + (r.stderr or r.stdout))
        print("  JS syntax check passed (node --check)")
    finally:
        os.unlink(tmp)

_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Full_Game -- click court landmarks</title>
<style>
 body{margin:0;background:#151517;color:#e8e8ea;font:14px/1.45 system-ui,sans-serif}
 header{padding:10px 14px;background:#1e1e22;border-bottom:1px solid #333;
        display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600}
 .tabs{display:flex;gap:6px;flex-wrap:wrap}
 .tab{padding:5px 10px;border-radius:5px;background:#2a2a30;cursor:pointer;
      border:1px solid #3a3a42;font-size:12px}
 .tab.on{background:#2563eb;border-color:#2563eb}
 .tab .n{opacity:.65;font-size:11px}
 main{display:flex;gap:14px;padding:14px;align-items:flex-start}
 #stage{position:relative;flex:1;min-width:0}
 #img{display:block;width:100%;height:auto;cursor:crosshair;border-radius:6px}
 #loupe{position:absolute;width:180px;height:180px;border:2px solid #22c55e;
        border-radius:50%;pointer-events:none;display:none;overflow:hidden;
        box-shadow:0 0 0 3px rgba(0,0,0,.6)}
 #loupe canvas{display:block}
 aside{width:330px;flex:0 0 330px}
 .lm{padding:7px 9px;border-radius:6px;background:#22222a;margin-bottom:5px;
     cursor:pointer;border:1px solid transparent}
 .lm.on{border-color:#2563eb;background:#1d2c4d}
 .lm.done{background:#14321f}
 .lm b{font-size:12.5px} .lm p{margin:3px 0 0;font-size:11.5px;opacity:.75}
 .lm .xy{font-size:11px;opacity:.7;font-family:ui-monospace,monospace}
 .ess{font-size:9.5px;background:#b45309;color:#fff;padding:1px 5px;border-radius:8px;
      vertical-align:middle}
 button{padding:8px 13px;border-radius:6px;border:0;background:#2563eb;
        color:#fff;font-weight:600;cursor:pointer}
 button.g{background:#16a34a} button.s{background:#3a3a42}
 svg{background:#1b1b20;border-radius:6px;margin-bottom:8px}
 .hint{font-size:12px;opacity:.7;padding:0 14px 12px}
 .warn{color:#fbbf24}
 .keys{margin-top:7px;display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;opacity:.85}
 kbd{background:#33333c;border:1px solid #4a4a55;border-bottom-width:2px;
     border-radius:4px;padding:1px 5px;font:11px ui-monospace,monospace;color:#e8e8ea}
</style>
<header>
  <h1>Full_Game — click court landmarks</h1>
  <div class="tabs" id="tabs"></div>
  <span id="status" style="flex:1;font-size:12px;color:#9ca3af"></span>
  <button class="g" id="dl">Download JSON</button>
  <button class="s" id="clr">Clear this frame</button>
</header>
<div class="hint">
  Pick a landmark on the right, then click it on the photo. A magnifier follows
  your cursor. <b>Clicks are cheap, frames are not &mdash; place as many as you
  can see on each frame.</b> Progress is saved in this browser as you go.
  <div class="keys">
    <span><kbd>click</kbd> place / move the point</span>
    <span><kbd>S</kbd> / <kbd>&darr;</kbd> / <kbd>Enter</kbd> skip to next landmark</span>
    <span><kbd>B</kbd> / <kbd>&uarr;</kbd> back one landmark</span>
    <span><kbd>D</kbd> / <kbd>Del</kbd> delete this landmark's point</span>
    <span><kbd>Z</kbd> undo last change</span>
    <span><kbd>&larr;</kbd> <kbd>&rarr;</kbd> or <kbd>1</kbd>&ndash;<kbd>5</kbd> switch frame</span>
  </div>
</div>
<main>
  <div id="stage">
    <img id="img" alt="frame">
    <div id="loupe"><canvas id="lc" width="180" height="180"></canvas></div>
  </div>
  <aside>
    <svg id="diag" width="330" height="205" viewBox="0 0 330 205"></svg>
    <div id="list"></div>
  </aside>
</main>
<script>
window.onerror = (m,_s,l)=>{ const e=document.getElementById('status');
  if(e){ e.textContent='JS error line '+l+': '+m; e.style.color='#f87171'; } };
const D = __PAYLOAD__;
const KEY = D.tag + "_chain_landmarks_v1";    // per-video, so two games never collide
let store = JSON.parse(localStorage.getItem(KEY) || "null");
if (!store) {                        // first open: start from what is already clicked
  store = {};
  for (const [f, rows] of Object.entries(D.prior || {})) {
    store[f] = {}; rows.forEach(r => store[f][r[0]] = [r[1], r[2]]);
  }
}
let fi = 0, li = 0, ready = false, natReady = false;
const hist = [];
const img = document.getElementById('img'), loupe = document.getElementById('loupe');
const lc = document.getElementById('lc').getContext('2d');
const nat = new Image();

function shot(){ return D.shots[fi]; }
function key(){ return String(shot().frame); }
function pts(){ store[key()] = store[key()] || {}; return store[key()]; }
function save(){ localStorage.setItem(KEY, JSON.stringify(store)); }

function tabs(){
  document.getElementById('tabs').innerHTML = D.shots.map((s,i)=>{
    const n = Object.keys(store[String(s.frame)]||{}).length;
    const m = Math.floor(s.t/60), sec = String(Math.floor(s.t%60)).padStart(2,'0');
    return `<div class="tab ${i===fi?'on':''}" data-i="${i}">${i+1}. ${m}m${sec}s
            <span class="n">${n} clicked</span></div>`;
  }).join('');
  document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{fi=+t.dataset.i;li=0;load();});
}

function list(){
  const p = pts();
  document.getElementById('list').innerHTML = D.landmarks.map((L,i)=>{
    const has = p[L[0]];
    const badge = L[4] ? ' <span class="ess">NEEDED FOR COURT SIZE</span>' : '';
    return `<div class="lm ${i===li?'on':''} ${has?'done':''}" data-i="${i}">
      <b>${L[0]}</b>${badge} ${has?`<span class="xy">${has[0].toFixed(0)}, ${has[1].toFixed(0)}</span>`:''}
      <p>${L[3]}</p></div>`;
  }).join('');
  document.querySelectorAll('.lm').forEach(e=>e.onclick=()=>{li=+e.dataset.i;list();diag();draw();});
}

function diag(){
  const L = D.landmarks[li], pad=16, W=330-2*pad, H=205-2*pad;
  const X=f=>pad+f/84*W, Y=f=>pad+f/50*H;
  let s = `<rect x="${X(0)}" y="${Y(0)}" width="${X(84)-X(0)}" height="${Y(50)-Y(0)}"
            fill="none" stroke="#8a8a95"/>
      <line x1="${X(42)}" y1="${Y(0)}" x2="${X(42)}" y2="${Y(50)}" stroke="#8a8a95"/>
      <ellipse cx="${X(42)}" cy="${Y(25)}" rx="${X(6)-X(0)}" ry="${Y(6)-Y(0)}" fill="none" stroke="#8a8a95"/>
      <rect x="${X(0)}" y="${Y(19)}" width="${X(19)-X(0)}" height="${Y(31)-Y(19)}" fill="none" stroke="#8a8a95"/>
      <rect x="${X(65)}" y="${Y(19)}" width="${X(84)-X(65)}" height="${Y(31)-Y(19)}" fill="none" stroke="#8a8a95"/>`;
  s += `<circle cx="${X(L[1])}" cy="${Y(L[2])}" r="7" fill="none" stroke="#fbbf24" stroke-width="2"/>
        <circle cx="${X(L[1])}" cy="${Y(L[2])}" r="2.5" fill="#fbbf24"/>
        <text x="${pad}" y="196" fill="#fbbf24" font-size="11">${L[0]}</text>`;
  document.getElementById('diag').innerHTML = s;
}

let overlay = null;
function draw(){
  if (overlay) overlay.remove();
  overlay = document.createElement('div');
  Object.assign(overlay.style,{position:'absolute',inset:'0',pointerEvents:'none'});
  const sc = img.clientWidth / shot().w;
  const p = pts();
  D.landmarks.forEach((L,i)=>{
    const v = p[L[0]]; if(!v) return;
    const d = document.createElement('div');
    const on = i===li;
    Object.assign(d.style,{position:'absolute',left:(v[0]*sc-6)+'px',top:(v[1]*sc-6)+'px',
      width:'12px',height:'12px',borderRadius:'50%',
      border:'2px solid '+(on?'#fbbf24':'#22c55e'),background:'rgba(0,0,0,.35)'});
    overlay.appendChild(d);
  });
  document.getElementById('stage').appendChild(overlay);
}

function load(){
  const s = shot();
  // BUG FIXED 2026-07-29: onload was attached AFTER src. An embedded data URI
  // can fire load synchronously, so draw() never ran and already-placed marks
  // were invisible -- the page looked broken on open.
  img.onload = ()=>{ ready = true; draw(); status(''); };
  img.onerror = ()=> status('could not decode frame '+s.frame, true);
  ready = false;
  nat.onload = ()=>{ natReady = true; };
  natReady = false;
  nat.src = 'data:image/jpeg;base64,' + s.b64;
  img.src = nat.src;
  if (img.complete && img.naturalWidth) { ready = true; draw(); }
  tabs(); list(); diag();
}

function status(msg, bad){
  const el = document.getElementById('status');
  el.textContent = msg || '';
  el.style.color = bad ? '#f87171' : '#9ca3af';
}

img.addEventListener('mousemove', e=>{
  if (!natReady) return;              // was drawing from an unloaded image = blank loupe
  const r = img.getBoundingClientRect(), sc = shot().w / r.width;
  const nx = (e.clientX - r.left) * sc, ny = (e.clientY - r.top) * sc;
  loupe.style.display='block';
  // keep the loupe on screen: it used to sit 200px ABOVE the cursor and vanish
  // off the top of the frame whenever you worked near the top edge.
  const lx = e.clientX - r.left, ly = e.clientY - r.top;
  loupe.style.left = Math.max(0, Math.min(r.width  - 184, lx - 92)) + 'px';
  loupe.style.top  = (ly > 210 ? ly - 200 : ly + 24) + 'px';
  const Z = 4, S = 180/Z;
  lc.imageSmoothingEnabled = false;
  lc.clearRect(0,0,180,180);
  try { lc.drawImage(nat, nx-S/2, ny-S/2, S, S, 0, 0, 180, 180); } catch(_e){}
  lc.strokeStyle='#22c55e'; lc.lineWidth=1;
  lc.beginPath(); lc.moveTo(90,60); lc.lineTo(90,120); lc.moveTo(60,90); lc.lineTo(120,90); lc.stroke();
});
img.addEventListener('mouseleave', ()=>loupe.style.display='none');

img.addEventListener('click', e=>{
  const r = img.getBoundingClientRect(), sc = shot().w / r.width;
  const nx = +((e.clientX - r.left) * sc).toFixed(1);
  const ny = +((e.clientY - r.top) * sc).toFixed(1);
  const tag = D.landmarks[li][0];
  hist.push({f: key(), tag: tag, prev: pts()[tag] || null});   // for undo
  if (hist.length > 200) hist.shift();
  pts()[tag] = [nx, ny];
  save();
  if (li < D.landmarks.length - 1) li++;
  tabs(); list(); diag(); draw();
});

document.getElementById('clr').onclick = ()=>{
  if (confirm('Clear every point on this frame?')) { store[key()]={}; save(); tabs(); list(); draw(); }
};
document.getElementById('dl').onclick = ()=>{
  const out = {};
  D.shots.forEach(s=>{
    const p = store[String(s.frame)] || {};
    const rows = Object.entries(p).map(([t,v])=>[t,v[0],v[1]]);
    if (rows.length) out[String(s.frame)] = rows;
  });
  const isBase = t => /lane_base|B_side_/.test(t);
  const anyBase = Object.values(out).some(rows => rows.some(r => isBase(r[0])));
  if (!anyBase) {
    // Message built with an array + join so it contains NO backslash escape
    // sequences. A previous version used "\n" written through a shell
    // heredoc, the backslashes were eaten, and real newlines inside a JS
    // string literal produced a SyntaxError that killed the whole script.
    status([
      "No BASELINE landmark yet -- cannot tell an 84 ft floor from a 94 ft one",
      "(it scored 0.32 vs 0.33 ft and refused to guess).",
      "Mark one orange NEEDED FOR COURT SIZE point on any frame showing an end."
    ].join(" "), true);
    return;
  }
  const thin = Object.entries(out).filter(([,v])=>v.length<5).map(([k])=>k);
  if (thin.length && !confirm('Frames with fewer than 5 points: '+thin.join(', ')+
      '\nThe court fit may be weak there. Download anyway?')) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  a.download = D.tag + '_chain_landmarks.json';
  a.click();
};
function refresh(){ tabs(); list(); diag(); draw(); }

function deleteCurrent(){
  const tag = D.landmarks[li][0], p = pts();
  if (p[tag] === undefined) { status('nothing placed on ' + tag + ' yet'); return; }
  hist.push({f: key(), tag: tag, prev: p[tag]});
  delete p[tag];
  save(); refresh(); status('removed ' + tag);
}

function undo(){
  const h = hist.pop();
  if (!h) { status('nothing left to undo'); return; }
  store[h.f] = store[h.f] || {};
  if (h.prev === null) delete store[h.f][h.tag];
  else store[h.f][h.tag] = h.prev;
  save(); refresh();
  status('undid ' + h.tag);
}

document.addEventListener('keydown', e=>{
  const k = e.key;
  if (k === 'ArrowDown' || k === 's' || k === 'S' || k === 'Enter'){
    li = Math.min(li + 1, D.landmarks.length - 1); refresh(); e.preventDefault(); return; }
  if (k === 'ArrowUp' || k === 'b' || k === 'B'){
    li = Math.max(li - 1, 0); refresh(); e.preventDefault(); return; }
  if (k === 'd' || k === 'D' || k === 'Delete' || k === 'Backspace'){
    deleteCurrent(); e.preventDefault(); return; }
  if (k === 'z' || k === 'Z' || ((e.ctrlKey || e.metaKey) && k === 'z')){
    undo(); e.preventDefault(); return; }
  if (k === 'ArrowRight'){
    fi = Math.min(fi + 1, D.shots.length - 1); li = 0; load(); e.preventDefault(); return; }
  if (k === 'ArrowLeft'){
    fi = Math.max(fi - 1, 0); li = 0; load(); e.preventDefault(); return; }
  if (k >= '1' && k <= '9'){
    const n = +k - 1;
    if (n < D.shots.length){ fi = n; li = 0; load(); e.preventDefault(); } return; }
});
load();
</script>
"""


if __name__ == "__main__":
    main()
