# GETTING A WHOLE GAME UNDER 10 MINUTES OF CLICKING

**DJ, 2026-07-29:** "16min of clicking to me is still unacceptable... an entire
game only 5-10 min of clicking. This is a must have for the product... low
friction is always a must in great apps."

Researched 2026-07-29 by me + three agents (deep CV maths, software survey,
measurement). **11 methods below.** Nothing built yet — this is the menu.

---

## THE ARITHMETIC WE HAVE TO BEAT

A full game video ≈ 60 min at 30fps ≈ **108,000 frames**.

| | frames/mark | marks | clicks | time |
|---|---|---|---|---|
| Original convention | 100 | ~1,080 | ~10,800 | ~12 h |
| + smart spacing (TEST 29) | ~300 | ~360 | ~1,800 | ~2 h |
| + 5 clicks not 10 (TEST 31) | ~300 | ~360 | ~1,800 | ~2 h |
| **TARGET** | — | — | **100–150** | **5–10 min** |

Needs a further **12–18×**. Trimming is exhausted; this is structural.

---

> ## ⚠️ TESTED 2026-07-29 — FACT 1 BELOW IS FALSE. READ TEST 32 FIRST.
> The tripod/pure-rotation premise **failed measurement**: 19–26 px error in the
> image *centre* against a 2 px bar, on both clips, and the centre/edge
> diagnostic rules out lens distortion as the cause. **M4's "20–40 clicks" and
> M9's "5–10 clicks" therefore have no measured foundation** — they were
> projections from this premise, and the premise does not hold on DJ's footage.
> **M1 (keyframe by VIEW) survives untouched** because it never depended on it:
> 13 distinct views measured in 5 minutes, ~200 clicks per game.
> Facts 2 and 3 below are left as written so the reasoning is auditable.

## THE THREE FACTS THAT MAKE THE TARGET REACHABLE

### 1. ~~The camera is a TRIPOD. It rotates; it does not travel.~~ ❌ DISPROVED
A camera rotating about a fixed point with fixed intrinsics relates views by a
**pure-rotation homography — 3 DOF, 4 with zoom.** Not the general 8.
**This was tested and is false for this footage — see TEST 32.** Probable
causes: the tripod head does not rotate about the optical centre (real
parallax), or the pan is partly a digital crop from a wide sensor.

### 2. So the clicks are paying for the wrong thing. *(rests on fact 1 — now unfounded)*
Today: 8 DOF × 360 keyframes ≈ **2,900 unknowns**, all bought with clicks.
Physically: 4 DOF per keyframe **which SIFT gives for free**, plus roughly
**15 global numbers** for the whole game (focal, principal point, one radial
term, nodal offset, court-plane pose). Clicks should scale with the ~15 globals,
not with the keyframe count. That is where 12–18× lives.

### 3. MEASURED 2026-07-29 — a game contains very few distinct views.
Test4, the longest clip on disk (5 min, 9,022 frames), 151 frames sampled:

```
matched to ONE reference frame:  151 / 151   (zero failures)
camera centre travel:            989 px  (~one frame width, at half scale)
distinct views, 60 px tolerance:  13
                120 px:            7
                200 px:            4
```

**13 distinct views in five minutes.** The camera returns to the same framings
constantly, so the view count grows far slower than time. Even tripling for a
full game is ~40 views × 5 clicks = **200 clicks**, and that is before any of
the smarter methods below.
*Caveat, stated honestly:* "matched" means a homography was found with ≥20
inliers — it does NOT prove the homography is accurate. And Test4's pan is
modest (~one frame width); a game following play end-to-end will pan more.

---

# THE 11 METHODS

## ⭐ M4 — PTZ reparameterisation *(top pick — grounded, not speculative)*
Replace the 8-DOF-per-keyframe unknowns in `phase1/refit_keyframes.py` with
(pan, tilt, roll, log-focal) per keyframe **plus ~15 shared globals**. Clicks
stop paying per keyframe and pay only for the globals.
**Clicks/game: 20–40. Accuracy: ≤0.2 ft expected.**
**Why it's the top pick:** the global optimiser, the dense SIFT
correspondences, and the leave-one-out harness *all already exist in this repo*.
**Risk:** zoom couples into principal point and lens distortion; a bad radial
term shows up at the court edges, exactly where there are no clicks.
**Test (~1 day):** swap `s3.H_to8`/`H_from8` for a 4-DOF+globals pack/unpack,
residuals untouched. Then refit TEST2 from only 2 of its 8 keyframes' clicks and
score the other 6.

## ⭐ M9 — Whole-game mosaic + loop closure
The game IS the panorama-stitching problem, solved to sub-pixel accuracy. Build
a keyframe graph over all frames, one global rotational bundle adjustment; a
panning camera revisits angles constantly so loop constraints kill drift.
**Clicks/game: 5–10 (one calibrated mosaic).**
**The old objection does NOT apply — verified.** `phase1/DECISIONS.md` rejected
an *open-loop accumulated ORB chain* (>40 px off on 74% of frames). The *global*
version was then built anyway (`refit_keyframes.py`) and **improved** accuracy:
consistency 8.1→0.6 px, landmarks 0.25→0.14 ft. Bundle-adjusted chaining is
already shipped here at N=7.
**Tool: OpenCV Stitching — Apache-2.0 (permissive, no copyleft problem), with a
bundle adjuster and warps built for a rotating camera.**
**Risk:** 10 moving players + floor glare poison correspondences during dense
play; 100k frames needs incremental solving (reports of minutes for 600 frames).

## ⭐ M3 — Calibrate the VENUE, not the game
Intrinsics belong to the lens; court-plane pose belongs to the tripod spot.
Solve both once per gym; later games there re-solve only the tripod pose.
**Clicks after the first game: 4–8.** Falls straight out of M4.
**Risk:** the tripod is never placed identically, so pose must be re-solved —
"zero clicks" would be dishonest. Needs a cheap "has the camera moved?" check.

## M1 — Keyframe by VIEW, not by clock
Mark one representative per distinct camera framing instead of one per time
interval. **Now measured: 13 views per 5 min (see above).**
**Clicks/game: ~100–200.** Hits the target on its own, with no new maths.
**Risk:** two framings that look alike but aren't (both baskets are similar).

## M2 — Skip dead ball *(half the game, half already built)*
~Half a game video is free throws, timeouts, inbounds, dead clock. None needs a
court. **A working dead-ball detector already exists** — clock-rhythm,
style-independent, 151/155 seconds correct (TEST 20), built and unwired.
**~2× reduction, and it multiplies with everything else.**
**Risk:** validated on one clip only.

## M5 — Snap to the painted lines *(zero clicks — a refiner)*
Take a rough court, detect edges, and pull the known template onto the real
painted lines (chamfer / iterative-closest-line). Multiplies with every method.
**Clicks: 0. Accuracy: 0.1–0.3 ft when it locks.**
**SEVERE RISK, flagged by the maths agent:** high-school gyms have volleyball
and badminton lines painted over the basketball lines. Parallel-line aliasing
can settle **one lane-width off and be confidently wrong** — this project's
worst failure class. Only ever adopt as a *bounded* refiner (reject any snap
that moves the court by more than half the minimum line spacing).
**Test (~2 hours, do this one first — ground truth is free):** perturb a known
good homography by 2 ft, run the snap, measure recovery. Also count how many
detected lines are not basketball lines.

## M6 — Fine-tune a court detector on DJ's gyms
KaliCalib already *finds* courts (TEST 30) but is imprecise on high-school
footage. Retrain on DJ's 187 clicked landmarks.
**Licence-clean alternatives found:** Roboflow Universe basketball court
keypoint models (**CC BY 4.0**, actually basketball-specific, downloadable
today — cheapest smoke test); PnLCalib (pretrained, newer net, but **GPL-2.0**,
same copyleft concern as KaliCalib).
**Needs the GPU DJ is renting.**

## M11 — Pan-tilt-zoom SLAM *(purpose-built for this exact problem)*
`github.com/lulufa390/Pan-tilt-zoom-SLAM` — online SLAM with ray landmarks,
built specifically for **broadcast basketball/soccer PTZ** tracking. Sparse
calibration propagated across pan and zoom is precisely our shape.
**Risk:** licence unstated — must verify before any product use.

## M7 — Nudge, don't place *(interaction, not maths)*
Stop asking for clicks on an empty frame. Show the court overlay and let the
coach **drag or rotate a rendered wireframe** until it lines up. One gesture
carries ~4 points of information and is verifiable at a glance — a real ~4×.
**Risk (measured):** at 2 ft off, a proposal is visibly wrong and dragging may
not beat clicking fresh (TEST 30). Depends on M5/M6 landing first.

## M8 — Progressive / on-demand calibration
Never calibrate a whole game up front. Calibrate the possessions the coach
actually opens, in the background, while they watch. Felt cost → ~zero.
Hides latency rather than removing work — but "low friction" is the requirement.

## M10 — Vanishing-point self-calibration
Sidelines and baselines give two orthogonal floor vanishing points; with square
pixels that solves focal length per frame with **zero clicks**.
**Speculative and downstream of M5** — VPs go unstable exactly when zoomed in.
Worth 30 minutes as a cross-check on M4's focal estimates, no more.

### Ruled out
- **COLMAP** default pipeline — fails under pure rotation (no triangulation
  baseline). Only its special `panorama_sfm` path could apply.
- **Commercial APIs** — none self-serve. Hawk-Eye / Second Spectrum / Pixellot /
  Sportlogiq are enterprise-contract only.
- **sportsfield_release** — non-commercial academic licence, patent-pending.

---

## HOW THEY MULTIPLY

```
M2 dead-ball skip    ~2x     already built, just unwired
M1 view keyframes   ~12x     MEASURED: 13 views / 5 min
M4 PTZ maths        ~2x      fewer clicks per mark
M3 venue reuse       ~Nx     across a season
M5 line snapping      —      removes clicks by improving accuracy
```

**M4 alone is claimed at 20–40 clicks/game. M1 alone measures to ~200. M3 makes
the second game at a venue nearly free.** The target is reachable more than one
way, which is the best news in this document.

---

## THE DECISIVE TEST — ALREADY RUN, with the CURRENT 8-DOF model

Refit TEST2 from a subset of its 8 keyframes, score the held-back marks:

```
keep 5 of 8   47/70 clicks (33% fewer)   worst held-out 0.27 ft   PASS
keep 3 of 8   29/70 clicks (59% fewer)   worst held-out 0.40 ft   PASS
keep 2 of 8   19/70 clicks (73% fewer)   worst held-out 73.3 ft   COLLAPSE
```

**The current model floors out at about 3 marks per span** — 2 collapses on both
TEST2 (73 ft) and HARD (39 ft), though TEST1 survived it. That is the number
M4 has to beat, and it is exactly why M4 matters: at 8 unknowns per keyframe the
fit starves. At 4 unknowns per keyframe plus shared globals it should not.

**So the baseline is established and the target is not yet met by trimming.**
The next test is M4's: swap the 8-DOF parameterisation for 4-DOF + globals and
re-run this identical holdout. Same harness, same data, one honest comparison.

## STANDING RULES
- Nothing adopts on a number. Holdout against DJ's 187 clicked landmarks, then
  DJ eyeballs an overlay.
- **Savings measured separately are not proven to compose.** TESTs 29 and 31
  each found ~2× and even that pair is still untested together.
- Licence must be checked before anything ships: Apache-2.0 (OpenCV) and
  CC BY 4.0 (Roboflow) are clean; GPL/CeCILL are not.
