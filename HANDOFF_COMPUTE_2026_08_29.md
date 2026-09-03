# Handoff — compute, cost and speed (updated 2026-09-03)

Same tagging rule as `HANDOFF_GPU_SESSION.md`, for the same reason:

- **[MEASURED]** — came out of a real run. Trust it.
- **[ESTIMATE]** — arithmetic on measured numbers. Could be wrong.
- **[UNKNOWN]** — never established. Do not act as if it is.

**THE HEADLINE: a full 95-minute game ran end to end for the first time.**
171,120 frames, merge + identity tail in **96.5 minutes**, no crash. It had never
been done before this session, and six real bugs stood between it and working.

**Nothing is running. `workersMax = 0`, `workersMin = 0`, zero workers.**

**WHO OWNS WHAT (2026-08-29).** This handoff is for the COMPUTE thread: cost,
wall clock, and getting shots to run at all. **Naming is a separate session** —
DJ is building five-on-court exclusion there (§6). Do not start naming work
here; do read §6, because it decides what compute is worth spending.

---

## 1. What a worker actually is [MEASURED 2026-08-23]

Read with `spikes/exp_machine.py` through `lab.py` — no rebuild needed.

| | value |
|---|---|
| container memory limit (cgroup) | **62 GB** |
| host RAM / available | 528 GB / 411 GB |
| **CPUs** | **128** |
| container disk | **32.2 GB** (this is the tight one) |
| GPU | RTX 4090, 25.4 GB VRAM |
| volume | ~1.3 PB free |

**">=3.85 GB of worker RAM" was never a limit** — it was the high-water mark of a
run that happened to survive, and it was wrong by 16x. A day of planned memory
work was cancelled by one 5-cent job. Read the machine before sizing anything.

**Container disk (32.2 GB) is now the real constraint**, and the ball layer
would blow it — see §5.

---

## 2. The full-game run [MEASURED 2026-08-23]

Slices 8 and 9 (the only ones never computed) then the merge over all ten.

| stage | seconds | peak GB |
|---|---|---|
| slice 8 / slice 9 | 1,987 / 2,068 (37.6 min wall, 2 workers) | — |
| merge 10 slices -> 171,120 frames | ~50 | 1.15 |
| team_events | 113 | 2.03 |
| windows | 27 | 5.15 |
| seed_queue | 71 | 8.38 |
| player_events | 97 | 8.40 |
| ocr_confirm (reader stubbed) | 184 | 8.40 |
| retro_merge / box_score | 39 / 34 | 10.77 |
| **merge job total** | **5,745 (96.5 min)** | **10.77** |

Merged caches: tracks 564.5 MB, on-court 288.3 MB (8 slices) [MEASURED].

**Spend this session ~$4.38** [ESTIMATE from measured job seconds; RunPod's
GraphQL rejects this API key, so confirm in the console]. Cumulative ~$35.70.
Two thirds bought the game; one third bought the bugs.

**The output was empty: 0 players named.** That is a naming problem, not a
compute one — see §6.

---

## 3. Six bugs, all of which only bite on a full game

1. **Black opening frames crashed `team_events`.** The film opens with ~15 pure
   black frames (brightness 0.0, zero SIFT keypoints) [MEASURED]; `event_frames`
   starts at the span start, and `H_court @ T` was composed without checking T.
   It would have died AFTER ten slices and a merge were paid for. Fixed: skip a
   frame that cannot be placed, the rule `oncourt.build` always used. Confirmed
   on the real slice 0 on the volume, which already records those frames with
   `"anchor": null`.
2. **The merge swallowed a corrupted slice.** The header check could not catch a
   slice whose header was right and whose FRAMES were another slice's — exactly
   what warm workers did to six of ten slices once. Tested: it merged silently,
   indices jumping 199 -> 50,000, and downstream stages index BY POSITION, so
   the result is one girl's floor time on another girl. Fixed: one integer
   compare per frame, refuses and names the slice.
3. **The merged cache had no `fps`.** `run_tracking` writes it, the merge did
   not, and the tail reads it. Nobody had hit it because the merge and the tail
   had never run together. Fixed: the merge inherits the first slice's header.
4. **Two stages held whole video frames in dicts.** 6.38 MB per frame
   [MEASURED]; `stage6` held 215 frames (1,372 MB) to look at 356 crops. Fixed:
   cut the crop while the frame is in hand, keep the crop (3.3 MB, 417x less).
   `.copy()` is load-bearing — `jersey_crop` returns a numpy view.
5. **The identity machine was quadratic.** Lost identities sat in a flat list
   scanned for every new body; nothing leaves it except on a relink. 5.1 -> 30.4
   -> 250.0 s per doubling, heading for ~5.7 h inside a 180-min cap. Fixed by
   keying the pool by track id — the only key `_match_lost` ever accepted.
   250 s -> 1.1 s, and PROVEN IDENTICAL (same md5 over a full dump of machine
   state on real tracks and 20,000 synthetic frames).
6. **No timeout on the vision client.** A request hung for 28 minutes with no
   error [MEASURED]. One hang burns a whole job to the 180-min cap, after the
   slices are paid for. Fixed: 90 s.

Plus `lab.py`'s credential regex `[A-Z_]+` could not match `RUNPOD_S3_ACCESS_KEY`
(the digit), so the two variables it needs to upload were the two it could never
read — every experiment died at the upload.

**STILL OPEN, not fixed: window 0 seeds nobody.** `stage4` seeds only when the
window CHANGES, and window 0 begins on the black frames where there are no
bodies. Confirmed identities in the whole game start at frame **13,921**
[MEASURED] — the first **7.7 minutes** has no nameable players. Generalises: any
window starting on a moment where the tracker sees nobody gets no seeds at all.

---

## 4. Speed work: done, and left

**Done and proven identical**
- `run_tracking.extract_subclip` now SEEKS instead of winding from frame 0.
  Slice 9 wound through 154,008 frames it threw away — 172 s [MEASURED]. Byte
  identical subclip (same sha256), 538x faster skip.
- Tail scaling after the fixes [MEASURED]: 102 / 205 / 400 s and 1.23 / 2.17 /
  3.71 GB at 1 / 2 / 4 slices. Linear.
- EasyOCR pool 6 -> 32 threads: 0.0357 -> 0.0250 s/crop [MEASURED]. 96 threads
  is no better — the interpreter's lock, not the hardware.
- Results now published to `/runpod-volume/results/<clip>/` — stills and JSON
  used to die with the container.

**Measured but NOT built**
- ~~Compact JSON caches~~ — **DONE 2026-09-03.** tracks 122.3 -> 56.0 MB a slice,
  on-court 77.5 -> 25.9 MB, parsed content identical.
- ~~Load each merged cache once~~ — **PARTLY DONE 2026-09-03.** `oncourt.load_checked`
  is memoised by (path, mtime, size), so a rebuilt cache is a different key and
  can never be served stale. The tracks cache is still parsed per stage: each one
  builds its own Track objects from it, so sharing needs more care than the
  on-court document did.
- **~~Batch the GPU SIFT~~ — NOT AVAILABLE, measured 2026-09-03.** kornia's
  `ScaleSpaceDetector` hard-asserts a batch of exactly 1
  (`KORNIA_CHECK_SHAPE(img, ["1", "C", "H", "W"])`), so the detector refuses
  batches at the front door. The ~70 ms of batch-size-1 GPU work is real and
  there is no batching lever on it. `spikes/exp_sift_batch.py` records this.
  **The anchor is now effectively a fixed 0.086 s/frame**: the resolution sweep
  is exhausted, subsampling was rejected on accuracy, batching is refused.
- **The concurrency cap is 10** [MEASURED, the API refused 20]. Per-second
  billing means 25 workers cost the SAME and finish in 40% of the time. BUT
  finer TRACKING slices add seams and change the box score; the ANCHOR is
  stateless per frame and can be split freely. Decouple them before asking.

**Operational notes**
- After raising `workersMax`, the endpoint refuses jobs for ~25 s with
  `409 ENDPOINT_PAUSED`. Retry, do not panic.
- Always restore `workersMax = 0` in a `finally` block.
- `spikes/exp_tail_real.py` is the dress rehearsal: merge + tail over slices
  already paid for, reader stubbed and COUNTED. Use it before spending.

---

## 5. Shots CAN run now (was: cannot) — updated 2026-09-03

The two blockers in this section are gone. Neither was fixed the way I first
proposed, and both wrong turns are recorded below so nobody retakes them.

**THE RIM IS NOW CARRIED ONLY WHERE AN ARC IS.** Tracing where the rim is
actually read: arcs are built from ball detections alone and never look at it;
`shot_attempts` consults it inside `classify_shot`, per arc segment; nothing else
touches it but the debug overlay. Carrying it across the whole span computed a
rim for every frame where nothing would ever ask — MEASURED 0.412 s/frame, 19.6
hours a game. `run_clip` now builds arcs FIRST and carries the rim only at frames
an arc touches (padded 15). PROVEN on TEST1's human-verified shots: same
detections, same arcs, rim carried both ways, attempts IDENTICAL. A partial track
records `only_frames` and `hoop_track_covers` refuses to reuse it as if it
covered the span.

**BALL DETECTION MOVED INTO THE SLICES.** It is per-frame and stateless, so it
splits like tracking; it was ~43 minutes of one machine in the tail. Each slice
now detects its own frames and the merge glues the logs into the exact file
`stage_ball_detect` writes, so the tail's fingerprint sees a log covering the
span and reuses it. Arcs stay whole-game — a shot can cross a slice boundary.
A partial ball log is REFUSED rather than glued (a hole loses every shot in it).
`tests/test_ball_chunk_merge.py` guards this, and it caught the first version:
the merged log carried no model/imgsz/conf, so the tail would have silently
re-detected all 171,120 frames every run.

**THE THREE DEBUG VIDEOS ARE GATED.** Ball, arcs and shot attempts each rendered
a full-span overlay, and each called `extract_subclip` first — so rendering one
re-encoded the whole span before drawing a line. [ESTIMATE] ~48 GB into a
MEASURED 32.2 GB of container disk: not slow, out of room.
`ball_stages.overlays_wanted` skips past 3,000 frames; `CV_BALL_OVERLAYS=1`
forces them back.

**TWO WRONG TURNS, recorded so they are not retaken:**
- **A GPU rim matcher is SLOWER and moves the rim.** `gpu_anchor.GpuMultiAnchor`
  exists, unused, behind its flag. MEASURED 1.03 s/frame vs 0.412, and it placed
  the rim ~1,800 px away: kornia's matcher finds ~5x more inliers than FLANN, so
  "keep the most inliers" ranks the keyframes differently — 39 of 40 frames chose
  a different keyframe.
- **"~2,200 hours" for the CPU rim was wrong.** That was the COURT anchor's
  47 s/frame at 4,000 features; the rim uses 1,500. It was 19.6 hours.

### What shots cost now
[ESTIMATE] rim ~4 min and ball ~4 min per slice, in parallel with work already
happening. Shots stopped being a separate problem.

---

## 5b. The original blocker text (kept for context)

`clips/Full_Game_9eb8bf2a.json` now carries **both rims** (near frame 166,842,
far 154,008) and a whole-game ball span, so `run_clip` WILL fire the shot layer.
It must be cleared for box-score runs (`ball_span_len = 0`) until:

1. **The rim tracker is still CPU SIFT against ALL five keyframes, per frame.**
   The single-keyframe CPU anchor measured 47-49 s/frame on a worker [MEASURED],
   so a full game is [ESTIMATE] ~2,200 hours. This is the blocker.
2. **Three full-game overlay videos** (ball detect, arcs, shot attempts) plus a
   full-game re-encode: [ESTIMATE] ~48 GB into **32.2 GB** of container disk.
3. It all runs in the single-machine tail, though ball detection and rim
   tracking are both per-frame and split as cleanly as tracking does.

`spikes/hoop_anchor.py` now accepts a rim marked on ANY frame (SIFT-matched to
its nearest keyframe, refuses a weak match) — needed because the right-hand
basket appears in NONE of the five calibration keyframes.
`spikes/make_rim_clicker.py` builds the browser page that produced them.

[ESTIMATE] once the rim tracker is on the GPU and both stages are chunked:
~34 min across 10 workers for the rim, ~4 min for ball detection, ~$8.

---

## 6. Why the run named nobody — read before touching the reader

Not a compute problem, but it decides what compute is worth spending.

- 23,288 candidates, **0 confident reads**, 0 named [MEASURED].
- Clicks to name them by hand = windows x people on court = 147 x 13.9 =
  **~2,043** [MEASURED]. Ranking by floor time does NOT collapse this: 30 clicks
  covers 11%, 400 covers 60%.
- Even naming all of them covers **232 player-minutes, ~24% of the game** —
  everyone who arrives mid-window is never seeded [MEASURED].
- **Every automatic route is measured-and-closed**: appearance re-ID made
  fragmentation WORSE (`DECISIONS` §11, 122 -> 131 ids — teammates wear identical
  uniforms); cross-track-id position relinks ~70% wrong; the resolution sweep is
  exhausted.
- **Better footage is CLOSED TOO** (DJ, 2026-08-29): coaches use Hudl, the test
  film is Hudl, "if it can't work on that it just can't work". `DECISIONS` §11
  ranked footage as remaining lever #1 — that ranking is now void.
- The reader is also **not reproducible** on these crops: the same 120 read twice
  gave 13 names then 2, with ONE agreeing [MEASURED].
- The cause is ANGLE, not size. Crops binned by size found nothing reproducible
  at any size, and a montage of real crops is backs, side-ons, arms and a
  referee next to a perfectly crisp 23. **30% of players had all ten read
  attempts inside one second** [MEASURED] — ten pictures of one pose. Fixed:
  attempts now spread across her whole time (1.6 s -> 3.9 s median span, same
  cost).

**Gemma cost, if it is ever switched on** [MEASURED call counts]: ~698,000 calls
a game today; ~256,000 with the first-read gate (shipped, proven decision-
identical over all 64 read sequences); ~21,000 with 12-crop sheets (built,
tested, NOT wired in — needs an accuracy diff first). Dollar cost [UNKNOWN] —
rate-limiting rather than billing errors suggests a free tier; confirm in Google
AI Studio.

**FIVE-ON-COURT EXCLUSION IS CLAIMED — DO NOT BUILD IT HERE.** DJ is building it
in a separate naming session (2026-08-29). Recorded so two threads do not write
the same thing twice, and so this one stays on compute.

What it is, for context only: four of five confirmed => the fifth is determined
by the roster, by LOGIC rather than similarity, so identical uniforms do not
defeat it — which is why it survives where appearance re-ID died. Never tried
(grepped both DECISIONS files). `stage7` already has the "one girl cannot be in
two places" check, built to REFUSE a merge; exclusion is the same constraint used
to CONCLUDE one. If it lands, it changes what the numbers in §6 mean, so re-read
them rather than re-deriving them.

`spikes/make_reseed_sheet.py` builds the ranked naming page: one card per
identity, six crops across her time, floor seconds, what the reader tried, and a
live coverage bar.

---

## 7. Rules that earned their place this session

1. **Read `DECISIONS.md` before proposing anything.** Appearance re-ID was
   recommended here as the top lever; it had been measured and refuted six weeks
   earlier, in a file written to prevent exactly that.
2. **Check a measurement can distinguish the answers before spending on it.** A
   sweep scored "agrees 36/36" both when two readers named three girls and when
   they named nobody.
3. **Never use `phase2/out` as scratch.** It is the pipeline's working directory;
   a fetched cache was overwritten mid-experiment by an unrelated run.
4. **Look at the pictures.** Thirty seconds of eyeballing a crop montage
   corrected two conclusions that four measurements had not.
5. **Rehearse on data already paid for.** `exp_tail_real.py` over slices 0-7
   found three of the six bugs for about 10 cents.


---

## 8. Where the clock stands (2026-09-03)

| | measured / estimated |
|---|---|
| parallel phase (tracking + anchor + ball + rim) | ~35 min at 10 workers |
| identity tail, skeleton | ~13 min |
| shot maths (arcs, attempts, outcomes) | ~3 min |
| jersey reading | set by NAMING, not by compute — see §6 |
| **everything but the reader** | **~48 min** |

**The only compute lever left that matters is WORKER COUNT.** The anchor is
~24 min of every slice and is now fixed (§4). It is stateless per frame, so it
splits as finely as you like with NO seams — unlike tracking, where every extra
slice is another boundary that splits a player in two. Per-second billing means
20 workers cost the SAME as 10.

- cap is 10 [MEASURED — the API refused 20]
- raising it is a support request, not engineering
- at 20 workers the parallel phase halves: **~30 min all in**

That is DJ's stated limit, with everything on except the reader. Under 30 needs
the reader to be small, which is exclusion's job in the naming session.
