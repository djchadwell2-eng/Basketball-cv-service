# Handoff — compute, cost and speed (session ending 2026-08-29)

Same tagging rule as `HANDOFF_GPU_SESSION.md`, for the same reason:

- **[MEASURED]** — came out of a real run. Trust it.
- **[ESTIMATE]** — arithmetic on measured numbers. Could be wrong.
- **[UNKNOWN]** — never established. Do not act as if it is.

**THE HEADLINE: a full 95-minute game ran end to end for the first time.**
171,120 frames, merge + identity tail in **96.5 minutes**, no crash. It had never
been done before this session, and six real bugs stood between it and working.

**Nothing is running. `workersMax = 0`, `workersMin = 0`, zero workers.**

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
- **Compact JSON caches.** tracks 122.3 -> 56.0 MB per slice, on-court 77.5 ->
  25.9 MB [MEASURED], parsed content identical. ~1.2 GB per game of volume
  traffic that is literally spaces.
- **Load each merged cache once.** The tail parses tracks **9x** and on-court
  **8x**; ~19 s and ~8 s each at full size => ~4 min [MEASURED, scaled]. Needs a
  check that no stage mutates the shared doc.
- **Batch the GPU SIFT.** Of the 0.086 s/frame anchor, ~5 ms is decode, ~2.2 ms
  RANSAC at 4,000 points [MEASURED] — so ~70 ms is GPU work at batch size 1 on a
  card built for batches. [ESTIMATE] a 2x there is $2.20 and ~10 min off every
  slice. Compare in FEET, not pixels.
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

## 5. Shots still cannot run

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

**NEXT NAMING IDEA, never tried** (grepped both DECISIONS files): **five-on-court
exclusion.** Four of five confirmed => the fifth is determined by the roster, by
logic rather than similarity, so identical uniforms do not defeat it. `stage7`
already has the "one girl cannot be in two places" check, used defensively.

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
