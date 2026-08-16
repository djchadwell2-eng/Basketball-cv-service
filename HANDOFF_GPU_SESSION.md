# Handoff — GPU / speed / web-app session

Everything below is from one working session. **Each claim is tagged:**

- **[MEASURED]** — a number that came out of a real run. Trust it.
- **[ESTIMATE]** — arithmetic on top of measured numbers. Could be wrong.
- **[UNKNOWN]** — not established. Do not act as if it is.

The reason for the tags: in this session I stated several things confidently
that were wrong, and each one cost DJ money or hours. The tags are the fix.

---

## 1. Current state of the system (as of the end of this session)

**Nothing is running. Spend is stopped.**

| Thing | Value | How known |
|---|---|---|
| RunPod endpoint | `nhqffi8lp2esit` | API |
| Template | `u0uo4v0z9q` | API |
| **workersMax** | **0 — nothing can run until raised** | API |
| workersStandby | **1 — bills around the clock, CONSOLE-ONLY, DJ must set to 0** | API (read-only field) |
| GPU types allowed | RTX 4090 only | API |
| Job time cap | 180 min | API |
| Account worker quota | 10 | API refused 20 |
| Network volume | `r5hc0v2j7v` in **US-NC-1**, 20 GB | API |
| Film on volume | `films/Full_Game_9eb8bf2a.mp4`, 3.62 GB | S3 listing |
| **Total spent, all days** | **$31.30** | billing API |

### The film and the game
`Full_Game_9eb8bf2a` = 171,120 frames, 95.1 min, 1080p30. Calibration was
approved by DJ at **0.185 ft mean / 0.442 ft max**, 5 keyframes
(600, 127200, 151200, 158700, 171000), court solved as 84 ft.

### What is already paid for and sitting on the volume
**Slices 0–7 of 10** — `chunks/Full_Game_9eb8bf2a/{000..007}_{tracks,oncourt}.json`,
~1.75 GB. **[MEASURED]** every header verified to cover its exact span; slice 0
downloaded and parsed in full: 17,112 frames, people in 17,105 of them, 5,112
distinct tracks. That is **frames 0–136,895 = 76.1 minutes** of the game,
tracked and court-mapped. **Missing: slices 8 and 9 (last 19 minutes).**

---

## 2. The measured performance picture

### Per-frame costs
| stage | hardware | s/frame | how |
|---|---|---|---|
| YOLOv8m@1280 detection | DJ's laptop (no CUDA) | **1.44** | [MEASURED] 5 frames |
| YOLOv8m@1280 detection | RTX 4090 | **0.011** | [MEASURED] speedtest job |
| SIFT camera anchor | laptop, TEST1 | **3.35** | [MEASURED] 8 frames |
| SIFT camera anchor | worker, full game | **47–49** | [MEASURED] 3–4 frames |
| **GPU anchor (kornia)** | 4090, first version | 2.405 | [MEASURED] |
| **GPU anchor** | after describing keyframes once | 0.588 | [MEASURED] |
| **GPU anchor** | after the iter_frames seek fix | **0.081–0.086** | [MEASURED] 60 frames × 2 spots |

**The GPU anchor agrees with the CPU one: 0.008 ft mean, 0.11 ft max, 0 failed
frames** [MEASURED, 30 frames at frame 600]. For scale DJ calls 0.21 ft "utter
perfection" and 0.94 ft "broken". This is not an accuracy trade.

### Fixes that produced those numbers
1. **`spikes/stage2_multikeyframe.iter_frames` now seeks** instead of grabbing
   from frame 0. **[MEASURED] 55.1 s → 0.3 s (199×) and pixel-identical** on
   frames near 60,000. `extract_frames` already did this via `fast_frames`;
   this iterator — the one the per-frame stages actually use — never got it.
2. **`gpu_anchor.py`** — kornia SIFT on the GPU, wired in at
   `phase1/stage1_court_roi.build_court_anchor` so every caller benefits.
   Falls back to CPU when there is no CUDA, so the laptop and the 395 tests
   are unaffected. `CV_GPU_ANCHOR=0` forces the old path.
3. **`phase2/stage4_seed_queue`** stopped rewinding the film to frame 0 for
   every debug still. **[MEASURED] 68.9 s → 0.4 s.**
4. **Streamed merge** (`serverless_handler.merge_streamed`) holds one slice at
   a time. **[MEASURED] on DJ's real slices: 3 slices 0.11 GB, 8 slices
   0.12 GB peak — flat.** Merged all 8 (136,896 frames) in 47 s, output
   verified ordered, no duplicates, track ids namespaced per slice.

### The identity tail (runs once, on one machine, after the parallel work)
[MEASURED] on 300 frames: total **86.4 s**, of which seed_queue was 68.9 s
(now 0.4 s) and ocr_confirm 11.7 s. The other six stages together: 0.7 s.

**With DJ's Gemma jersey reader enabled, ocr_confirm went 11.7 s → 650.8 s**
[MEASURED]. It makes one API call at a time (3 reads × 2 crops per candidate).
The calls are independent, so concurrency is the obvious fix — untried.

### Anchor subsampling (checking every Nth frame instead of every frame)
[MEASURED] on the real game, GPU anchor as reference, 60 frames/spot:

| spot | N=2 | N=5 | N=10 | N=30 |
|---|---|---|---|---|
| minute 0.3 (camera settled) mean/max ft | 0.008 / 0.05 | 0.007 / 0.07 | 0.006 / 0.07 | 0.007 / 0.07 |
| minute 33 (camera roaming) mean/max ft | 0.101 / 0.76 | 0.105 / 0.84 | 0.102 / 0.78 | 0.127 / 0.97 |

**The sharp bit: even N=2 costs 0.755 ft at minute 33.** That error is the
anchor's own frame-to-frame jitter, not the skipping. Skipping more barely
makes it worse. **Conclusion: subsampling is not needed** — anchoring every
frame is ~4.1 GPU-hours [ESTIMATE from 0.086 s/frame], and it is the only
option that costs no accuracy.

---

## 3. Web app work

- **`/analyze` IS the new-game page now.** Rosters (01) + film (02) feeding
  `/api/cv-setup`. The old Gemini-only upload is gone from the UI (its API and
  `AnalysisTabs` remain, serving `/history/[id]` for old games). `/setup/new`
  redirects to `/analyze`. Two upload doors became one.
- **Approving the court starts the analysis automatically** and lands the coach
  on `/history/<videoId>` — the real tabbed UI, not the raw `/measured/<clip>`.
- **`registerGame()`** in `/api/cv-setup` inserts the Supabase `videos` row and
  links videoId ↔ clip. Without it `/history/<id>` 404'd, which is why browser
  games never appeared.
- **`/api/cv-setup/[clip]/film`** streams the film off disk with byte ranges
  (Supabase refuses a 3.4 GB file — [MEASURED] 413 on an 87 MB test).
- **`/history/[id]` renders with CV data and no AI pass.** It used to 404
  unless a Gemini `game_patterns` row existed.
- **"Name these players" card** in `MeasuredStats` → `/reseed/<clip>`. That
  screen and its API existed all along; nothing ever pointed at them, which is
  why DJ saw "0% identified" with no way to act.
- **`lib/measuredStory.ts` + `/api/measured/[clip]/story`** — THE QUICK PASS.
  Gemini reads the measured numbers and returns JSON shaped for the tabs
  (identity, tendencies, a line per possession, observations, not_measured).
  [MEASURED] on TEST1: cited real numbers, refused to state a pace, and only
  spoke about a made basket because `make_miss_available` was true and that
  shot carried a measured score change.
- **`possessions` added to the app's contract type** — the CV side had been
  writing them all along; the app simply did not read them.

**Still not wired:** the tab components themselves (Film Room reading
possessions, Tendencies reading tendencies). The data and the story now exist;
the rendering does not.

**Also not wired:** the app still sends ONE job to ONE machine. A browser-run
full game would be ~7 h and hit the cap. The 10-way split lives only in
`run_chunked.py`, driven from a terminal.

---

## 4. Mistakes made this session — read this part twice

Each of these was stated confidently and was wrong.

1. **Ran `npm run build` while the dev server was up.** Wiped the dev server's
   files → 404s on every asset. DJ thought his app was broken.
2. **Claimed the camera anchor runs 3× and sharing it is free.** An Opus review
   checked the code: it is 2×, and the third caller (`hoop_anchor`) matches
   EVERY keyframe rather than the nearest, so "just share the cache" would have
   silently moved the rim and changed every shot outcome.
3. **Said the credit balance was not the problem.** It was: the endpoint later
   returned `402 Payment Required`. Hours were spent chasing datacenters.
4. **Quoted "$1.50–2 per full game".** That is the price of ONE GPU-hour. A
   full game is ten machines for ~30 min ≈ **5+ GPU-hours ≈ $6–9**.
5. **Widened GPU types to H200/H100/L40S** while hunting capacity and left them
   on. [MEASURED] rate went **$1.33/GPU-h → $3.03/GPU-h**. Now 4090 only.
6. **Launched all 10 slices before proving one.** 17 jobs failed; ~$10.50 spent
   with no result delivered.
7. **Killed all workers mid-run** to stop spend — which is what FAILED the 8
   in-flight slices. Their "FAILED" status was me, not a bug.
8. **Polled a job status by submitting jobs.** Queued 49 dead jobs that blocked
   the endpoint.
9. **Told DJ "nothing came back" from the $10.50.** Wrong — 8 of 10 slices had
   been saved to the volume and are fully valid.

---

## 5. Open risks, honestly rated

| risk | status |
|---|---|
| Downstream stages load the whole merged cache | **[MEASURED] 2.39 GB for 8 slices; [ESTIMATE] ~7.5 GB for 10 + on-court + models** |
| Worker RAM | **[UNKNOWN]**. Only ≥3.85 GB is proven (a tail run peaked there and lived). API does not report it. |
| Identity tail at full-game scale | **[UNKNOWN]** — never run past 300 frames |
| Gemma reader at game scale | **[MEASURED] 55× slower than EasyOCR**, sequential API calls; currently DISABLED on the worker (key removed from template env) |
| RunPod capacity | **[MEASURED] variable.** US-IL-1 had none for hours; US-NC-1 answered instantly |
| Out-of-bounds detection | never fired on real film (other session's note) |

---

## 6. Rules for the next session

1. **Tag every claim** [MEASURED] / [ESTIMATE] / [UNKNOWN]. If it is not
   measured, say so in the same sentence.
2. **Never spend before proving one unit.** One slice (~$0.61) before ten.
3. **Read spend before and after every run** and report it unprompted.
4. **Never poll by submitting jobs.** `/health` is a free read.
5. **Never kill workers mid-run** without saying what it will destroy.
6. **Check the deployed image sha** (`{"mode":"version"}`) before trusting any
   result. Stale workers have wasted 44 minutes and produced fake errors.
7. **Never run a production build while the dev server is up.**
8. **A file that only exists on the laptop does not exist in the cloud.** Three
   separate outages came from uncommitted modules (`clip_registry`,
   `fast_frames`, `possessions`).

---

## 7. The cheapest next step

Slices 0–7 are permanent. To finish this game:

1. Raise `workersMax` above 0 (currently 0).
2. Run **slice 8 alone** — [ESTIMATE] ~$0.61, ~30 min.
3. Then slice 9.
4. Then the merge — now streamed, [MEASURED] 0.12 GB peak — followed by the
   identity tail, which is the [UNKNOWN] part.

Do not run 2–4 as one command. Each step should report before the next starts.
