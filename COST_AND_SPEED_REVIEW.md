# Cost & speed review — one 95-minute game on RunPod serverless

Written 2026-08-19. Companion to `HANDOFF_GPU_SESSION.md`, whose tags are kept.

**Nothing here was run in the cloud. No money was spent producing it.** Every
new number came from reading the code or measuring locally against the real
film (`uploads/1785535735594_88mw2l.mp4`).

Tags as in the handoff: **[MEASURED]** came out of a real run · **[ESTIMATE]**
is arithmetic on top of a measured number · **[UNKNOWN]** was never established.

Full report, with the arithmetic laid out:
https://claude.ai/code/artifact/6b58e348-e4c7-4abd-8665-721d00ca9b47

---

## New measurements taken for this review (laptop, real film)

| what | value | how |
|---|---|---|
| sequential `grab()` (skip a frame) | **1.51 ms/frame** | [MEASURED] 300 frames |
| full decode `read()` | **4.89 ms/frame** | [MEASURED] 300 frames |
| seek to frame 600 / 60,000 / 153,996 | 0.17 / 0.12 / 0.07 s, **landed exactly** | [MEASURED] |
| decode + mp4v **re-encode** (`extract_subclip`) | **29.1 ms/frame**, 70.3 kB/frame | [MEASURED] 300 frames |
| mp4v copy vs source pixels | **mean 3.0 levels, max 76, 0/24 frames identical** | [MEASURED] |
| YOLO on source vs mp4v copy | **loses 1–3 people/frame**, boxes shift up to 39 px | [MEASURED] 6 frames |
| tracks cache, one slice, `indent=2` vs compact | **122.3 → 56.0 MB** (2.18×), parsed content identical | [MEASURED] |
| on-court cache, one slice | **77.5 → 25.9 MB** (2.99×) | [MEASURED] |
| merged tracks cache parse time (whole game) | **~19 s** ; on-court **~8 s** | [MEASURED, scaled] |
| RAM per held video frame | **6.38 MB** | [MEASURED] 40 frames |
| `cv2.findHomography` RANSAC, 4,000 pts | **2.21 ms** | [MEASURED] |
| calibration refit (`_setup` + `_solve`) | **10.9 s**, reproduces 0.185 / 0.442 ft | [MEASURED] |

---

## 1. Where the money goes, per game

171,120 frames, 10 slices, RTX 4090 at $1.33/GPU-hour.

| line | arithmetic | GPU-h | $ | tag |
|---|---|---:|---:|---|
| camera anchor (incl. its own decode) | 171,120 × 0.086 s = 14,716 s | 4.09 | 5.44 | rate [MEASURED], total [ESTIMATE] |
| person detection | 171,120 × 0.011 s = 1,882 s | 0.52 | 0.70 | rate [MEASURED], total [ESTIMATE] |
| re-encoding each slice to a temp mp4 | 171,120 × 29.1 ms = 4,980 s | 1.38 | 1.84 | [ESTIMATE] |
| winding the film forward from frame 0 | 770,040 × 1.51 ms = 1,163 s | 0.32 | 0.43 | [ESTIMATE] |
| re-decoding that temp mp4 for the tracker | 171,120 × 4.89 ms = 837 s | 0.23 | 0.31 | [ESTIMATE] |
| merging ten slices | 47 s for 8 → ~60 s | 0.02 | 0.02 | [MEASURED] |
| **compute that has a number** | ≈ 39 min/slice | **6.56** | **8.73** | [ESTIMATE] |
| identity tail (one worker, whole game) | never run past 300 frames | — | — | **[UNKNOWN]** |
| cold starts (10 × 3.9 GB pull) | pull never timed | — | — | **[UNKNOWN]** |
| standby worker (`workersStandby=1`) | 24 GPU-h/day × $1.33 | 24/day | 31.92/day | [ESTIMATE] |
| volume storage (20 GB) | RunPod per-GB rate not established | — | — | **[UNKNOWN]** |
| Gemma jersey reads | bills as GPU time, not as an API invoice | — | — | **[UNKNOWN]** |

**Reality check on my own table.** The three plumbing lines use laptop CPU rates
applied to a worker nobody has profiled, reading a film off a network volume. My
total says 39 min/slice; the handoff's independent estimate says ~30. So those
lines are likely 30–50% high on real hardware. Call the compute bill
**$6.80–$8.73**, which is where the handoff's corrected $6–9 already sat.

**The standby line is dormant, not free.** `workersMax` is 0, so nothing can run
and standby cannot bill. Total spend of $31.30 across all days is itself the
evidence that it has NOT been charging a full GPU-hour rate around the clock. It
wakes up the moment `workersMax` goes above 0.

---

## 2. Savings, ranked by value ÷ (risk × effort)

At $1.33/GPU-h, $1 buys 45 GPU-minutes. Almost every proof below is a two-minute
warm job (~$0.05) or free and local.

1. **Stop holding whole video frames in dictionaries.** `stage4_seed_queue`
   holds one frame per identity window; `stage6_ocr_confirm` holds one per OCR
   attempt. At 6.38 MB/frame [MEASURED]: stage 4 = **3.63 GB resident** at ~570
   windows (HARD's rate of ~1 window / 10 s over 95 min); stage 6 = **38 GB** at
   2,000 candidates × 3 crops, **255 GB** at 4,000 × 10. Only ≥3.85 GB of worker
   RAM has ever been proven. *Risk:* none — same frames, same order, fewer
   resident. *Proof (~$0.03):* merge slices 0–1 (already paid for; `merge_chunks`
   already takes an explicit chunk list) and run the tail on 34,224 frames.
   **This is not a speed-up, it is the run.** As written the tail cannot finish a
   full game; it runs out of memory before it runs out of time.

2. **`workersStandby` → 0.** Up to $31.92/day [ESTIMATE] once `workersMax` > 0.
   Console-only field. *Risk:* none — it buys an unmeasured cold-start saving at
   a measured price. *Proof:* free, read billing before and 24 h after.

3. **Write the caches compact instead of `indent=2`.** Whole game
   **1.99 GB → 0.82 GB** [MEASURED]; dump 14.9 s → 9.6 s per slice. *Risk:* none
   — parsed content asserted identical. *Proof:* already run locally; repeat once
   against a real slice off the volume.

4. **Seek to the slice instead of winding from frame 0.**
   `run_tracking.extract_subclip` still does `for _ in range(start): cap.grab()`
   — it never got the fix `iter_frames` and `fast_frames` got. 770,040 wasted
   frames × 1.51 ms = **0.32 GPU-h, $0.43**, and **3.9 min off slice 9** (the
   slice that gates the merge), plus ~16 GB of pointless network-volume reads.
   *Risk:* none using the existing seek-then-verify helper, already proven
   pixel-identical on this file. *Proof:* free and local.

5. **Load each merged cache once, not nine times.** The tail parses the merged
   tracks file **9×** and the on-court file **8×** (three stages load both again
   inside `window_boundaries.load_windows`). At 19 s and 8 s that is **235 s ≈
   3.9 min** [MEASURED, scaled]. *Risk:* low-medium — a shared doc can be mutated
   by one stage and seen by the next; hand out a copy or verify read-only use.
   *Proof:* free — run the tail twice on the 300-frame span and byte-diff outputs.

6. **Make the three Gemma reads of one crop concurrent.** `_read_gemma` does its
   three votes sequentially inside a 6-wide pool, so only 6 calls are ever in
   flight; fanning out takes it to 18 without touching crop selection, attempt
   budget, agreement denominator or the 0.85 threshold. Up to **3×** off the
   tail's dominant cost [ESTIMATE] — less in practice, since 16-wide was
   [MEASURED] to be rate-limited. *Risk:* "identical" is unprovable for any
   concurrency because the reader is already nondeterministic (that is why it
   reads 3×); what is provable is that the algorithm is untouched.
   *Proof (~$0.05):* stage 6 alone on TEST1 at 6-wide and 18-wide, 3 runs each.

7. **Batch the SIFT on the GPU.** Of the [MEASURED] 0.086 s/frame, ~5 ms is
   decode, ~2.2 ms is RANSAC, a few ms is host transfer — **~70 ms is GPU work at
   batch size 1** on a card built for batches. A 2× there takes the anchor from
   4.09 to ~2.4 GPU-h: **$2.20 and ~10 min off every slice** [ESTIMATE]. Largest
   remaining lever that costs no accuracy. *Risk:* real but bounded — batched
   kernels can differ in the last bits and flip a match. *Proof (~$0.05):* one
   `lab.py` exec, 60 frames batched vs not, compared **in feet**, against the
   0.008 ft bar the GPU anchor already cleared.

8. **Split the anchor finer than the tracking, then ask for more workers.**
   Tracking is stateful — every extra slice is another seam and a seam splits a
   player into two identities. The anchor is stateless per frame and has no such
   cost, and its per-frame output is a 3×3 matrix. Decoupled, the anchor's
   4.09 GPU-h drops from **24.5 min at 10 workers to 9.8 min at 25** for the same
   dollars (per-second billing). The cap of 10 is [MEASURED] (the API refused 20);
   raising it is a support request, not a purchase. *Risk:* none while tracking
   stays at ten slices; finer *tracking* slices would change the box score.

9. **Run the tail somewhere that is not a 4090.** It is JSON, OpenCV crops and
   HTTP waits; its only GPU work is 40 frames of YOLO in `stage2_generate_events`
   (0.4 s GPU vs ~58 s CPU). *Proof:* don't build it yet — step 3 below hands you
   the tail's real duration for free. Move it only if that is over ~20 min.

10. **Ship the calibration `.npz` on the volume.** 10.9 s per worker [MEASURED],
    11 jobs ≈ 2 min aggregate. The real value is that every worker then provably
    uses the same court instead of eleven solves that happen to agree.

11. **Trim ~350 MB of unused weights from the image.** `yolov8x.pt` (137 MB),
    `yolo11x-pose.pt` (118 MB), `yolov8n-pose.pt` (6.8 MB) sit at the repo root
    where `.dockerignore`'s `models/*` rule does not reach them; nothing on the
    full-game path loads them (only `yolov8m.pt`). ~9% off a 3.9 GB image.
    Cold-start saving [UNKNOWN] — the pull was never timed.

12. **Delete the temp-mp4 round trip entirely — CHANGES THE ANSWER.**
    **1.61 GPU-h, ~$2.15, ~10 min off every slice** — the biggest single compute
    win. But [MEASURED] today: the mp4v copy the detector actually sees differs
    from the source by mean 3.0 grey levels / max 76 on 24 of 24 frames, and YOLO
    on the copy **loses 1–3 people per frame** with boxes shifted up to 39 px. So
    this is a fidelity *fix* that happens to be free — the source frames are the
    ground truth. It should make the box score better. It still moves numbers DJ
    has already looked at, so it is his call, not mine. *Proof:* free and local —
    build the tracks cache both ways on 500 frames and diff.

---

## 3. Structurally wasteful, not merely slow

- **The film is walked four times to be used twice.** Per slice: wind from frame
  0, decode the span, encode the span, decode the encoded span for the tracker,
  decode the span *again* for the anchor. Tracking and anchoring open the same
  file and step over the same frames independently. One pass could feed both —
  and the encode in the middle is not just wasted, it is lossy.
- **The anchor is welded to the tracks and then thrown away.** Most expensive
  thing in the pipeline (4.09 of ~6.5 GPU-h), and its per-frame output is a 3×3
  matrix — a few hundred bytes. Never stored; only the on/off verdict derived
  from it. Any re-run (new weights, re-tracked slice, changed margin) re-pays
  four GPU-hours to re-derive numbers that were already right.
- **Two dictionaries that hold whole video frames** (§2.1). Written when a span
  was 300–600 frames; neither has ever met a game. This is the concrete content
  of the handoff's "identity tail at full-game scale: [UNKNOWN]" — it is not
  slow, it is fatal.
- **The tail parses the same two files seventeen times.** 9 + 8, no memo.
- **Eleven workers each re-solve the same deterministic calibration.**
- **Detection happens twice on the same frames, two different ways.**
  `stage2_generate_events` re-runs YOLO on its 40 sampled frames and re-anchors
  them, when the tracks and on-court caches already cover every one. Only 40
  frames — but two code paths detecting the same bodies can disagree, and nothing
  checks that they don't.
- **Caches written pretty-printed** at 1.2 GB — ~1.17 GB per game of volume
  traffic that is literally spaces.
- **`export_span` is a loaded gun.** It re-encodes the *entire* tracking span to
  mp4: 171,120 × 29.1 ms ≈ **83 minutes and ~12 GB**, then copies it. It costs
  nothing today only because it looks the clip up as a module attribute, fails to
  find `Full_Game_9eb8bf2a_CLIP`, and is swallowed by a `try/except`. The obvious
  tidy-up — switching it to `get_clip` like every other caller — would silently
  add over an hour and 24 GB of disk I/O to every full-game run.
- **Nothing overlaps.** The merge waits for all ten slices, though it folds them
  in start order anyway and could take slice 0 the moment it lands. And nothing
  runs while the jersey reader waits on the network, which is where most of the
  tail's clock will go.

---

## 4. What I would NOT do

Each is a real saving, paid for in accuracy.

- **Subsample the anchor.** Already measured and rejected. Worth restating why:
  even N=2 costs 0.755 ft at minute 33 and N=30 costs 0.974 — the error barely
  grows with skipping because it is the anchor's own jitter. That makes it a very
  tempting trade, and still a court-position trade.
- **`GEMMA_READS` 3 → 1.** Exactly 3× on the tail's biggest cost; deletes the
  only confidence signal the reader has. 2-of-3 = 0.67 under the 0.85 bar is the
  whole mechanism — it refused a real "23" reading [3, 30, 3] and a referee
  reading [13, 10, 10].
- **Drop the second-crop corroboration.** It exists because two of the reader's
  mistakes came back unanimous (a 44 read as 14 three times).
- **Cut `MAX_ATTEMPTS` / raise `OCR_STRIDE`.** Proportional time saving,
  proportional loss in girls named. Naming players is the product.
- **Lower `N_FEATURES` below 4,000 or tighten `LOWE_RATIO`.** Changes which
  points match → changes the homography → moves every player.
- **A smaller detector or a lower `imgsz`.** Same objection, and it is the
  shrinking-images line already drawn.
- **Go back to H100 / L40S.** [MEASURED] $3.03 vs $1.33/GPU-h — needs 2.3× to
  break even, and the anchor is bound by SIFT and matching at batch size 1, not
  raw FLOPs. Fix the batching first, then re-ask.
- **Cache decoded frames as JPEGs on the volume.** Lossy — the same class of
  error as the temp mp4 already in the pipeline.
- **Keep a warm standby to dodge cold starts.** 24 measured hours a day against
  an unmeasured pull. Time the pull first.
- **Launch all ten slices before one is proven.** Not an accuracy trade, a
  history lesson: 17 failed jobs, ~$10.50, and the recovery is the eight slices
  this whole plan leans on.

---

## 5. Cheapest credible path to one completed full-game run

Do NOT run these as one command. Each reports before the next starts.

| step | what | cost |
|---|---|---|
| 0 | Fix it on the laptop: stream the two frame dicts, compact JSON, seek instead of wind. All three provably output-identical. | $0.00 |
| 1 | `workersMax` → 1, `workersStandby` → 0, one `{"mode":"version"}` job to confirm the build. | ~$0.01 |
| 2 | Merge slices **0–1** and run the tail on 34,224 frames. Buys nothing new; answers the one unknown that can lose the game. | ~$0.05 |
| 3 | Merge slices **0–7** and run the tail on 136,896 frames (80% of the game, already paid for). Merge is [MEASURED] 47 s / 0.12 GB peak. Also gives the tail's real duration. | $0.30–0.90 |
| 4 | Slice 8 alone (frames 136,896–153,995). | ~$0.61 |
| 5 | Slice 9 alone (frames 153,996–171,119). | ~$0.61 |
| 6 | The real merge + tail, all ten slices. | $0.30–0.90 |

**Total $1.90–$3.10** [ESTIMATE], with the jersey reader in the state the worker
is actually in today — EasyOCR, since the Gemma key was removed from the template
env. Switching Gemma back on adds [ESTIMATE] 3–14 h of tail at $1.33/h, i.e.
**$4–$19**, which is exactly why the read-concurrency fix belongs before that
switch is flipped.

### What could still go wrong

- **Worker RAM is still [UNKNOWN].** Only ≥3.85 GB proven. Even with the frame
  dicts fixed, the tail holds the merged tracks doc ([MEASURED] 2.39 GB for eight
  slices) plus ~5.7 million `Track` objects built from it. Steps 2 and 3 exist to
  find that wall on data that is already paid for.
- **Container disk, not the volume.** The merged caches land in
  `/app/phase2/out`: ~2 GB pretty, ~0.8 GB compact. Limit never established.
- **The 180-minute job cap applies to the merge job**, which carries the whole
  tail. A slow jersey pass does not fail gracefully; it times out at the end.
- **Slice 9 is unexplored film.** Nothing has decoded to the end of this file.
- **Capacity is not a guarantee.** The volume pins the datacenter; US-IL-1 had
  nothing for hours.
- **What "finished" will and will not contain.** This game's config has no
  `ball_span_len` and no `hoop_anchors`, so `run_clip` skips the entire ball/shot
  layer — deliberately, and it says so out loud. A completed run gives a box
  score, floor time and zones for 95 minutes. It gives **no shots, no shot chart
  and no make/miss**. Not a bug, but worth knowing before the run, not after.
