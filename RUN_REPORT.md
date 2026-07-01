# Full-clip integration + safety diagnostic — HARD.mp4

Read-only run. No Phase 1/Phase 2 internals, schemas, `set_confirmed`, calibration
code, or `read_jersey()` were modified. `OCR_CONFIRM_THRESHOLD` left at 0.85. The
only new code is the thin harness `run_diag_hard.py`. Nothing was patched — where a
seam blocked the run, it is reported here for you to decide the fix.

## Headline
- **Calibration HOLDS across the entire 91 s clip** — every sampled frame gets a
  strong direct-anchor match, sub-pixel, even 1500 frames from the nearest keyframe.
- **The full P1+P2 end-to-end run could NOT be executed on HARD**, for structural
  reasons, not a crash: there is no combined entry point, and every downstream
  component is hardcoded to TEST1's frame ranges / roster / seed-labels / tracks
  cache. Running them on HARD as-is would target the wrong frames and has no HARD
  roster, so I stopped at those seams and report them (per your constraint).

## Setup
- `run_diag_hard.py` selects the HARD clip via the existing `ACTIVE` knob at runtime
  (the one exposed config), then invokes existing functions read-only.
- HARD.mp4 = **2746 frames @ 30 fps = 91.5 s**, keyframes `[600,700,…,1200]`.
- Referenced `phase2/HANDOFF.md` **does not exist** in the repo (only Phase-1 handoff
  JPGs). Read `phase1/DECISIONS.md` + `phase2/DECISIONS.md` instead.

## 1. Completion
- **Calibration path: ran end-to-end on the full clip, no crash.**
- **P1 team-events + heatmap/zones, and all of P2 (tracking → identity → OCR): NOT
  RUN.** Not a crash — they are not wired for HARD (see §5 seams). Stopped and
  reported rather than patching.

## 2. Calibration hold across the WHOLE pan  (the part that DID run)
Keyframe-consistency re-fit on HARD (config-driven, generalized cleanly):
- keyframe mutual-consistency: **mean 20.4 → 0.7 px** (re-fit works on HARD too)
- landmark court-fit: **mean 0.75 ft / max 1.75 ft** (HARD's validated-baseline range)

Direct-anchor profile, sampled every 100 frames across the clip (match to nearest
keyframe):

```
 frame  t(s)   kf  dist  inliers  reproj_px
   300  10.0  600   300    1536     0.66
   500  16.7  600   100    2464     0.62
   600  20.0  600     0    9642     0.00   (keyframe self-match)
  1200  40.0 1200     0   10638     0.00   (keyframe self-match)
  1500  50.0 1200   300    1512     0.67
  2000  66.7 1200   800    1362     0.71
  2400  80.0 1200  1200    1151     0.80
  2700  90.0 1200  1500    1121     0.64
```

- **25/25 sampled frames had a strong anchor (≥200 inliers), sub-pixel reproj
  (0.59–0.80 px) across the full 300–2700 range** — including frame 2700, which is
  **1500 frames (50 s) from its nearest keyframe** yet still matched at 1121 inliers.
- **No low-texture / fast-motion stretch broke the anchor at this sampling.** HARD is
  a broadside court pan; enough court texture is shared with a keyframe everywhere.
- **Caveat (do not overread):** this is a 100-frame-spaced sample (25 frames). It
  does **not** test individual motion-blurred frames between samples; a per-frame
  sweep would be needed to catch transient blur. What it shows: no systematic
  calibration breakdown anywhere in the pan.

## 3. Safety property over the 91 s  — NOT empirically exercised on HARD
The identity layer did not run on HARD (§5), so I cannot report empirical
zero-continuity-confirms / cross-window-relinks / disagreement behavior **for this
clip**. Two honest statements:
- **Structurally, the guarantees are clip-independent and hold by construction:**
  `CONFIRMED` is reachable only through `set_confirmed(provenance in {seed,
  second_signal})` (continuity has no valid provenance), and each window uses a fresh
  machine with an empty lost pool. These are code-path invariants, not tuned to
  TEST1.
- **Empirically they were validated on TEST1's span only** (prior stages: 0
  continuity-confirms, cross-window relinks 1→0, disagreements flag-not-overwrite).
  Extending that evidence to HARD's full 91 s requires the seams in §5 resolved.
- **No CONFIRMED-via-continuity was produced anywhere** (nothing was confirmed on
  HARD at all — 0 confirms, see §6).

## 4. Output integrity — N/A this run
No `player_events` or box score were generated for HARD (identity layer not run). The
stamping + confirmed-only logic is structural and was validated on TEST1; it was not
re-exercised here.

## 5. Integration seams that block a full HARD run  (reported, NOT patched)
These are the specific reasons P1-events + P2-identity cannot run end-to-end on HARD
as the code stands. Each is a per-clip parameterization gap, not a bug:

1. **No combined P1+P2 entry point.** No script imports both a Phase-1 stage and a
   Phase-2 stage. `process_game.py` is the rejected World-B draft, not this system.
2. **Frame ranges are TEST1-hardcoded.**
   - `phase1/stage2_generate_events.py: GEN_FRAMES = range(120,581,10)`
   - `phase1/stage1_court_roi.py: SAMPLE_FRAMES = range(120,581,30)`
   - `phase2/run_tracking.py: SPAN_START=300, SPAN_LEN=120`
   HARD's calibrated pan is `600..1200`; these target `120..580`, a different part of
   HARD. (They wouldn't crash — the profile shows 300–500 anchors to kf600 — but they
   sample the wrong window and are not "the full pan.")
3. **Roster + seed labels are TEST1-only.** `phase2/roster.py`: `ROSTER_NUMBERS =
   {5,13,24}` and `SEED_LABELS` keyed `"TEST1"` (Milford/Little Miami). HARD is Winton
   Woods "Warriors" — `seed_number_for("HARD", …)` returns `None` for every track, so
   no candidate has a position hypothesis and **OCR can auto-confirm nothing on HARD**.
4. **No HARD tracks cache.** Phase-2 stages read `{ACTIVE}_tracks_raw.json`; only
   `TEST1_tracks_raw.json` exists.
5. **Full-clip tracking is compute-infeasible here.** ByteTrack over 2746 frames on
   this CPU-only box ≈ **~90 min** for detection alone (before SIFT anchoring + OCR).
   Not a single-pass operation in this environment.
6. **The "~15 s" possession window is actually `DEMO_WINDOW_SECONDS = 2.0`** in
   stages 3–6 (a known demo stand-in, per DECISIONS §3) — flagged so it isn't mistaken
   for real possession logic on a full clip.

## 6. Auto-confirms across the full clip
**Zero.** Not because OCR failed — because HARD has no roster/seed-labels, so the
promote path has no position hypothesis to agree with. Nothing to eyeball.

## 7. The rigged-low baseline numbers (data, NOT a quality signal)
As instructed, recorded and explicitly labeled as the known rigged-low pool
(seed-everyone + partial 3-number roster + 2 s window), **not** a quality signal:
- HARD this run: **0 confirms / 0 reads** (no HARD roster — structural, not model).
- TEST1 prior run (for reference): 2 auto-confirms, per-possession confident-read
  rate 11% vs per-frame 3%.

## Bottom line
The **calibration spine is solid across the full 91 s** and generalizes to HARD via
config. The **identity/team-stats layers are not yet integrated for a second clip** —
they are TEST1-wired stage scripts with no combined runner, no HARD roster, and a
full-clip compute cost that this box can't do in one pass. These are the atomic
integration gaps to decide on next; I changed nothing.
