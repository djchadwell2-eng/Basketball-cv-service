# REVIEW.md — read-only codebase review + feature gap analysis

Date: 2026-07-02. Scope: full repo at commit `55de0e8`. Method: read
`phase1/DECISIONS.md` + `phase2/DECISIONS.md` first (note: **`phase2/HANDOFF.md`
does not exist** — see §1), then every file on the pipeline path, then supporting
modules, then standalone/demo scripts, then the World-B leftovers. Nothing was
changed. Severity scale: **CRITICAL** (wrong output or safety-property risk on the
path you actually run) / **SHOULD-FIX** (real defect, fires under realistic
conditions) / **COSMETIC**. Effort estimates are solo-dev realistic.

Judged against your stage: solo dev, pre-customer, two clips. "GOOD" below means
genuinely good at any stage — patterns to protect.

---

## 0. Executive summary

**What you've built is structurally sound where it matters most.** The identity
state machine really does have a single choke point: `ident.state =
IdentityState.CONFIRMED` appears **exactly once in the entire repo** — inside
`set_confirmed` (phase2/identity.py:126) — and the grep-audit in §6.1 proves every
caller routes through `seed` or `second_signal` provenance. The calibration engine
(direct nearest-keyframe anchoring + consistency re-fit) is validated on two gyms
with honest numbers. The abstention principle survives all the way to the box
score. The DECISIONS.md discipline is better than what most funded teams keep.

**The five defects that matter most, ranked:**

1. **The dual-config "loud assertion" cannot fire — it's a tautology.**
   `run_clip._sync_and_guard` sets `cc.ACTIVE = config.name` and then checks
   `cc.ACTIVE != config.name` — it re-reads the value it just wrote. The guard
   documented in phase2/DECISIONS.md §4 as the thing standing between you and
   "team_events extracted from the wrong video" provides zero protection. The
   *actual* protection is import ordering (sync happens before the stage modules
   bind their module-level constants), and nothing enforces or checks that.
   (§6.2 has the honest fix.)
2. **The tracks cache has no staleness validation.** run_clip *prints* the cached
   span but never compares it (or the clip name, or the video) to the ClipConfig.
   Stages 4 and 6 sample frames from `CLIP.video_path` at cached frame indices and
   OCR crops out of them — a stale or wrong-clip cache silently produces garbage
   crops and garbage identity evidence. This is the exact silent-failure class the
   rest of the architecture is built to prevent. (§6.4.)
3. **Stage 6's OCR outcomes are never persisted.** The pipeline's canonical output
   (`{clip}_player_events.json`, written by stage 5) is generated *before* OCR
   runs, and stage 6 writes only stdout + stills. run_clip's INTEGRITY section
   reads the stage-5 JSON, so the reported box score is pre-OCR. Your most
   valuable signal (auto-confirms, disagreements) evaporates when the console
   scrolls away.
4. **`phase2/stage1_states.py`'s IRON-RULE proof harness is broken and its claims
   are stale.** It calls `machine.promote_via_second_signal(ident)` — the
   pre-implementation zero-extra-args stub signature. Today that raises
   `TypeError` (uncaught; the harness crashes), and the printed claim
   "promote_via_second_signal() -> NotImplementedError (seam, unbuilt)" is no
   longer true. Your safety property's demo/verification script cannot run.
5. **README.md documents the rejected World-B draft as the product.** Anyone
   (including future-you or a future Claude session) reading the repo front door
   is directed to `process_game.py`, the 94-ft NBA court mapper, and the k-means
   team assigner — all explicitly rejected. The real pipeline (run_clip) is
   documented nowhere outside docstrings and DECISIONS.md.

**Do-first list** (all small, all before the fair OCR remeasure): make the cache
check real (compare clip/span/video against config, fail loud), make the
dual-config guard compare *bound stage values* against the config, fix
stage1_states' proof, persist stage-6 outcomes to JSON, add the regression suite
(§7.21) so everything after is safe. Roughly two evenings total. ROADMAP.md
Phase 0 sequences this.

---

## 1. Docs vs code — drift report

Rule applied: the CODE is truth; docs that disagree are listed as drift.

| Doc claim | Code truth | Action |
|---|---|---|
| Task brief says read `phase2/HANDOFF.md` | File does not exist anywhere in the repo (RUN_REPORT.md already noted this on the HARD diagnostic) | Either write it or stop referencing it. The DECISIONS.md pair currently *is* the handoff. |
| README.md (entire file) | Describes World B (`process_game.py`, `src/court_mapping.py` 94×50 court, k-means teams, `--recalibrate` flow). The actual system is `run_clip.py` + `clip_config.py` + phase1/phase2. | SHOULD-FIX: rewrite README (~1 hr). Highest-leverage doc fix in the repo. |
| phase2/DECISIONS.md §4: dual-config reconciled by sync "**plus a loud assertion**" | The assertion is tautological and cannot fire (run_clip.py:34-41). The sync is real; the guard is not. | Fix the guard (§6.2), then update DECISIONS. |
| phase2/windows.py docstring: "WINDOW = ~15s is a STAND-IN" | Actual window is `CLIP.accumulation_window_seconds` = 2.0s (the callers pass it). DECISIONS.md §3 already corrected this; the docstring didn't get the memo. | COSMETIC: one-line docstring fix. |
| phase2/stage1_states.py prints "promote_via_second_signal() -> NotImplementedError (seam, unbuilt)" | The method has been implemented since the stage-6 build; the call itself now TypeErrors | SHOULD-FIX (see §4.1). |
| phase1/DECISIONS.md, tasks/todo.md | Accurate against code everywhere I checked (450-skip removal, refit wiring, threshold values, seed lock). Genuinely well-maintained. | None. |
| RUN_REPORT.md §5 seams | Historical snapshot; most seams since closed by run_clip/ClipConfig. Fine as a dated report. | Optional: add one line at top pointing to run_clip as the resolution. |

---

## 2. Pipeline path, file by file

### 2.1 `run_clip.py` (entry point)

**GOOD**
- The right shape exists: one entry point, calibration → cache-read → P1 → P2 →
  integrity, with loud sectioning. Building this *fresh* instead of resurrecting
  `process_game.py` was the correct call.
- Refusing to track inline (hard `SystemExit` if the cache is missing, with the
  exact command to run) is the right ergonomics for a 90-min-per-clip operation.
- The INTEGRITY section re-reading the stage-5 artifact and asserting "every event
  stamped" is validate-by-eye culture applied to itself.

**BAD**
- **CRITICAL — tautological guard** (lines 34-41): `cc.ACTIVE = config.name` then
  `if cc.ACTIVE != config.name: raise`. This can never raise. What it *should*
  check is whether the values the stage modules actually **bound at import time**
  match the config — that's the desync that extracts team_events from the wrong
  video. Exact fix (still a patch, but a real one): after importing
  `stage2_generate_events as gen`, assert
  `gen.st.s2.VIDEO_PATH == config.video_path` and
  `gen.st.s2.cfg.ACTIVE == config.name` and `gen.CLIP is config` — i.e. compare
  the *downstream bindings*, not the variable you just wrote. ~20 min. Root-cause
  option costed in §6.2.
- **SHOULD-FIX — cache read with no validation** (lines 73-78): the doc already
  contains `clip`, `span_start`, `span_len`; you print them and proceed. One `if`
  makes staleness loud:
  `doc["clip"] == config.name and doc["span_start"] == config.tracking_span_start
  and doc["span_len"] == config.tracking_span_len` → else raise with both values.
  ~15 min now; full fingerprinting is §7.20.
- **SHOULD-FIX — INTEGRITY reads a pre-OCR artifact**: stage 6 runs after stage 5
  but writes no JSON, so lines 104-111 report the box score *without* the OCR
  layer. Today that's "only" misleading; once retroactive merge exists it would be
  wrong. Minimal fix: stage 6 writes `{clip}_ocr_confirms.json` and run_clip
  reports both. ~30 min.
- **SHOULD-FIX — an empty cache crashes uninformatively**: `min(fidx)` on an empty
  `frames` list raises `ValueError: min() arg is an empty sequence` — the real
  cause (bad video path at cache time, see §2.4) is three steps upstream. Guard
  with a clear message. ~5 min.

**NEEDS IMPROVEMENT**
- run_clip's "coach output" section runs only `stage3_team_stats` (console text).
  The actual coach-facing artifacts — `stage3_heatmap.py` and `stage3_demo.py` —
  are never invoked, so the one entry point does not produce the demo PNG. Two
  import+main() lines. ~5 min, do now.
- The CALIBRATION section calls `rk._setup()` + `rk._solve()` **fresh**, while the
  pipeline stages call `refit_keyframes.refit()` which prefers the cached `.npz`.
  Two consequences: (a) you pay a duplicate solve every run; (b) the numbers
  printed under CALIBRATION can describe a *different solution* than the one the
  pipeline uses (if the npz is stale). Fix: call `refit()` once, report its
  consistency by projecting through the returned `Hs`, and let the stages reuse
  the same cached result. ~30 min, do when touching calibration next.
- No CLI: `python run_clip.py` silently means TEST1 (the `ACTIVE_CLIP` default).
  `python run_clip.py HARD` via a 6-line `__main__` lookup would remove a
  wrong-clip footgun. ~15 min.
- Private-API reach-in (`rk._setup`, `rk._solve`) — promote them to public names
  when you touch refit next. COSMETIC.

### 2.2 `clip_config.py` (the future API contract)

**GOOD**
- Exactly the right idea at exactly the right size: one frozen-ish dataclass, ten
  plain-English fields, per-clip constants with their provenance documented in
  comments ("the exact values relocated from..."). The `Team` sub-dataclass with
  `frozenset` numbers is a clean roster shape.
- The docstring honestly states the remaining split (calibration inputs still in
  spikes/clips_config) instead of pretending the merge happened.

**BAD**
- **SHOULD-FIX — zero validation** (feature #25): nothing checks the video exists,
  the ranges are sane, `name` matches a `clips_config` entry, or — the live one —
  that `seed_labels` values are on the roster. HARD_CLIP today has
  `seed_labels={4: 23, 10: 30}` against roster `{5, 13, 24}`: numbers 23/30 are
  **structurally unconfirmable** (the closed-set filter in `read_jersey` can never
  emit them), so AGREE is impossible on HARD *by config construction*. You know
  this (it's the stand-in rig), but nothing marks it, and in three months "0
  confirms on HARD" will look like an OCR result instead of a config artifact. Fix
  when building the fair remeasure: a `validate()` method that fails loud, with an
  explicit `allow_off_roster_seeds=True` escape hatch for deliberate rigs. ~1-2 h.

**NEEDS IMPROVEMENT**
- As the web-app contract, three fields won't survive serialization as-is:
  `event_frames`/`render_sample_frames` are `range` objects (not JSON), and
  `seed_labels` keys are ints (JSON keys are strings). Don't fix now — but when
  you write the API layer, define the JSON form first and make ClipConfig parse
  it (`from_json`/`to_json` + validate = the whole contract). Noted in §6.6.
- `tracks_cache_path` being caller-visible is right for now; the web layer should
  derive it, not accept it from a client.

### 2.3 `cache_tracks.py`

**GOOD**
- The one-comment landmine warning ("set BEFORE importing run_tracking (binds at
  import)") is honest, and `cache()` does the ordering correctly.
- Caching only the tracking span (not the whole clip) is the right cost decision.

**BAD**
- **SHOULD-FIX — single-shot-per-process assumption is unguarded**: calling
  `cache(TEST1_CLIP)` then `cache(HARD_CLIP)` in the *same* Python process
  silently writes the second cache from the **first** clip's span/paths
  (`run_tracking` is already imported; its module-level `CLIP` binding is stale).
  Same class as §6.3. Cheap guard: `cache()` checks
  `"run_tracking" not in sys.modules` before the import and raises with an
  explanation, or run_tracking re-reads `clip_config.ACTIVE_CLIP` inside `main()`
  instead of at module level (the real fix, ~15 min).

### 2.4 `phase2/run_tracking.py` + `phase2/tracking.py`

**GOOD**
- `tracking.py` is exemplary: the validated detector config as module constants,
  a dumb `(track_id, bbox)` emitter, an explicit docstring stating continuity ≠
  identity. This is the "keep the safety-critical part small" principle done
  right.
- Writing `span_len = n` (frames *actually* extracted) rather than the requested
  length is honest bookkeeping.

**BAD**
- **SHOULD-FIX — a bad video path produces an empty cache with no error**:
  `cv2.VideoCapture(bad_path)` doesn't throw; `grab()` returns False (ignored);
  the writer writes 0 frames; tracking yields nothing; a `frames: []` cache is
  written "successfully". The failure finally surfaces as run_clip's `min()`
  ValueError. Violates abstention-first in spirit (silently proceeding on a bad
  input). Fix in `extract_subclip`: `if not cap.isOpened(): raise`; after the
  loop, `if n == 0: raise`; and in `main()`, warn loudly if `n < SPAN_LEN`
  (span exceeds clip length). ~15 min. Note `spikes/stage2_multikeyframe.
  extract_frames` (line 100) already does this correctly — copy its pattern.
- **COSMETIC**: the temp subclip is re-encoded (mp4v), so ByteTrack sees
  slightly-recompressed frames while stage 4/6 crop from the *original* video at
  the same indices. Bounded and harmless today; disappears if you ever feed
  frames to `model.track` directly. Note only.
- **COSMETIC**: temp file never deleted (deterministic name, overwritten per run).

### 2.5 `phase1/stage1_court_roi.py` (ROI filter + the anchor factory)

**GOOD**
- `build_court_anchor()` is the pipeline's best abstraction: one closure that
  returns `(T, inliers, reproj, kf)` per frame, built on the refit keyframes, with
  the confidence measured from the *same match that positions players*. The
  docstring explains the direction of every matrix. This is the piece to protect.
- The horizon guard (`w`-sign vs a known on-court point) and the frame-edge foot
  drop are both documented with *why*, and both came from eyeball-found leaks —
  the validate-by-eye loop working as intended.

**BAD**
- **SHOULD-FIX — `anchor()` failure crashes instead of flagging**: `anchor()`
  deliberately returns `(None, 0, inf, k)` on a failed/degenerate match, but both
  consumers (`main()` line 207-208 here, and stage2_generate_events lines 46-47)
  immediately compute `H_court @ T` → `TypeError` on None. On TEST1/HARD this is
  unreachable (matches never fail); on the *next* gym it's how the pipeline dies
  mid-run with a stack trace instead of emitting a `low_confidence` frame. The
  machinery to do the right thing already exists (`confidence_state`, the schema's
  `low_confidence` state). Fix: on `T is None`, skip the frame's detections and
  emit a `low_confidence` event with `inliers=0` (stage 2), or skip-and-log
  (stage 1 render). ~20-30 min. Do before the multi-gym phase.
- **COSMETIC**: reads every frame sequentially (`for f in range(total): cap.read()`)
  and processes only samples — correct (frame-accurate) but wasteful; fine at
  current scale. The pattern to keep is `extract_frames`'s grab/retrieve.

**NEEDS IMPROVEMENT**
- This file is both a library (`build_court_anchor`, `on_court`, `pixel_to_feet`
  imported by three other stages) and a script (its own YOLO render main). Split
  when convenient — the library half belongs somewhere named like
  `phase1/court_anchor.py`. ~1 h, defer until it annoys you.

### 2.6 `phase1/stage2_generate_events.py`

**GOOD**
- Reuses stage 1's classification verbatim (no drift between the render you
  eyeballed and the events you persist), and the homography confidence comes from
  the same match that positioned the players — the guardrail finally gates the
  thing it measures, as the docstring says.
- Deterministic detection sort before writing JSON — byte-stable outputs.

**BAD**
- **This is the dual-config file**: `GEN_FRAMES`/output path from `clip_config`,
  video + clip label from `st.s2` (spikes). Fully analyzed in §6.2.
- Same `anchor()` None crash as §2.5.
- **COSMETIC**: the frame-450 verification block (lines 96-104) is TEST1-lore; it
  no-ops harmlessly on other clips (`if ev450:` guards it) but will print a
  confusing TEST1 story if any future clip's range includes literal frame 450.
  Move to a comment or gate on `CLIP.name == "TEST1"` when next editing.

### 2.7 `phase1/refit_keyframes.py`

**GOOD**
- The right fix at the right layer (dense adjacent-keyframe correspondences into
  the *same* least_squares, warm-started) with before/after consistency printed
  and the landmark-fit regression check (`must stay tight`) in the report. The
  numbers it claims in DECISIONS.md are the numbers the code computes.

**BAD**
- **SHOULD-FIX — the npz cache has no invalidation inputs**: `refit()` returns the
  cached solution keyed only by clip name. Re-click a landmark in
  `clips_config.py` (there's a tool for exactly that, `spikes/reclick_ft.py`) or
  change keyframes, and every downstream stage silently keeps calibrating with the
  old solution until someone remembers "delete to recompute" (docstring-only
  protocol). Fix inside the fingerprinting pass (§7.20): store a hash of
  `(KEYFRAMES, LANDMARKS, EXCLUDE_REGIONS, video_path)` in the npz and recompute
  on mismatch. ~30 min.
- **COSMETIC**: `CACHE` binds `s2.cfg.ACTIVE` at import — same import-order
  constraint as everything else (§6.3); correct under run_clip's ordering.

### 2.8 `phase1/team_event_schema.py`

**GOOD**
- The best-written module in phase1: the *why identity-free* essay is the
  architecture rule made durable; explicit iron rules (no track_id, order-
  independent, team never guessed); versioned documents; lossless round-trip with
  a self-check main. As the spine contract, this is web-app-ready in shape.

**NEEDS IMPROVEMENT**
- `IDENTITY_STATES` / `TEAM_LABELS` / `CONF_STATES` are declared but never
  enforced — `Detection(identity_state="grue")` serializes happily. A 5-line
  `__post_init__` check would make the vocab real. COSMETIC now; do it when the
  schema next changes. The known illustrative `_example_event` main is fine.

### 2.9 `phase1/stage3_team_stats.py`, `zones.py`, `stage3_heatmap.py`, `stage3_demo.py`

**GOOD**
- Pure deterministic reads over the JSON, no perception, trust-filter applied in
  exactly one function (`load_trusted`) that everything else reuses. `zones.py`
  ties boundaries to real court features (FT line, arc top) from config rather
  than magic fractions, and prints spot-checks for eyeball verification.
- The demo artifact honestly labels itself ("proves the map works, not real
  season stats") — abstention in the marketing layer, which coaches will trust
  more, not less.

**BAD**
- **COSMETIC (known, logged)**: `zones.py` demo main hardcodes
  `TEST1_team_events.json` + frame 300 (lines 107-111); `stage2c_validate.py`
  likewise (§4.3). Both are report-only paths, already inventoried in DECISIONS.
- `zones.py` binds `_COURT = cfg.active()["court"]` at import (module-level) —
  same §6.3 constraint. Works under run_clip ordering.

### 2.10 phase2 core: `identity.py` + `windows.py`

**GOOD — this pair is the crown jewel; protect it.**
- `identity.py` holds *only* state logic (no detection, no court math), which is
  what makes the safety property auditable in one sitting. The provenance
  frozenset + raise-on-anything-else is the structural guarantee working exactly
  as advertised. Evidence dicts on every transition (gap frames, distances,
  predicted centers) mean every state is *explainable* — that's your future
  review-UI payload already in place.
- `promote_via_second_signal` handles four outcomes (including the subtle
  `no_position_hypothesis`: a confident OCR read with nothing to agree with
  correctly abstains rather than confirming off one signal — a decision many
  would have gotten wrong).
- DISAGREE → UNKNOWN with both numbers in evidence is the swap-detector, kept
  honest (flag, never resolve).
- `windows.py` is 45 lines and does exactly one thing (fresh machine per window).
  Containment as code, not convention.

**BAD**
- **SHOULD-FIX — `seed()` is a silent no-op for an unknown track_id**
  (identity.py:134-139): if a hand-verified seed label references a track that
  isn't present at the seed frame (typo, or ByteTrack renumbered after a re-run),
  nothing happens and nothing says so. You'd debug "why didn't #23 confirm" for an
  hour. Fix: return a bool and/or `print/log a WARNING`, and have stage 4/6 report
  "seeded k of n labels; unmatched: [...]". ~15 min. This matters *now* because
  seed labels are hand-maintained per clip.
- **COSMETIC/foot-gun to document**: `promote_via_second_signal` doesn't check the
  identity's current state — called on a CONFIRMED identity with a disagreeing
  read, it would demote CONFIRMED → UNKNOWN. Arguably *correct* abstention
  behavior, but it's an undocumented reachable transition; today stage 6 only
  feeds it candidates. When retroactive merge lands, decide explicitly and write
  it into DECISIONS.md.
- **COSMETIC**: the `_lost` pool never expires within a window (gap-filter makes
  stale entries unmatchable, so behavior is correct; memory is trivial at this
  scale). Note only.

**NEEDS IMPROVEMENT**
- The relink dials (`MAX_GAP_FRAMES=30`, `MAX_MATCH_DIST_PX=150`, `AMBIG_RATIO=1.5`)
  are resolution- and fps-implicit (150 px means something different at 720p vs
  4K; 30 frames assumes 30 fps). Fine for now; when a third clip with different
  fps/resolution arrives, move them onto ClipConfig or normalize by fps and frame
  height. ~30 min, defer until that clip exists.

### 2.11 `phase2/roster.py` + `phase2/ocr_reader.py`

**GOOD**
- `ocr_reader.py` is a genuinely clean seam (assessed fully in §6.5): closed-set
  filtering, the single documented autonomy dial, lazy easyocr import so the other
  stages don't pay the load, small-crop refusal returning `[]` (= abstain).
- `roster.py` as a thin accessor over ClipConfig kept the OCR stage's call sites
  stable across the refactor — the right way to move config.

**BAD**
- **SHOULD-FIX — `seed_number_for(clip, track_id)` ignores its `clip` argument**
  (roster.py:19-21). A caller passing `"TEST1"` while `ACTIVE_CLIP` is HARD
  silently gets HARD's seeds. Either use the argument (assert it matches
  `ACTIVE_CLIP.name`) or drop it from the signature. ~10 min.
- `from clip_config import ACTIVE_CLIP` at module top = the frozen-at-import
  binding that makes the whole phase-2 stack one-clip-per-process (§6.3).

### 2.12 phase2 stages 3–6 (the run_clip-invoked drivers)

**GOOD**
- All four take `span_start`/`fps`/`span_len` from the **cache doc**, not the
  config — internally consistent with the tracks they process (the right choice;
  it localizes the staleness problem to the cache check).
- stage4's honesty block ("seeds EVERY track present... does not change the
  mechanism being validated") and stage6's readability measurement (per-frame vs
  per-possession printed side by side) are the honest-measurement culture in
  code. The review queue is persisted as JSON with `why` strings — the future
  review UI's data layer already exists.
- stage6's accumulation loop (best on-roster read across the window, capped
  attempts, stride) is the validated temporal-accumulation argument implemented
  exactly as designed, and `promote_via_second_signal` is fed the *accumulated*
  best read — the three-outcome application point is singular and auditable.

**BAD**
- **SHOULD-FIX — stage6 writes no machine-readable output** (see summary #3).
  `outcomes`, `best`, and the queue-after state die with the console. Fix: dump
  `{clip}_ocr_confirms.json` with the four outcome lists + evidence. ~30 min.
  This JSON is also the input contract for retroactive merge (§7.5) — you'll
  need it in weeks anyway.
- **SHOULD-FIX (scale note, not today's bug)**: stage2/stage6's
  `read_span_frames` loads the whole span into RAM (~6 MB/frame × span). At 120
  frames it's ~0.7 GB — fine. At a full-clip span it's 17 GB — the pattern, not
  the parameter, is the limit. When spans grow: iterate frames or fetch on
  demand per OCR attempt. Defer until spans grow.
- **COSMETIC**: `imgs[f]` KeyError if the video yields fewer frames than the
  cache claims (only reachable with a stale/mismatched cache — fixed by the
  cache check upstream).
- **COSMETIC**: stage4 line 143 `frames[sf - span_start]` assumes the cache is
  dense/contiguous from span_start (true by construction today; breaks silently
  if a future cache ever skips frames — one `assert fr[0] == sf` makes it loud).
- **COSMETIC**: stages 3, 4, 5, 6 each re-run the windowed state machine from
  scratch (4× compute of identical state). Harmless at 120 frames; consolidate
  into "compute identity once, stages consume it" only when spans grow or a
  stage gets a real cost.

---

## 3. Supporting modules (spikes engine + src/camera_tracking)

### 3.1 `spikes/clips_config.py`

**GOOD** — for what it holds (calibration inputs), the shape is right: per-clip
dicts with the palette factored out, clicked landmarks as literal data you can
diff, comments recording gym/team context, and `active()` reading at call time.
The "do not change; validated with these exact values" warning on HARD is the
right kind of guardrail comment.

**BAD** — none internally. Its *existence as a second config* is the debt (§6.2).

### 3.2 `spikes/stage1_keyframe_match.py`, `stage2_multikeyframe.py`, `stage3_optimize.py`, `stage4_courtmap.py`

**GOOD**
- `extract_frames`/`extract_frame`: sequential-decode frame accuracy with an
  explicit docstring on *why* `cap.set(CAP_PROP_POS_FRAMES)` is banned (silently
  wrong frames). This discipline is why your calibration numbers are trustworthy.
  It errors loudly on unreadable frames — the pattern run_tracking should copy.
- Fixed RANSAC seeds everywhere (`cv2.setRNGSeed(0)`) = reproducible geometry.
- `stage2`'s weak-pair flagging ("consider adding an intermediate keyframe") is
  abstention applied to calibration inputs.
- `stage4_courtmap.COURT_MODEL` derives from per-clip court dims (the HS-vs-NBA
  lesson, institutionalized), and `to_px` guards behind-camera projections.
- `stage3_optimize`'s extent check ("collapse/explode guard") and the "AFTER
  worse than BEFORE signals a setup problem, not a result to accept" line — this
  is the epistemic hygiene most CV code never has.

**BAD / COSMETIC (all small)**
- `stage4_courtmap.py:59-68`: the three `_far` tags hardcode `50.0` instead of
  `COURT_WID` while everything else derives from config. Harmless until a
  non-50-ft-wide court exists; one-line fix when touching the file.
- `stage2_multikeyframe.py:56-80` `LANDMARK_TAGS` carries 94-ft court-feet coords
  (R_FT at 75, right baseline at 94) and line 92 sets `COURT_LEN=94` — these feed
  only the palette + the "ideal court" comparison plot, *not* the pipeline's
  court math (stage4's COURT_MODEL does that, correctly, from config). Worth one
  comment so nobody ever "fixes" the pipeline against the wrong table. COSMETIC.
- `run_optimization()` re-runs SIFT + least_squares from scratch on every call
  and is called by `refit._setup()` → run_clip pays it once per run (the refit
  npz then caches the *solution*). Acceptable; noted under run_clip §2.1.

### 3.3 `src/camera_tracking.py` (the one src/ module still alive)

**GOOD** — clear WHY docstring, per-step sanity check (`_is_sane` scale bounds),
motion-lost counting surfaced to callers, player masking with padded boxes.

**Status note** — it's now used *only* by `stage4_courtmap.main()` (the spike
overlay renderer); the live pipeline replaced chained tracking with direct
anchoring. When World B is cleaned up (§5), this file stays (or moves to spikes/).

---

## 4. Standalone / demo scripts

### 4.1 `phase2/stage1_states.py`

- **SHOULD-FIX (headline defect #4)**: line 77 `machine.promote_via_second_signal(ident)`
  → `TypeError` (signature is now `(ident, number, confidence)`), uncaught (the
  `except NotImplementedError` no longer matches), so the IRON-RULE proof harness
  *crashes*; and its printed claims ("seam, unbuilt") describe a build state
  that's months gone. Exact fix (~15 min): call
  `machine.promote_via_second_signal(ident, None, None)` and assert the result is
  `"no_confident_read"`; keep the provenance-refusal check as-is (it still works);
  update the prints to say "second-signal path exists; confirms only via
  set_confirmed(second_signal); no-read abstains." Better: fold this into the
  regression suite (§7.21) so the proof runs on every change instead of never.
- Also reaches into `machine._by_track` (COSMETIC).

### 4.2 `phase2/stage2_recovery.py`

- **GOOD**: the state-colored overlay + reappearance stills are your best
  eyeball tool for the identity layer; the break log printing (gap, distance,
  result) matches the evidence schema.
- **COSMETIC**: `OUT_DIR` used without `os.makedirs` → on a fresh clone,
  `cv2.VideoWriter`/`imwrite` **fail silently** (OpenCV returns False, no
  exception) and you get no artifacts and no error. One `os.makedirs(OUT_DIR,
  exist_ok=True)` line. Same applies to stage4/5/6 (stage5 has makedirs via
  `open()`? No — `open()` doesn't create dirs; stage5 writes
  `phase2/out/...json` and *would* crash loudly with FileNotFoundError, which is
  at least loud). The dirs exist in your working copy, so this only bites a
  fresh checkout / CI.
- O(n²) machine replay for stills — fine at this scale.

### 4.3 `phase1/stage2c_validate.py`, `phase1/stage4_overlay.py`, `zones.py` main, `team_event_schema.py` main

- All four report-only TEST1/frame hardcodes confirmed exactly as logged in
  DECISIONS/your brief: `stage2c_validate.py:22` (TEST1 path),
  `stage4_overlay.py:24,30` (frames 120–580 + TEST1 handoff frames [171, 271,
  371, 461, 541]), `zones.py:107-111` (TEST1 + frame 300), schema example
  (illustrative, fine). No *new* members of this bug class found in phase1/,
  phase2/, or the root entry points — the sweep in §6.3 lists every remaining
  hardcode and where it's benign.
- stage4_overlay generalization = feature #14; the handoff frames should be
  *computed* (anchor-switch = midpoint between adjacent keyframes) instead of
  listed, which makes it clip-generic for free.

### 4.4 `run_diag_hard.py`

- **GOOD**: the harness discipline (set `cfg.ACTIVE` before any imports, comment
  saying exactly that) is the correct pattern under the current import-time
  binding regime — evidence the constraint is *workable* when known.
- COSMETIC: `SAMPLES = range(300, 2746, 100)` hardcodes HARD's length; it's a
  HARD-named diagnostic, acceptable. Superseded for integration purposes by
  run_clip; keep as the full-pan calibration profiler.

---

## 5. World B (`process_game.py`, `src/` except camera_tracking, `render_heatmaps.py`, `out.json`, `heatmaps/`)

Confirmed still present and exactly as documented in tasks/todo.md: single-frame
click calibration, **94×50 NBA court** (`src/court_mapping.py`,
`render_heatmaps.py:27`), k-means jersey teams, occupancy-side possession
estimate. Nothing in the live pipeline imports any of it except
`src/camera_tracking.py` (used by the spike overlay main).

**The problem is no longer that it exists — it's that README.md advertises it**
(§1). A reader lands on the repo, follows README, and runs the rejected system
with the wrong court model.

**Recommendation (sequenced, respecting "leave untouched" until now):**
1. Now: rewrite README around run_clip/ClipConfig (1 hr) — kills the active harm
   without deleting anything.
2. After the regression suite exists (Phase 0): delete `process_game.py`,
   `src/detection.py`, `src/court_mapping.py`, `src/team_assignment.py`,
   `src/team_stats.py`, `src/schema.py`, `render_heatmaps.py`, `out.json`,
   `heatmaps/` in one commit (git history preserves them forever; "leave
   untouched" made sense while it was the only end-to-end reference — run_clip
   removed that reason). Keep `src/camera_tracking.py` (still imported).
   30 min including the grep-for-imports check.

---

## 6. The six requested audits

### 6.1 Every entrance to CONFIRMED — proven closed

Method: grep for every assignment to `.state` and every use of
`IdentityState.CONFIRMED` / `set_confirmed` / `seed(` / `promote_via_second_signal`
across all `*.py`.

Findings, exhaustively:
- `ident.state = IdentityState.CONFIRMED` occurs **once**: identity.py:126,
  inside `set_confirmed`, after the provenance check (raise on anything outside
  `{seed, second_signal}`; the frozenset at identity.py:44 is not mutated
  anywhere).
- `set_confirmed` callers: `seed()` (identity.py:136, provenance="seed"),
  `promote_via_second_signal` (identity.py:155, provenance="second_signal",
  AGREE branch only), and stage1_states.py:72 which calls it with
  `provenance="position_continuity"` **expecting refusal** (and is refused).
- `seed()` callers: stage4_seed_queue.py:83, stage5_player_events.py:74,
  stage6_ocr_confirm.py:82 — all the window-start seed-everyone stand-in, i.e.
  the sanctioned first signal.
- `promote_via_second_signal` callers: stage6_ocr_confirm.py:122 (with the
  accumulated best read) and the broken stage1_states.py:77 call (TypeErrors
  before executing; cannot confirm anything).
- All other `.state =` assignments set LOST / CANDIDATE / UNKNOWN (identity.py
  lines 161, 183, 204, and the UNKNOWN constructor default) — uncertainty-
  increasing or lateral, never confirming.
- No test/demo code bypasses the gate. No pickle/JSON deserialization
  reconstructs Identity objects (states are only ever *written* as strings into
  event JSONs, never read back into machines).

**Verdict: the safety property holds by construction, exactly as claimed.** The
one blemish is that the script that *demonstrates* it crashes (§4.1).

### 6.2 The dual-config debt — guardrail assessment + root-fix costing

**Is the assertion guardrail sufficient? No — it is vacuous.** run_clip.py:34-35
writes `cc.ACTIVE = config.name`, then line 35 checks the same attribute it just
wrote. There is no execution path where it raises. What actually keeps HARD runs
correct today is **import ordering**: `run()` syncs both selectors *before* the
first import of `refit_keyframes` / `stage4_courtmap` / `stage2_generate_events`,
whose module-level constants (`_CLIP = cfg.active()`, `VIDEO_PATH`, `CACHE`,
`GEN_FRAMES`, `OUT_JSON`...) then bind from the synced state. That ordering is an
unwritten invariant; nothing checks it; a second `run()` in the same process, or
any future top-of-file import of a stage module in run_clip, breaks it silently.

**What the guard should be (patch tier, do now, ~20-30 min):** verify the
*bindings*, not the source variable. After the stage imports inside `run()`:

```python
import stage2_generate_events as gen
assert gen.st.s2.cfg.ACTIVE == config.name
assert gen.st.s2.VIDEO_PATH == config.video_path   # the one that really matters
assert gen.CLIP is config
```

That catches import-order violations, double-run staleness, *and* name/path
mismatches between the two configs — the actual failure modes. (Belt-and-
suspenders: also refuse a second `run()` per process with a module flag until
§6.3 is fixed properly.)

**Root fix costing (make ACTIVE_CLIP the single source of truth):**
- Small version (~1-2 h): `spikes/clips_config.py` keeps the CLIPS dict but
  `active()` resolves via `clip_config.ACTIVE_CLIP.name`, and `ACTIVE` becomes a
  derived value (or is deleted; run_diag_hard sets ACTIVE_CLIP instead). Stage
  modules stop reading `s2.VIDEO_PATH` for the video and use
  `ACTIVE_CLIP.video_path` (assert it equals the clips_config entry during the
  transition). Standalone stage2 invocations then *cannot* desync.
- Full version (merge calibration inputs into ClipConfig): move keyframes /
  landmarks / exclude_regions / court dims into ClipConfig (or a
  `CalibrationConfig` it owns). ~0.5-1 day incl. re-running TEST1+HARD to verify
  byte-identical outputs. **Recommended timing:** small version in Phase 0; full
  merge when you build the web API (the contract forces the merge anyway —
  don't do it twice).

Honest cost of *not* fixing now: every new stage/script added inherits the
import-order trap invisibly, and the class of bug you've already hit twice
(TEST1 hardcodes) keeps a structural breeding ground.

### 6.3 Remaining hardcode + frozen-binding inventory (assume more exist — verified sweep)

Clip/frame hardcodes remaining, complete list (root + phase1 + phase2):
- `phase1/stage4_overlay.py:24,30` — frames 120–580 + TEST1 handoff list
  (report-only; generalize as feature #14).
- `phase1/stage2c_validate.py:22` — TEST1 path (report-only).
- `phase1/zones.py:107-111` — TEST1 path + frame 300 in demo main (report-only).
- `phase1/stage2_generate_events.py:96` — literal 450 in a verification print
  (benign; TEST1-lore).
- `phase2/stage3_windows.py:30` — `STANDIN_WINDOW_SECONDS = 15.0` print-only
  (documented).
- `run_diag_hard.py:29` — HARD length in a HARD-named diagnostic (acceptable).
- `spikes/*` mains — HARD-era output names/frames (spike history; leave).

**The subtler, more important class — config values frozen at import time**
(the reason "sync then import" is load-bearing): `phase2/roster.py:13`
(`ACTIVE_CLIP` binding → ROSTER_NUMBERS + seed lookups), `phase2/run_tracking.py:24-28`,
every phase2 stage's `TRACKS_JSON`/`OUT_JSON`/window constants,
`phase1/zones.py:37` (court dims), `phase1/refit_keyframes.py:49` (CACHE path),
`spikes/stage2_multikeyframe.py:37-86` (video/keyframes/landmarks/regions),
`spikes/stage4_courtmap.py:51` (court dims). Consequence: **one clip per Python
process** is a hard, unstated system invariant. Today's CLI usage satisfies it;
a future long-lived web worker calling `run()` per job is the exact scenario
that violates it silently (roster/zones/paths stay on the first job's clip).
Fix class: stages read `clip_config.ACTIVE_CLIP` inside `main()` (call-time)
instead of at module top — mechanical, ~1-2 h across ~10 files; bundle with the
§6.2 small root fix or do at web-integration time. Until then: document the
invariant in run_clip's docstring and add the double-run refusal guard.

### 6.4 The tracks cache — staleness + missing-cache behavior

- **Missing cache:** handled well — run_clip refuses with the exact
  regeneration command (run_clip.py:68-72). Standalone stages, however, crash
  with a raw `FileNotFoundError` from `open()` (acceptable; loud).
- **Stale cache: no protection at all.** The cache doc *carries* `clip`,
  `span_start`, `span_len`, `fps` — and no reader validates any of it against
  the active config. Failure modes, concretely: (a) change
  `tracking_span_start` in clip_config and forget to re-run cache_tracks →
  stages process the *old* span while you believe you're analyzing the new one
  (internally consistent, so nothing crashes — the worst kind); (b) point
  `video_path` at a different/re-trimmed file → stage 4/6 crop pixels at cached
  bbox coordinates from frames the tracker never saw → OCR evidence is garbage
  with confident formatting; (c) empty cache from the §2.4 silent-extract bug →
  distant `min()` ValueError.
- **What the cache is missing to make staleness detectable:** identity of the
  source video (path + size + mtime, or a cheap first-frame hash), the model
  name, and the requested-vs-actual span. All are one dict away at write time
  (run_tracking.py:70). Then a tiny `load_cache(config)` helper (used by
  run_clip *and* the four stages, replacing their copy-pasted `load()`) compares
  and raises with both sides printed. ~1-2 h total = feature #20. **This is the
  single highest-value robustness fix in the repo** — it converts the "wrong
  crops silently" class into a refusal, which is the abstention principle
  applied to your own artifacts.

### 6.5 The OCR seam — is `read_jersey()` genuinely pluggable?

**Yes — verified by reading the call surface.** The contract is
`read_jersey(crop_bgr, roster_numbers) -> [(int, float), ...]`; callers
(stage6, `best_on_roster_read`) consume only that. Everything engine-specific
(easyocr import, upscaling, allowlist) is inside the function; easyocr is
lazy-imported so a PaddleOCR swap wouldn't even change import costs elsewhere;
the state machine (`promote_via_second_signal`) sees only `(number, confidence)`
and reads the threshold from `ocr_reader.OCR_CONFIRM_THRESHOLD` — one module to
touch, zero state-machine changes. Swap = rewrite one ~15-line function body.

Two honest caveats to write down before ever swapping:
1. **Confidence scales are not comparable across engines.** 0.85 is an
   *EasyOCR* number; PaddleOCR's confidences are calibrated differently. A swap
   invalidates the dial's meaning → re-run the fair-remeasure protocol before
   trusting any threshold. (Say this in DECISIONS.md when the time comes.)
2. Crop geometry (`jersey_crop`'s 15-85% × 15-50% torso box) and the two size
   gates (`MIN_CROP_HEIGHT_PX=24` in the reader, `MIN_OCR_HEIGHT=90` bbox gate
   in stage6 — consistent: 90 × 0.35 ≈ 31 px > 24) are part of the measured
   system; hold them fixed across the remeasure so you're measuring the roster/
   seeding change, not a crop change.

### 6.6 Web-app-facing embarrassment audit (input validation / paths / partial failure)

What breaks the day a real ClipConfig arrives over an API, in order of pain:
1. **No validation** (§2.2 / feature #25) — bad video path currently yields the
   §2.4 silent empty cache; malformed roster yields confusing downstream
   behavior; span beyond clip length yields a short cache with no warning.
2. **One-clip-per-process invariant** (§6.3) — a worker that processes two jobs
   without restarting silently mixes clips. Until root-fixed: **one subprocess
   per job** is the deployment rule; write it down.
3. **Serialization mismatches** (§2.2): `range` fields and int dict keys don't
   survive JSON; decide the wire format before the web app guesses one.
4. **Absolute local paths** (`C:\Users\djcha\Downloads\...`) in both configs —
   the API version must accept an uploaded-file reference and resolve it
   server-side; calibration inputs (clicked landmarks) are still in
   spikes/clips_config keyed by name, which a web client can't supply — the
   §6.2 full merge is a prerequisite for real remote use.
5. **Partial failure mid-pipeline**: any stage exception kills `run()`; artifacts
   already written stay on disk with no marker of which run produced them
   (feature #19 manifests) and no resumability (feature #9). Acceptable
   pre-web; must exist before jobs run unattended.
6. Silent-artifact-loss on missing out/ dirs (§4.2) — trivial makedirs fixes.

None of this blocks local work. All of it should be finished before the first
remote job — ROADMAP Phase 6/7 sequences it.

---

## 7. DELIVERABLE 2 — Complete feature gap analysis

Format per feature: **Status** (built / partial / absent) → what's missing → how
to build it at your level → effort → when.

### Core pipeline features

**1. Individual player tracking — the coach-named must-have.**
**Status: partial — the hard, safety-critical substrate is genuinely built;
the product layer on top is absent.**
Built and validated: ByteTrack continuity; the four-state machine with the
provenance lock; per-window containment; seeding; OCR second signal with
three-outcome resolution; identity-stamped presence events; confirmed-only box
score; review queue JSON. That was the riskiest 40% and it's done *correctly*.
Missing, in dependency order (this chain IS the roadmap spine):
  a. **Fair inputs** — ROI-mask seeding (#10) + real rosters (#11) so the
     candidate pool is ~10 players, not crowd/refs. Without this, nothing about
     OCR performance is knowable.
  b. **The fair OCR remeasure** (#4) → the true per-possession read rate = your
     deployment envelope. Everything downstream is sized by this number.
  c. **Retroactive stat merge** (#5) so a mid-window confirm credits the
     candidate span that preceded it.
  d. **Per-player aggregation keyed by jersey number.** Today's box score keys
     are `(window, identity_id)` — internal ids a coach has never heard of.
     A player's game line = union over windows of confirmed spans for their
     number. Small code, but it *defines* the product output. (~1 evening once
     c exists.)
  e. **Per-player court positions.** player_events currently carry NO court
     coordinates — the P2 tracks are never joined to the P1 homography. The
     join is cheap and already proven piecewise: `anchor(f)` + bbox
     bottom-center → `pixel_to_feet` → court_feet stamped onto each
     player_event. This unlocks per-player zone time / distance / heatmaps —
     real coach value *before* any ball detection exists. (~1 evening.)
  f. **Stat vocabulary beyond presence.** Until ball/shot detection (#6/#7),
     honest per-player stats are position-derived: minutes on floor, zone
     occupancy, distance covered, sprint counts. Say so plainly in the product;
     coaches respect "we don't guess."
  g. **Seeding + review UI** (#15/#16) — the coach's one-click-per-player
     surface. CLI/HTML stand-ins first; real UI in the web app.
Distance to "coach clicks a player, gets a stat line": with a, b, c, d, e done —
a *film-room-credible* per-player floor-time/position line on a clip. Realistic
solo pace: 4–8 focused weekends after Phase 0 hardening. Shooting stats add
#6/#7 later.

**2. Court homography / calibration engine.**
**Status: built and validated (your strongest asset).** Two gyms, mutual-
consistency re-fit generalized (TEST1 0.7 px; HARD 20.4→0.7 px), sub-pixel
anchors 1500 frames from a keyframe, per-frame confidence gating stats.
Strengths: direct anchoring (no chain drift); global consistency fixed at cause;
quality signal from the same match that positions players; config-driven per
clip; deterministic. Honest limits and what breaks it:
  - **Manual cost per clip**: pick keyframes + click ~5-12 landmarks per
    keyframe. That's the scaling bottleneck (#3 addresses it).
  - **Hard cuts / broadcast footage**: keyframe chain assumes one continuous
    pan; replays, score graphics wipes, multi-camera cuts each break the chain
    silently-ish (weak-pair flags help). Follow-cam single-shot only, for now —
    a *footage requirement*, fine for the target user if stated.
  - **Camera moves between games/quarters** (tripod bump): keyframes/landmarks
    are per-clip constants; a bump mid-clip = new calibration segment. No
    detection for this today (anchor inlier collapse would show it — the
    guardrail would flag, which is the designed behavior).
  - **Texture dependence**: SIFT needs court features in view; extreme zoom-ins
    or feature-poor floors reduce inliers (the confidence gate catches it).
  - **Frame-accurate sequential decode** makes long-video random access slow —
    a runtime cost at full-game scale, not a correctness issue.

**3. Auto-calibration.**
**Status: absent (correctly — sequenced after multi-gym proof).**
Realistic path at your level, in order:
  a. **First shrink the manual work, don't eliminate it.** You already have the
     machinery to propagate landmarks: adjacent-keyframe SIFT homographies. Click
     landmarks on ONE keyframe; project them into the other keyframes via the
     chain; snap/verify visually. Cuts per-clip clicking ~5×. ~1-2 evenings,
     pure reuse of existing functions. Do this during the multi-gym phase.
  b. **Classical line detection** (white-mask → LSD/Hough → intersect → RANSAC
     against the court model) is a known rabbit hole: gym floors have logos,
     colored paint, reflections; expect weeks of per-gym tuning. Only attempt
     after ≥5 gyms of footage exist to test against, and keep clicks as
     fallback.
  c. **Learned court-keypoint models** (the approach broadcast-sports tools
     use) need labeled data or an open model that fits HS gyms — evaluate
     later; don't train models solo yet.
Footage variety needed before committing: 5–10 gyms across lighting/floor
styles, which the multi-gym calibration proof (ROADMAP Phase 4) accumulates
anyway. Decision gate lives there.

**4. Player ID / jersey OCR — the fair remeasure.**
**Status: mechanism built + validated on rigged inputs; the *measurement* is the
missing piece.** Plan (= ROADMAP Phase 1):
  a. ROI-mask seeding (#10) so candidates are on-court bodies.
  b. Real rosters for both clips (#11) — numbers + colors from your dad/film.
  c. Rebuild the lost montage tool (#16) for eyeball verification of every
     confirm.
  d. Persist stage-6 outcomes (§2.12) and run both clips end-to-end.
  e. Record: per-frame vs per-possession confident-read rate, agree/disagree/
     no-read counts, and **eyeball-verify every single auto-confirm and
     disagreement** against video.
Then the autonomy-dial framework (only after the true number exists): the dial
question is "what fraction of possessions self-identify vs need a coach click."
≥~50-60% per-possession with zero wrong confirms → OCR is the workhorse;
~20-50% → OCR assists, review queue carries more, improve crops/reader before
touching the dial; <~20% → inputs or engine work (crop quality, zoom, PaddleOCR
behind the seam), never the threshold. Any *lowering* of 0.85: one notch at a
time, watching the disagreement (swap-flag) rate — which is your canary metric,
not the confirm count. Effort: the remeasure itself ~2-3 evenings once inputs
exist.

**5. Retroactive stat merge.**
**Status: absent (designed only), correctly gated behind the fair read rate.**
Build shape when its time comes:
  - Trigger: **inside the AGREE branch only** — the same event that calls
    `set_confirmed(provenance="second_signal")`. Structurally impossible to fire
    on reappearance/position because the merge function is *called from* the
    promote path, nowhere else. (Never a separate scan that "finds" merges —
    that's how position-alone merging sneaks in.)
  - Mechanics: the Identity object already spans relinks (same object through
    LOST→CANDIDATE), so "the candidate span" = this identity's event rows with
    `identity_state == "candidate"` in this window. On AGREE, re-stamp them
    `confirmed_retroactive` (a NEW state value — keep live-confirmed and
    retro-confirmed distinguishable forever; the box score can count both, the
    audit trail stays honest).
  - The contradiction check you specified: before merging, if the confirmed
    number already has confirmed presence overlapping the same frames in that
    window (impossible for one physical player) → do NOT merge; emit a
    contradiction flag to the review queue. It's a free error detector — a
    contradiction means a seed, a relink, or OCR is wrong somewhere.
  - Effort: ~2-3 evenings including unit tests (this one *needs* the tests
    first — it's the one new feature with silent-mis-credit potential; test the
    contradiction case and the never-on-reappearance property explicitly).

**6. Ball detection.**
**Status: absent from this repo entirely** (person-class only everywhere; the
"arc" spikes are 3-pt line drawing, not ball). Prior "flickery but
arc-detectable" evaluation lives only in project memory — treat it as a
hypothesis, not a result. Approach at your level: spike first — YOLO
`classes=[32]` (COCO sports ball) at imgsz 1280-1920, LOW confidence threshold,
on a shot-heavy TEST1 segment; plot raw detections per frame; expect heavy
flicker + false positives (heads, logos). Then the real component: a
**trajectory layer** (associate detections across frames; fit short parabolic
segments; a segment's physics-consistency is its confidence — abstention
built-in: no clean arc, no ball claim). Do NOT fine-tune a custom detector
until the stock-model + trajectory approach is measured. Defer until identity
ships (ROADMAP Phase 5); spike ~2 evenings, trajectory layer ~2-4 weekends.

**7. Shot detection.**
**Status: absent; downstream of #6.** Realistic decomposition:
  - **Attempt detection** (achievable): upward arc terminating near the hoop
    region. Hoop pixel location is FREE from your homography (project court
    (5.25, 25) / (78.75, 25) + rim height heuristic, or one-time click per
    clip). Shooter location = arc origin → court feet via homography → **shot
    location for charts**; nearest CONFIRMED identity at release = shooter
    (with identity_state stamped on the shot event — abstention extends
    naturally: "shot by unconfirmed player #_ (review)").
  - **Make/miss** (hard): ball-through-net is occlusion-plagued at gym camera
    angles. Expect 70-85% at best initially → treat outcome as a *candidate
    label* with a review queue, exactly like identity. Never present unreviewed
    makes as fact. Scoreboard-OCR cross-check is a later "second signal" for
    outcomes (the same design pattern you already own).
  - Shot charts = location + outcome + shooter joined — falls out of the above
    plus matplotlib you already have (heatmap infra reuses).
Effort: attempts+location ~3-4 weekends after #6; outcomes an open-ended
research-y tail — timebox it and ship "attempts + locations + reviewed
outcomes" first. Coach value is already large at that tier.

**8. Possession detection.**
**Status: absent (fixed 2.0s stand-in, honestly labeled everywhere).**
Signals available *before* ball detection: per-frame mean court-x of on-court
bodies (team_events already carry this!) with hysteresis + minimum-duration →
half-court possession segments; direction-of-bulk-motion changes as boundaries.
That's World B's one decent idea, rebuilt deterministically over your trusted
team_events (~1-2 evenings). With ball detection later: ball-side + ball-
proximity refines boundaries and enables team-attribution of possessions.
Effects when it lands (plan for these, don't retrofit): the containment window
and the OCR accumulation window become possession-length (longer windows → more
OCR attempts per candidate → per-possession read rate likely *improves*);
`accumulation_window_seconds` becomes a fallback for clips where possession
detection fails (keep the field; abstention again). Box score windows become
possessions = per-possession stats become real. Sequence AFTER the fair
remeasure so the remeasure has a stable denominator (ROADMAP Phase 3).

**9. Auto-recovery / pipeline resilience.**
**Status: absent, and mostly correctly absent at your stage.** What exists:
stage-separated architecture + caches = manual resumability (re-run one stage);
per-frame calibration confidence = graceful *quality* degradation. What's
missing, tiered: (i) per-stage try/except in run_clip that records
failure-and-continues where downstream doesn't depend (or aborts loudly where
it does), (ii) run manifest (#19) marking which artifacts a run completed,
(iii) mid-clip calibration-collapse policy (if >N% frames low_confidence →
abort with report, don't emit thin stats). (i)+(iii) ~1 evening together; do
alongside possession work. Full job-level recovery belongs to the web phase —
building it now is overengineering.

**10. ROI-mask seeding.**
**Status: absent; first half of the fair remeasure; all parts exist.** At each
window-start seed point: `anchor(f)` (from stage1_court_roi) → per-frame
`p2f` → each track's `feet_pixel()` → `pixel_to_feet` → `on_court()` →
seed only on-court tracks. It's ~20 lines in a helper shared by stages 4/5/6.
One decision to make explicitly: refs are on-court (Phase-1 rule) so they'll
still be seeded — fine for the remeasure (they have no roster number →
`no_position_hypothesis` → they sit in the review queue honestly). ~1 evening
including eyeball verification of seed stills. Do first in Phase 1.

**11. Real per-game rosters.**
**Status: schema built (ClipConfig.teams), data fake.** Needed: actual numbers
+ colors for TEST1 (Milford/Little Miami) and HARD (Winton Woods + opponent)
from film/your dad — an evening of film work, no code. Plus the validate()
gate (#25) so partial/typo'd rosters fail loud, and the seed-labels⊆roster
check with the deliberate-rig escape hatch. This is also a *product* feature:
the web app's roster-entry form maps 1:1 to `Team` — keep that dataclass the
single shape.

**12. Web app integration.**
**Status: absent by design; contract half-formed (ClipConfig).** Right-sized
plan when its phase arrives: (i) define the JSON wire form of ClipConfig
(§2.2/§6.6: ranges → explicit lists or {start,stop,step}, string keys,
uploaded-video reference instead of local path, calibration payload — requires
§6.2 full merge); (ii) CV service stays a **worker, not a web server**: a
Supabase `jobs` table (queued/running/done/failed + progress text + artifact
refs), a Python loop that claims a job, runs `run_clip.run()` **in a
subprocess** (the §6.3 invariant), uploads artifacts (JSONs/PNGs) to storage,
writes results rows. No FastAPI/Redis/Celery yet — the jobs table IS the queue
at your scale. Progress = stage-boundary heartbeats run_clip already prints
(write them to the row). (iii) Results land: `team_events`, `player_events`,
box score, review queue → storage + typed rows the Next.js app reads.
~2-3 weekends when sequenced (Phase 7), near-zero rework if ClipConfig
validation (#25) and manifests (#19) already exist.

**13. Gemini analyst layer.**
**Status: absent (correctly — nothing real to narrate yet).** Build shape:
input = ONE JSON: box score (confirmed + review counts), zone occupancy, team
stats, possession summaries, (later) shot chart aggregates — never video,
never raw frames. Prompt rules: "every number you state must appear verbatim
in the input; if a stat is absent, say it was not measured; distinguish
confirmed from candidate explicitly." Guardrail in *code*, not prompt-trust: a
post-generation check extracting numeric claims from the output and verifying
each exists in the input JSON (regex tier is fine v1); failures → regenerate
or flag. Output = markdown scouting report a coach edits, marked "narrative
derived from measured stats above" so the deterministic table is always
visible truth. Cheap (~1-2 evenings) *once real per-player stats exist* — and
actively harmful before (fluent narration of rigged numbers is confident-wrong
at its worst). Gate: after retroactive merge + real rosters produce a real box
score (ROADMAP Phase 8). Never in the perception path, ever.

### Quality-of-life / high-value additions

**14. Overlay video renders (coach-watchable annotated clips).**
Worth it: **yes, early** — it's your validate-by-eye instrument AND the demo
weapon. Status: all pieces exist separately (stage4_overlay: court, TEST1-
hardcoded; stage2_recovery: state-colored boxes). Build: one generalized
renderer off ClipConfig — frames from config, handoff markers *computed* from
keyframe midpoints, court + identity boxes + jersey numbers on CONFIRMED.
~1-2 evenings. When: Phase 3 (after fair remeasure so the boxes show real
confirms; before dad-facing demos).

**15. Review-queue UI.**
Worth it: yes — abstention needs a place for the human to answer, or the queue
is decorative. Status: data layer built (`{clip}_review_queue.json` + evidence
strings); no surface. V1 (pre-web, ~1-2 evenings): a generated static HTML page
— one row per item: crop image (from #16 machinery), why-string, frame link,
accept/reject radio → downloads a decisions JSON. That decisions file feeds
seeds/corrections back in (via the SAME `seed()` gate — provenance "seed",
human click). Real UI belongs in the web app (Phase 7) reusing the same JSON
contracts.

**16. Jersey-crop montage as a first-class tool.**
Worth it: yes, and **the current tool is lost** — montage PNGs exist in
phase2/out but no script in the repo generates them (built as throwaway,
deleted). Rebuild as `phase2/make_montage.py`: read tracks cache + video, crop
`jersey_crop()` per candidate per sampled frame, tile with track_id/frame
labels. ~1 evening. When: Phase 1 (it's the eyeball instrument for the
remeasure itself).

**17. Multi-game aggregation.**
Worth it: eventually — it's the season-stats product story. Blocked on: real
rosters + jersey-keyed per-game box scores (#1d) + a player-identity-across-
games convention (roster number + team within a season = fine v1). Build:
a small aggregator over per-game JSONs (dict-merge tier, ~1 evening). When:
Phase 7 with the web app (storage gives it a natural home). Not before —
aggregating rigged numbers is negative value.

**18. Export (PDF/CSV box scores, shot charts, reports).**
Worth it: yes at coach-delivery time. CSV of the box score: trivial (~1 hr)
once #1d exists — do it then (it's also your own analysis tool). PDF report
(demo-artifact style: heatmap + box score + narrative): ~1-2 evenings via
matplotlib you already use. When: CSV in Phase 2; PDF in Phase 7/8 alongside
Gemini narrative.

**19. Run manifests.**
Worth it: **yes, early** — reproducibility is your debugging superpower and
two clips are already confusing enough ("which config produced this JSON?").
Build: run_clip writes `{clip}_run_manifest.json`: ClipConfig as dict, cache
fingerprint, refit-npz fingerprint, git commit (`git rev-parse HEAD`), stage
wall-clock times, artifact paths + which stages completed. ~1 evening. When:
Phase 0/3 boundary (cheap enough for Phase 0 if bundled with #22).

**20. Tracks-cache fingerprinting.**
Worth it: **yes — top-priority robustness fix in the repo** (full analysis
§6.4; extends to the refit npz §2.7). Fail-loud staleness = abstention applied
to your own artifacts. ~1-2 h. When: Phase 0, first.

**21. Tiny regression suite.**
Worth it: **yes — the enabler for everything else.** Shape (pytest, minutes to
run, no video needed for the safety tests):
  - Safety-property unit tests over synthetic tracks: continuity cannot
    confirm (provenance refusal); occlusion→LOST attributes nothing;
    relink→CANDIDATE never CONFIRMED; ambiguous→UNKNOWN; window reset empties
    the lost pool (0 cross-window relinks by construction); DISAGREE→UNKNOWN +
    evidence; no-read stays CANDIDATE; seed of unknown track_id is loud (after
    §2.10 fix). This *replaces* broken stage1_states as the IRON-RULE proof,
    running on every change.
  - Golden test: run stages 3-6 + stage2_generate's *reader* against the
    committed TEST1 fixtures and diff outputs. **Blocker: phase1/out + phase2/out
    are gitignored — your golden artifacts (TEST1 team_events, tracks cache,
    player_events, the `_baseline/` dir) are not in version control.** Commit
    the small JSONs as `tests/fixtures/` (they're 0.1-0.7 MB; fine).
  - `zone_of` table test (the spot-checks in zones.main, asserted).
~1-2 evenings. When: Phase 0. Also pin `ultralytics` in requirements.txt while
you're there — it's the *unpinned* half of your validated detector config, and
a silent upstream tracker change would invalidate byte-reproducibility claims
(easyocr is already pinned; do the same for ultralytics + torch).

**22. Structured logging + per-stage timing.**
Worth it: modestly, and cheap. The 90 minutes is already known to be detection
(~2 s/frame CPU × frames); what you lack is per-stage wall-clock in run_clip +
cache_tracks (a `time.perf_counter()` wrapper around each section, written into
the manifest #19). Full `logging` conversion: skip for now — prints are fine
solo; revisit at web time when output must land in job rows. ~1 h bundled with
#19.

**23. GPU path for tracking.**
Worth it: **not yet — but it becomes existential at full-game scale, so plan
it.** Math: ~2 s/frame CPU → a 32-min half (~57k frames) ≈ 32 hours/game.
Caching makes the current 120-frame-span workflow livable; full games do not
fit CPU, period. Decision (Phase 6 gate): local NVIDIA GPU if you have/buy one
(ultralytics uses it with zero code change — install CUDA torch), else a
rented cloud GPU per game (~$0.5-2/game at spot prices, batch overnight). No
code work now beyond keeping tracking behind `iter_tracks()` (already true).

**24. Batch processing.**
Worth it: later, trivial then. `run_clip.run()` is already a function; batch =
a loop + per-clip try/except + manifests (#19) + subprocess isolation (§6.3).
~30-60 min when a real multi-clip night first exists (Phase 6/7). Don't build
a scheduler.

**25. ClipConfig validation.**
Worth it: yes, early (full spec §2.2/§6.6): video exists/opens/frame-count ≥
span end; ranges non-empty and inside clip; rosters non-empty valid ints;
seed_labels ⊆ roster (with explicit rig override); name ∈ clips_config.CLIPS;
accumulation > 0. `validate()` called by run_clip + cache_tracks. ~1-2 h.
When: Phase 0/1 boundary (before the remeasure — it documents the HARD rig
explicitly).

**26. Additional gaps found in this review (same treatment):**
  - **26a. Persist stage-6 OCR outcomes** (§2.12) — prerequisite for merge;
    ~30 min; Phase 0.
  - **26b. Per-player court positions (tracks × homography join)** — described
    in #1e; the cheapest new coach value in the codebase; ~1 evening; Phase 2.
  - **26c. README rewrite** (§5) — ~1 h; Phase 0.
  - **26d. World-B deletion** (§5) — after the suite exists; ~30 min; Phase 0
    tail.
  - **26e. run_clip runs heatmap/demo stages + CLI clip arg + refit reuse**
    (§2.1) — ~1 h total; Phase 0.
  - **26f. `seed()` loud on unknown track_id** (§2.10) — ~15 min; Phase 0.
  - **26g. Landmark auto-propagation across keyframes** (#3a) — the pragmatic
    80% of auto-calibration; ~1-2 evenings; Phase 4.
  - **26h. Pin ultralytics/torch versions** (#21) — ~10 min; Phase 0.
  - **26i. Identity-relink dials onto config/fps-normalized** (§2.10) — when
    clip #3 arrives; ~30 min.
  - **26j. Double-run process guard** (§6.3) — ~15 min; Phase 0 with the §6.2
    guard fix.

---

## Closing note on what NOT to change

The things below look "improvable" and should be left alone; they are pulling
their weight exactly as designed:
- The seed-everyone stand-in *as a mechanism* (ROI-mask changes the pool, not
  the mechanism).
- `OCR_CONFIRM_THRESHOLD = 0.85` — untouched until the fair remeasure, then
  moved only by the §7.4 framework.
- Team events' identity-freedom — permanent; nothing in this review or roadmap
  writes identity into the spine.
- The per-window reset containment — even after possession detection replaces
  the window *lengths*, keep the reset-at-boundary semantics.
- The 4× state-machine recompute across stages 3-6 — inefficiency that buys
  isolation; consolidate only when spans grow.
- spikes/ as-is — it's your calibration engine's lab notebook; renaming/
  moving it buys nothing.
