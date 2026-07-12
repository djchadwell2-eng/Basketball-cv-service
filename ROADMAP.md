# ROADMAP.md — from today to "a coach uploads a game and gets player stats + a scouting report"

Written 2026-07-02, from the full read-only review in REVIEW.md. This file is
designed to survive without that conversation: context is restated, every phase
says why it exists, and the rules that produced this plan are at the bottom
(Principles Card). When a future session (human or AI) proposes reordering,
it must argue against the *reasons* written here, not just the order.

**Where the project stands today (one paragraph):** The calibration engine
(pixel→court-feet homography with keyframe re-fit + direct anchoring) is
validated on two gyms with a per-frame quality gate. The identity-free team
layer (team_events → zones/heatmaps) works end-to-end. The individual-identity
layer (four-state machine, `set_confirmed` provenance lock, per-window
containment, jersey-OCR second signal with strict 0.85 auto-confirm) is built
and safety-validated — but on deliberately rigged inputs (seed-everyone pool,
fake 3-number roster, fixed 2.0s windows), so **no current OCR/confirm number
is a quality signal**. One entry point (`run_clip.py`) runs everything from a
per-clip `ClipConfig`. The spine of everything below is **individual player
tracking** — the coach-named must-have.

**The one number that steers the whole roadmap:** the TRUE per-possession
jersey-read rate, measured on fair inputs (Phase 1). It does not exist yet.
Until it does, do not resize any OCR-dependent plan, and never touch
`OCR_CONFIRM_THRESHOLD`.

---

## Phase map (the spine is bold)

| Phase | Name | Rough effort (solo evenings/weekends) |
|---|---|---|
| 0 | Harden the floor | 2–4 evenings |
| **1** | **Fair OCR remeasure** | **3–5 evenings + film work** |
| **2** | **Identity becomes a product number** | **2–3 weekends** |
| 3 | Real windows + watchable output | 2–3 weekends |
| 4 | Calibration at scale (multi-gym) | 2–4 weekends, elapsed over footage gathering |
| 5 | The ball layer (shots) | 4–8 weekends, timeboxed tail |
| 6 | Full-game scale | 1–2 weekends + a hardware/cloud decision |
| 7 | Web app connection | 3–4 weekends |
| 8 | The analyst layer (Gemini) | 2–4 evenings |

Phases 0→2 are strictly ordered. 3 and 4 can interleave after 2. 5 needs 3.
6 needs 5 only for "full stat" games (it can ship position-stats-only earlier).
7 needs 2 (and wants 6). 8 needs 7's real outputs (or at minimum 2's).

---

## PHASE 0 — Harden the floor

**Goal (plain English):** before changing how seeding and rosters work (Phase 1
rewires the identity layer's inputs), build the safety net that tells you
instantly if you broke something, and remove the known silent-failure traps.
Right now the repo's guarantees are enforced by memory and eyeballs; this phase
puts them in code that runs in minutes.

**Why this exists / what it unlocks:** every later phase edits code near the
safety-critical core. Without a regression suite, each edit costs a manual
TEST1+HARD re-validation (hours) or, worse, doesn't get one. With it, the
whole roadmap moves faster than it otherwise could.

**Build order:**
1. **Regression suite** (pytest, no video needed for the safety half):
   unit tests on the identity machine — continuity-confirm refused;
   occlusion→LOST attributes nothing; relink→CANDIDATE (never CONFIRMED);
   ambiguous→UNKNOWN; window reset empties the lost pool; OCR DISAGREE→UNKNOWN
   with evidence; no-read stays CANDIDATE. Plus a `zone_of` table test.
2. **Commit golden fixtures**: TEST1 `tracks_raw.json`, `team_events.json`,
   `player_events.json` into `tests/fixtures/` (they're gitignored today —
   your byte-identical validation currently lives only on one laptop's disk).
   Add a golden test: stages re-run over the fixture cache, outputs diffed.
3. **Tracks-cache + refit-cache fingerprinting**: cache files gain
   video path/size/mtime + span + model; every loader compares against the
   active ClipConfig and refuses loudly on mismatch. (REVIEW §6.4 — the single
   highest-value robustness fix.)
4. **Make the dual-config guard real**: run_clip's current assertion checks a
   value it just wrote (it can never fire). Replace with assertions on the
   *imported stage bindings*: `stage2's VIDEO_PATH == config.video_path`,
   `clips_config ACTIVE == config.name`, `stage CLIP is config`. Add a
   refuse-second-`run()`-per-process guard (one clip per process is a real
   invariant today).
5. **Small loud-failure fixes** (each ≤30 min): `run_tracking.extract_subclip`
   raises on unopenable video / 0 frames; `seed()` warns on unknown track_id;
   stage 6 writes `{clip}_ocr_confirms.json` (persist the most valuable
   signal); fix `stage1_states.py`'s broken IRON-RULE call (or retire it into
   the test suite); `os.makedirs` in stage2_recovery; run_clip also invokes
   `stage3_heatmap` + `stage3_demo` and takes a clip-name CLI arg.
6. **Docs to truth**: rewrite README.md around run_clip/ClipConfig (it still
   documents the rejected World-B draft); then delete World B
   (`process_game.py`, `src/` except `camera_tracking.py`,
   `render_heatmaps.py`, `out.json`, `heatmaps/`) in one commit — git history
   keeps them. Pin `ultralytics` + `torch` versions in requirements.txt (the
   validated detector config is currently version-unpinned).

**Do NOT build yet:** the full config merge (clips_config into ClipConfig) —
that's forced (and paid for once) by the web contract in Phase 7. No logging
framework. No CI service — `pytest` run by hand is enough.

**Done when (eyeball-verifiable):**
- `pytest` runs green in under a minute and *fails* if you hand-hack a
  continuity-confirm into `update()` (try it once, revert — that's the proof
  the net works).
- Deleting one line of the TEST1 fixture cache and running run_clip produces a
  refusal naming the mismatch, not a run.
- `python run_clip.py HARD` produces the heatmap + demo PNGs and an
  `ocr_confirms.json`, and the README describes the system you actually have.

**Beginner mistakes to avoid here:**
- Writing tests against today's *outputs* in fine detail (brittle) instead of
  the *properties* (0 continuity-confirms, stamps present, counts match).
- "While I'm in here" refactors of identity.py — this phase adds guards and
  tests around the core; it does not restyle it.
- Letting the fixture JSONs drift from the code that made them — regenerate
  fixtures ONLY via a documented command, in a dedicated commit.

**Dad validation:** none needed (internal). Effort: 2–4 evenings.

---

## PHASE 1 — The fair OCR remeasure  *(decided order: this comes first)*

**Goal:** produce the project's steering number — the TRUE per-frame and
per-possession confident-read rates, on a candidate pool of actual on-court
players matched against actual rosters. Today's ≈3%/11% came from a pool that
was mostly crowd/refs/turned players against a 3-number fake roster; it is
rigged-low and means nothing except that temporal accumulation works (the
3.7× frame→possession gap).

**Why this exists:** every sizing decision downstream — how big the review
queue is, how much coach clicking the product needs, whether the autonomy dial
can ever move, whether OCR is the workhorse or the assistant — is a function
of this number. Measuring it is cheaper than speculating about it.

**Build order:**
1. **ROI-mask seeding**: at each window-start, seed only tracks whose feet map
   on-court (compose the existing `anchor()` + `pixel_to_feet` + `on_court()`
   — ~20 lines in a helper shared by stages 4/5/6). Refs will still be seeded
   (on-court by Phase-1 rule); they'll have no roster number and will sit in
   the review queue honestly — expected, not a bug.
2. **Real rosters** for TEST1 and HARD in `ClipConfig.teams` — both teams,
   numbers + jersey colors, from film (an evening with your dad; no code).
   HARD's real numbers include 23/30 — the current stand-in roster {5,13,24}
   makes those seeds structurally unconfirmable, which is why HARD showed 0
   confirms; the fair config removes that rig.
3. **ClipConfig.validate()**: video opens; ranges inside clip; span sane;
   roster non-empty; `seed_labels ⊆ roster` with an explicit override flag for
   deliberate rigs. Wire into run_clip + cache_tracks.
4. **Rebuild the jersey-crop montage tool** as a real script
   (`phase2/make_montage.py`) — the previous one was throwaway code and is
   lost; only its PNGs remain. It's the eyeball instrument for this phase.
5. **Run the remeasure** on TEST1 + HARD: record per-frame vs per-possession
   confident-read rate, agree/disagree/no-read counts. **Eyeball-verify every
   auto-confirm and every disagreement against the video** (montage + stills).
   Write the numbers and verdicts into `phase2/DECISIONS.md` as a dated entry.

**Do NOT build yet:** retroactive merge (gated behind this number, by
decision); possession detection (keep 2.0s windows so the measurement has a
stable denominator); any OCR engine swap (measure the current engine first —
and if you ever swap, confidences are not comparable across engines: the 0.85
dial must be re-derived); any threshold change.

**⛔ GATE 1 — the autonomy-dial decision (write the outcome in DECISIONS.md):**
Let R = per-possession confident-read rate on fair inputs, and require
**zero wrong confirms** (eyeball-verified) at 0.85.
- **R ≥ ~50%** → OCR is the workhorse. Proceed to Phase 2 as planned; the
  review queue is a cleanup tool.
- **~20% ≤ R < 50%** → OCR assists; coach seeding + review carry more weight.
  Proceed to Phase 2 unchanged (merge pays off regardless), but invest next in
  crop quality / reader (behind the seam) before ever considering the dial.
- **R < ~20%** → inputs or engine problem. Diagnose with the montage (are
  numbers *humanly legible* in crops? if not, it's resolution/zoom, not OCR);
  try PaddleOCR behind `read_jersey()`; do NOT lower the threshold to
  manufacture confirms — the swap-flag (disagreement) rate is your canary,
  not the confirm count.
Any future dial move: one notch at a time, re-eyeballing confirms, watching
disagreements.

**Done when:** a dated DECISIONS.md entry contains the fair R for both clips,
the gate verdict, and stills proving every confirm was correct — and you can
say out loud what fraction of possessions self-identify.

**Beginner mistakes:**
- Quietly "improving" crops/threshold mid-measurement — freeze the measured
  system (crop geometry, threshold, stride) for the whole remeasure.
- Treating refs-in-queue or turned-away no-reads as failures — both are the
  design working (abstention).
- Forgetting that seeding changes → different candidate pools → the Phase-0
  golden test for stage4/5/6 *will* legitimately change; regenerate fixtures
  deliberately, don't loosen the test.

**Dad validation:** roster entry; and show him the montage — "can YOU read
these numbers?" calibrates what OCR should be expected to do.
Effort: 3–5 evenings + film work.

---

## PHASE 2 — Identity becomes a product number  *(decided order: merge after remeasure)*

**Goal:** turn validated machinery into the thing a coach actually asked for:
"click a player once, get that player's line for the clip" — floor time and
position stats, keyed by jersey number, with honest gaps.

**Build order:**
1. **Retroactive stat merge** — tests first (this is the one feature with
   silent-mis-credit potential):
   - Fires ONLY inside the AGREE branch of `promote_via_second_signal`
     (the same event that calls `set_confirmed(provenance="second_signal")`).
     Structurally never on reappearance/position — merging on position alone
     re-credits the wrong player on a normal occlusion swap, the exact failure
     this layer exists to prevent.
   - Re-stamps the candidate span's events as `confirmed_retroactive` (a NEW
     state value — keep live vs retro distinguishable forever).
   - **Contradiction check**: if the confirmed number already has confirmed
     presence overlapping those frames in that window → do NOT merge; emit a
     contradiction flag to the review queue (a free error detector).
2. **Jersey-keyed box score**: aggregate per player number across windows
   (confirmed + confirmed_retroactive counted; candidate/unknown surfaced),
   replacing the internal `(window, identity_id)` keys in the coach-facing
   output.
3. **Per-player court positions**: join P2 tracks to the P1 homography
   (`anchor(f)` + bbox bottom-center → court feet) and stamp court_feet onto
   player_events → per-player zone time, distance, heatmap. This is the
   cheapest new coach value in the codebase and needs no new perception.
4. **CSV export** of the box score (trivial once 2 exists).
5. **Review bundle v1**: generated static HTML from the review-queue JSON +
   crops — one row per item, accept/reject → decisions JSON that feeds back
   **through the same `seed()` gate** (human click = seed provenance).

**Do NOT build yet:** possession windows (Phase 3); any UI framework (static
HTML is enough to learn what the review flow needs); multi-game aggregation
(one game must be right first).

**Done when:** for TEST1 and HARD you can print "#23: X:XX floor time, N
possessions-equivalent windows, zone breakdown, K events pending review" — and
scrubbing the video confirms the line by eye. The merge tests include the
contradiction case and a never-on-reappearance property test.

**Beginner mistakes:**
- Letting the merge "helpfully" also resolve near-miss cases (close position,
  low-confidence read) — the three outcomes stay exactly three.
- Counting retro-confirmed as if live-confirmed *in the audit trail* — the box
  score may sum them; the events must keep the distinction.
- Building the review UI pretty before building it truthful (crop + why +
  decision is the whole v1).

**Dad validation:** the first real per-player line — have him pick any player
on film and check the number against his own eyes. His reaction here is the
product signal that reorders (or confirms) everything after.
Effort: 2–3 weekends.

---

## PHASE 3 — Real windows + watchable output

**Goal:** replace the last big stand-in (fixed 2.0s windows) with possession
detection v1, and make the pipeline's work *visible* (overlay renders,
manifests) so validation and demos stop being console archaeology.

**Build order:**
1. **Possession detection v1 (no ball needed):** per-frame mean court-x of
   on-court bodies over the trusted team_events + hysteresis + minimum
   duration → half-court possession segments. Deterministic read over existing
   JSON. Keep `accumulation_window_seconds` as the fallback when detection is
   unconfident (abstention).
2. Windows/containment/OCR-accumulation adopt possession boundaries (longer
   real windows → more OCR attempts per candidate → the per-possession read
   rate from Phase 1 likely *improves*; re-record it once, same protocol).
3. **Generalized overlay renderer** (from stage4_overlay + stage2_recovery):
   ClipConfig-driven; court lines + identity-state boxes + jersey numbers on
   confirmed; handoff markers computed from keyframe midpoints, not hardcoded.
4. **Run manifests + per-stage timing**: every run_clip writes config, cache
   fingerprints, git commit, stage durations, artifacts produced.
5. **Resilience tier 1**: per-stage failure policy in run_clip (abort loudly
   where downstream depends, continue-and-record where not); calibration
   collapse policy (>N% low_confidence frames → abort with report).

**Do NOT build yet:** ball-based possession refinement (Phase 5 feeds back
into this); job queues; full logging frameworks.

**Done when:** possession boundaries eyeball-match play flow on both clips
(watch 10 transitions side-by-side with the segment list); the overlay video
shows a confirmed player keeping their number through an occlusion with the
state colors telling the story; a manifest exists for every artifact on disk.

**Beginner mistakes:**
- Over-fitting possession hysteresis to TEST1's rhythm — tune on one clip,
  verify on the other, and accept "unconfident → fallback window" instead of
  forcing a boundary.
- Rendering overlays at full frame-rate/resolution by default (slow) — sample
  or downscale; it's an eyeball tool, not a broadcast product.

**Dad validation:** possession boundaries (he counts possessions off film,
you compare); the overlay video is your best show-and-tell to date.
Effort: 2–3 weekends.

---

## PHASE 4 — Calibration at scale  *(decided order: auto-calibration only after multi-gym proof)*

**Goal:** prove the calibration engine (and its per-clip setup cost) across
several *new* gyms, cut the manual clicking to minutes, and only then decide
what auto-calibration needs to be.

**Build order:**
1. Gather 3–5 new-gym clips (different floors, lighting, camera heights —
   your dad's network is the source). For each: keyframes + landmarks in
   `clips_config.py`, a ClipConfig, refit, eyeball overlay. Record setup
   minutes per clip + fit quality in a table.
2. **Landmark auto-propagation** (the pragmatic 80% of auto-calibration):
   click landmarks on ONE keyframe; project into other keyframes via the
   existing adjacent-keyframe SIFT homographies; verify visually and accept/
   nudge. Reuses functions you already trust; should cut per-clip clicks ~5×.
3. Fix the anchor-failure crash path if not already done (a failed SIFT match
   on a new gym must produce a `low_confidence` frame, not a TypeError).
4. Re-verify the per-frame confidence gate flags real trouble on at least one
   hard clip (motion blur stretch, zoom-in) — the guardrail must earn a save.

**⛔ GATE 3 — auto-calibration decision:** if median setup is ≤ ~15–30 min per
game with propagation, manual calibration is a viable product step (coach or
you-as-service does it) → defer full auto-cal further. If setup stays >1 hr or
error-prone across gyms → prioritize auto-cal, choosing classical line-fitting
only with ≥5 gyms of test footage, clicks kept as fallback. (Learned keypoint
models: evaluate only if classical fails; don't train models solo yet.)

**Do NOT build yet:** Hough/line-detection auto-cal before this footage
exists; support for broadcast footage with hard cuts (state it as a footage
requirement instead: single continuous follow-cam).

**Done when:** the table exists (gym, setup minutes, mean/max ft error,
overlay verdict) for 3–5 new gyms, and the gate verdict is written in
DECISIONS.md.

**Beginner mistakes:**
- Accepting a numerically-good fit without the overlay eyeball (clean numbers
  on a subtly wrong court are the confident-wrong trap).
- Tuning engine constants per-gym "just this once" — per-clip values belong in
  config; the engine stays fixed.

**Dad validation:** he supplies footage variety and confirms overlays "sit on
the paint" per gym. Effort: 2–4 weekends, elapsed over footage gathering; can
interleave with Phase 3.

---

## PHASE 5 — The ball layer (shots)

**Goal:** the stat jump coaches feel most: shot attempts, shot locations, shot
charts — with outcomes handled honestly (as candidate labels with review)
rather than confidently wrong.

**Build order:**
1. **Ball spike** (~2 evenings, measure before building): YOLO sports-ball
   class, low confidence, high imgsz, on a shot-heavy segment; plot raw
   detections; measure flicker/false-positive reality. (No ball code exists in
   the repo today; the prior "flickery but arc-detectable" note is a
   hypothesis to re-verify, not a result.)
2. **Trajectory layer**: associate detections across frames; fit short
   parabolic segments; physics-consistency = confidence; no clean arc → no
   ball claim (abstention).
3. **Shot attempts**: upward arc terminating at the hoop region (hoop pixel
   location projected from court model via the homography — free). Shooter =
   nearest identity at release, stamped with identity_state (an unconfirmed
   shooter is a review item, same pattern as everything else).
4. **Shot location** = arc origin → court feet → shot chart rendering (reuse
   heatmap infra).
5. **Make/miss, timeboxed**: attempt the simple visual discriminators; treat
   outcomes as candidate labels feeding the review queue. Ship "attempts +
   locations + reviewed outcomes" rather than stalling for automatic
   make/miss. (**⛔ GATE 4:** if automatic outcome accuracy on eyeballed
   samples is honestly <~85%, ship review-based outcomes and move on;
   scoreboard-OCR as an outcome second-signal is a later, separate project —
   same second-signal design pattern you already own.)
6. Feed ball position back into possession detection (refines Phase 3).

**Do NOT build yet:** custom-trained ball detectors (only if stock + trajectory
measurably fails); rebound/assist/steal inference (each is its own project on
top of reliable shots + identity).

**Done when:** for one clip, a shot chart whose every dot you can scrub to on
video: right location, right shooter (or honestly flagged), outcome either
auto-correct or in review. Dad confirms the chart against his memory of the
game.

**Beginner mistakes:**
- Chasing make/miss accuracy for weeks — the timebox exists because attempts +
  locations + review already clears the coach-value bar.
- Letting the shot layer write into team_events (it must not — new event
  types sit beside the spine, never inside it).
- Trusting arc fits near occlusions (hands, backboard) — the physics-
  consistency gate must be allowed to say "no claim."

Effort: 4–8 weekends. Needs Phase 3 (possessions give shots their game
context).

---

## PHASE 6 — Full-game scale

**Goal:** go from 4-second validated spans to full halves/games without the
pipeline becoming untrustworthy or un-runnable.

**Build order:**
1. **⛔ GATE 5 — compute decision (forced by arithmetic):** CPU detection at
   ~2 s/frame means a ~32-min half (~57k frames) ≈ 32 hours — not viable. A
   local NVIDIA GPU (zero code change via ultralytics + CUDA torch) or a
   rented cloud GPU per game (~$0.5–2 spot, batch overnight) is required for
   full games. Decide by wallet + logistics; the code is already
   tracking-behind-`iter_tracks()` either way.
2. Streaming frame access replaces load-span-into-RAM in stage 2/6 patterns
   (the current pattern is ~6 MB/frame; a full game does not fit).
3. Batch runner: loop over ClipConfigs, one subprocess per clip (the
   one-clip-per-process invariant), manifests + per-clip pass/fail summary.
4. Auto-recovery tier 2: resumable stages off cached artifacts; partial-result
   reporting ("first half analyzed; second half failed calibration at frame N
   — see report").

**Do NOT build yet:** distributed anything; queue frameworks; live/near-real-
time processing.

**Done when:** one full half runs unattended overnight and produces the
Phase-2/3/5 outputs + manifest, with any failures reported, not silent.

**Beginner mistakes:**
- Scaling before Phase 0's fingerprints/manifests exist (debugging a 6-hour
  run without provenance is misery).
- Letting window count explode memory (per-window machines are cheap, but the
  stage-recompute-×4 pattern and RAM-loaded frames are not — fix the patterns,
  not with a bigger laptop).

Effort: 1–2 weekends + the hardware/cloud decision.

---

## PHASE 7 — The web app connection

**Goal:** the coach-facing loop: upload game + enter rosters → job runs →
box scores, charts, review queue, exports appear in the app.

**Build order:**
1. **Freeze the contract**: ClipConfig's JSON wire form (ranges → explicit
   lists/objects; string keys; uploaded-video reference instead of local
   path). This forces the calibration-config merge (clips_config into
   ClipConfig) — do it here, once, with the Phase-0 golden tests proving
   nothing moved.
2. **Worker, not web server**: a Supabase `jobs` table (queued/running/done/
   failed + progress text + artifact refs); a Python loop claims jobs and runs
   `run_clip` in a subprocess; artifacts (JSONs/PNGs/CSVs) to storage; typed
   result rows for the app. No FastAPI/Redis/Celery — the table IS the queue
   at this scale.
3. Roster-entry UI (maps 1:1 to the `Team` dataclass) and the real review UI
   (replacing static HTML; decisions still route through the seed gate).
4. Exports: PDF report (demo-artifact style) + CSVs downloadable per game.
5. Multi-game aggregation: per-player season lines across processed games
   (roster number + team within a season = identity convention v1).

**Do NOT build yet:** auth/multi-tenant hardening beyond Supabase defaults;
payments; team management features — one coach (your dad) end-to-end first.

**Done when:** your dad uploads a clip + roster from his own browser and gets
the box score + review queue without you touching a terminal.

**Beginner mistakes:**
- Building API surface for features that don't exist yet (contract-first is
  for what Phase 2–5 actually produce).
- Letting the web layer compute *anything* statistical — it displays what the
  CV service's deterministic outputs say, full stop.
- Processing two jobs in one worker process (the invariant: subprocess per
  job).

Effort: 3–4 weekends.

---

## PHASE 8 — The analyst layer (Gemini)

**Goal:** the scouting narrative — an LLM that reads the measured, structured
outputs and writes what a coach would say about them. Text-only, analytics-
layer-only, never watches video, never computes a stat.

**Build order:**
1. Input document: one JSON — box score (confirmed vs review counts), zone
   occupancy, possession summaries, shot-chart aggregates.
2. Prompt contract: every number stated must appear verbatim in the input;
   absent stats are declared "not measured"; confirmed vs candidate always
   distinguished; output is markdown a coach edits.
3. **Guardrail in code, not prompt-trust**: post-generation check extracting
   numeric claims and verifying each appears in the input; failures →
   regenerate/flag. The deterministic tables always render above the
   narrative.
4. Iterate tone with your dad ("does this read like a scout wrote it?").

**Do NOT build:** LLM anywhere in perception; LLM-computed aggregates; agentic
multi-step pipelines. One JSON in, one narrative out.

**Done when:** a generated report for a real game where every sentence
survives your dad's fact-check against the tables, and the guardrail
demonstrably catches a planted hallucination test.

**Beginner mistakes:** shipping narrative before the numbers deserve it
(fluent prose on wrong stats is the confident-wrong trap at its most
persuasive); prompt-tweaking instead of guardrail-writing when it invents a
number. Effort: 2–4 evenings.

---

## Decision gates (the numbers that steer)

| Gate | Where | The question | Branches |
|---|---|---|---|
| G1 | End of Phase 1 | Fair per-possession read rate R (zero wrong confirms at 0.85) | R≥~50%: OCR is workhorse → proceed. 20–50%: proceed; invest in crops/reader next; queue carries more. <20%: fix inputs/engine (montage tells you which); NEVER the threshold. |
| G2 | Any threshold change, ever | Swap-flag (disagreement) rate while lowering one notch | Disagreements appear → step back up; the queue shrinks only via better signals. |
| G3 | End of Phase 4 | Median per-gym setup minutes with landmark propagation | ≤~30 min: defer auto-cal. >1 hr: prioritize auto-cal (classical first, clicks as fallback). |
| G4 | Phase 5 | Automatic make/miss accuracy on eyeballed sample | <~85%: ship attempts+locations+reviewed outcomes; outcomes stay review-based. |
| G5 | Phase 6 | Full-game compute | GPU in hand → local; else cloud GPU per game; CPU full games are arithmetic-impossible. |

---

## PRINCIPLES CARD — carry into every future session

1. **Abstention is first-class.** Unknown beats guessed, everywhere: identity
   states, homography confidence, shot outcomes, possession boundaries. If a
   layer can't be confident, it says so and routes to review — it never
   silently proceeds.
2. **Single choke points with provenance locks.** All confirmation flows
   through `set_confirmed(provenance ∈ {seed, second_signal})`. Every new
   confirming feature (retroactive merge, review-UI clicks, scoreboard
   cross-checks) routes through the SAME gate. Position/motion continuity can
   never confirm anything.
3. **Deterministic stats never route through an LLM.** Code computes; the LLM
   narrates what code computed, in the analytics layer only, with a numeric
   guardrail in code.
4. **Team events stay identity-free forever.** New capabilities are new layers
   beside the spine, never fields inside it.
5. **One atomic change at a time; commit before and after.** Tests (Phase 0)
   exist to keep this fast, not to replace it.
6. **Validate by eye over reported numbers.** Clean-looking numbers on rigged
   or subtly-wrong inputs are not measurements. Every confirm, every overlay,
   every boundary earns at least one human look before it's believed.
7. **The rigged-numbers warning.** Any metric measured on stand-in inputs
   (seed-everyone, fake rosters, fixed windows) is labeled as such and never
   drives a decision. The first real number is Phase 1's fair read rate.
   Never lower `OCR_CONFIRM_THRESHOLD` to chase confirms.
8. **Fail loud on your own artifacts.** Caches, configs, and fixtures carry
   fingerprints; mismatches refuse to run. Silence is the enemy, not failure.
9. **Rosters make OCR closed-set.** Both teams, numbers + colors, entered
   before analysis — a product feature and a correctness feature at once.
10. **Solo-dev bar.** Prefer the 20-line reuse of a validated piece over the
    new framework; defer anything that doesn't unblock the current phase;
    subprocess-per-clip until the one-clip-per-process invariant is
    engineered away.
