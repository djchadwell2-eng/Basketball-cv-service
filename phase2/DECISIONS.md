# Phase 2 — settled decisions (individual player identity)

Plain-English record so a future session inherits these without re-deriving them.
Phase 2 sits on NEW `tracks` + `player_events` layers ON TOP OF the identity-free
`team_event` spine (Phase 1). Do NOT modify team_events. No LLM anywhere.

## 1. The identity state-machine CONTRACT (safety-critical — do not violate)
The whole point: **never silently attribute a stat to the wrong player.** A spike
proved position-continuity alone is not safe (one normal single occlusion produced
a silent swap), so confident identity is made structurally impossible without a
second signal.

- **Four states**, carried by every track every frame: `CONFIRMED`, `LOST`,
  `CANDIDATE`, `UNKNOWN`. **Base state is `UNKNOWN`** (abstain — claim nothing until
  identified).
- **`set_confirmed` is a single choke point** (`phase2/identity.py`). It REFUSES
  unless `provenance in {seed, second_signal}`. Position/motion continuity has NO
  valid provenance, so it can produce at most `CANDIDATE` — **never `CONFIRMED`.**
  This is THE safety property: no silent swaps.
- **`LOST` attributes nothing** (occlusion in progress). A reappearance relinked by
  position/motion continuity → `CANDIDATE` **with evidence** (gap frames, distance
  from the motion prediction, reappearance position). An **ambiguous** reappearance
  (2+ lost tracks contend, no clear winner) → `UNKNOWN`.
- **Per-window re-seed boundary** (`phase2/windows.py`): identity resets per window
  (fresh machine, empty lost pool), so **no relink crosses a boundary** — a break in
  window N cannot corrupt window N+1. (Verified: cross-window relinks 1 → 0.)
- **`player_events` are stamped with `identity_state` at the moment.** Uncertainty
  propagates: a candidate's event is a candidate event; during `LOST` nothing is
  attributed. **The box score trusts `CONFIRMED` only** and surfaces
  candidate/unknown events for review — never silently counts them.

- **WARNING to future sessions:** do NOT reduce the review queue by loosening the
  candidate bar or letting continuity auto-confirm. That silently reintroduces the
  swaps this design exists to prevent. **The queue shrinks ONLY by adding an
  independent signal (OCR), never by lowering the confidence bar.**

## 2. The OCR seam — the next build
`promote_via_second_signal()` is a deliberate, unimplemented stub. **Jersey OCR is
the next step, and the ONLY sanctioned path (besides an explicit human seed) to
promote `CANDIDATE` → `CONFIRMED`.** Agreed design:
- **Per-game roster** (both teams: numbers + jersey color) makes it **closed-set
  matching** (read against a known small set, not open-world OCR).
- **Three outcomes:**
  - position + jersey **AGREE** → auto-confirm (silent, allowed — it's a real 2nd signal);
  - **DISAGREE** → abstain / flag (this is the swap-detector);
  - **no confident read** → stay `CANDIDATE`.
- **One visible tunable autonomy threshold**, grown by watching for swaps — never by
  loosening the rule.

## 3. Known stand-ins to replace
- **Seed-everyone** (Stage 4 seeds every track present at a window start) → replace
  with **ROI-mask + per-game roster seeding**, so the box score covers the ~10
  on-court players, not the crowd/bench.
- **Fixed 2.0s window** (`accumulation_window_seconds` — the containment-boundary /
  OCR-accumulation stand-in; code value is 2.0s, not the "~15s" earlier drafts said)
  → replace with **real possession detection**. Neither is final possession logic.

## 4a. FAIR REMEASURE — TEST1 (2026-07-06) — first fair-protocol OCR numbers
The rigged-low caveat is RESOLVED for TEST1's protocol: candidate pool = ROI
on-court only (phase2/oncourt.py, strict majority per (window, track)); roster
= REAL, user-entered from film (Milford white/red: 3, 13, 23, 44, 10; Little
Miami green/yellow: 24, 5, 32, 14, 30). Threshold untouched at 0.85.

Numbers (TEST1 span 300..+120, 2.0s windows, 6 on-court candidates):
- per-FRAME: 60 crops -> 8 any on-roster read (13%), 6 confident >=0.85 (10%)
- per-POSSESSION: 2 of 6 candidates got >=1 confident read = **33%**
- auto-confirms 2/2 EYEBALL-VERIFIED CORRECT (#5 @ 1.00 f=310, #13 @ 0.99
  f=322); disagreements 0; continuity confirms 0.
- Reads that land are near-certain (0.99-1.00): the strict 0.85 bar rejected
  only 2 of 8 reads — the threshold is NOT the bottleneck.
- Expanding the roster 3 -> 10 numbers changed NOTHING (identical 8 reads):
  no raw read was being lost to the small stand-in roster on this span. The
  POOL (crowd) was the binding rig; remaining no-reads are genuinely
  unreadable crops (turned away / small), not filtering.

**DO NOT GATE ON THIS SAMPLE.** n = 6 candidates / 2 windows / 4 seconds.
(Superseded the same day by the WIDE sample below.)

### 4a-WIDE (2026-07-06, same day): span widened to the FULL pan 120..580
(461 frames, 8 x 2.0s windows; window size unchanged — same protocol, bigger
sample; caches rebuilt: 122 track_ids, 461/461 frames anchored, on-court mean
12.7). Fair wide-sample numbers (n = 24 on-court candidates):
- per-FRAME: 232 crops -> 19 any on-roster read (8%), 13 confident (6%)
- per-2.0s-WINDOW: 3 of 24 candidates >=1 confident read = **12.5%**
- outcomes: agree 1 (eyeball-verified: green #5 @ 1.00 f=128), disagree 0,
  no_confident_read 21, **no_position_hypothesis 2** — two PERFECT reads
  (#5 @ 1.00 f=310, #24 @ 0.993 f=562) correctly ABSTAINED because no seed
  hypothesis existed to agree with (the two-signal rule working as designed).
- containment 12 -> 0 cross-window relinks; 0 continuity confirms; coach
  queue 28 on-court items over 15s (124 crowd/bench excluded+counted).

THREE distinct bottlenecks are now separately measurable (do not conflate):
1. READ rate (OCR capability): 6%/frame, 12.5% per 2s window. Reads that
   land are 0.99-1.00 -> the 0.85 dial is still not the bottleneck.
2. WINDOW LENGTH: 2.0s stand-ins are ~7x shorter than real possessions. The
   G1 per-POSSESSION number does NOT exist yet — it arrives with possession
   detection (ROADMAP Phase 3). Do NOT read 12.5% as the G1 rate. (Note:
   stage6 MAX_ATTEMPTS=10/candidate also caps long-window gains; revisit
   with possession windows.)
3. HYPOTHESIS COVERAGE: 2 of 3 confident reads were "wasted" (correctly)
   because only 2 hand seed labels exist. Coach-click seeding (every
   on-court player gets a hypothesis) converts those directly into
   confirm-or-flag. This is a seeding-coverage problem, not an OCR problem.

FINDING — seed labels do not survive re-tracking: seed_labels are keyed by
ByteTrack track_id; rebuilding the tracks cache reshuffles ids, silently
invalidating hand-verified labels (t17=#13 no longer matched anything this
run; t6=#5 happened to survive). Safety held regardless (agree still requires
an OCR match; wrong labels can only produce flags, never wrong confirms) —
but label coverage is unreliable across re-tracks. Real seeding must be
position/click-based per run, or labels re-verified after every re-track.

### 4b. FAIR REMEASURE — HARD wide (2026-07-06): the cross-gym gap + diagnosis
Real roster entered (Milford white/red 1,3,13,24,44; Winton Woods black/green
10,3,23,0,20 — **#3 on BOTH teams**). Seed labels retired (ids don't survive
re-tracking; #30 contradiction flagged to user) -> pure READ-RATE run, confirms
0 by construction. Span 600..1200 (601 frames, 11 windows, 41 on-court
candidates; containment 45->0; 0 disagreements; 0 continuity confirms).

- per-FRAME: 396 crops -> 5 any (1%), 3 confident (1%)
- per-2s-WINDOW: 1 of 41 = **2%**  (TEST1: 6%/frame, 12.5%/window)
- the ONE confident read: **#3 @ 0.999** — the dual-team number; only the
  (unbuilt) jersey-color tiebreak could team-attribute it. First real case.

MONTAGE DIAGNOSIS (scratch crop-tiling of the largest on-court boxes, both
clips, eyeballed): HARD's close-camera zone is occupied by REFS + courtside
adults (a coach in red, a gray-shirted adult at the edge) — actual players
stay small/far for most of the pan. When a HARD player DID get close (t1496,
#24 white/red), the number is PERFECTLY legible to a human — but he was
seeded-confirmed, so stage6 never attempted him (candidates-only policy).
VERDICT: the gap is DISTANCE / crop-size distribution + attempt selection,
NOT jersey contrast (black jerseys never got close enough to be tested).
Implication ranking: (1) best-crops-first attempt selection in stage6,
(2) footage/zoom guidance, (3) color tiebreak for dual-team numbers.
Protocol v1 (stride-sampled attempts) baseline recorded above; v2
(best-crops-first) measured same day:

### 4c. ATTEMPT POLICY v2 — best-crops-first (2026-07-06): ~2x on BOTH clips
stage6 now spends its unchanged attempt budget (<=10/candidate, >=OCR_STRIDE
frames apart) on each candidate's LARGEST boxes instead of stride-from-window-
start. Threshold untouched. Same-day side-by-side (v1 JSONs preserved as
*_ocr_confirms.stride_v1.json):
- TEST1: confident reads 13 -> 22 of 232 crops; windows w/ >=1 confident read
  3/24 (12.5%) -> **6/24 (25%)**. Distinct players read: #5 (x3 windows, 1.00),
  #32 (0.884), #14 (0.923), #24 (0.999). Disagreements 0.
- HARD: confident reads 3 -> 7 of ~395; windows 1/41 (2%) -> **2/41 (5%)**.
  Reads: #24 @ 1.00 (the montage's human-legible close-up player, now found
  because his big crops get attempted) + #3 @ 0.999 (dual-team number).
  Disagreements 0.
Verdict: crop SELECTION roughly doubles read rate at zero safety cost (29
confident reads across both clips, zero contradictions). HARD's remaining gap
vs TEST1 is footage distance (players small across the pan) — the zoom/4K
guidance stands as the product-side lever. Confirms remain hypothesis-gated
(1 on TEST1, 0 on HARD) until click-seeding lands — by design.

## 5. RETROACTIVE STAT MERGE — built 2026-07-06 (Phase 2, unit 1)
`phase2/stage7_merge.py`, wired into run_clip after stage6. Contract (11 tests
written BEFORE the code; suite 37):
- Triggered ONLY by AGREE outcomes — records that exist solely because
  promote_via_second_signal routed through the set_confirmed gate (the gate now
  emits machine.confirmations). The merge takes NO position input: merging on
  reappearance/continuity is inexpressible, not just forbidden.
- Restamps ONLY candidate events -> `confirmed_retroactive` (a NEW event-level
  state, distinguishable from live `confirmed` forever). LOST gaps are never
  invented; unknown events never restamp; un-agreed candidates are untouched.
- CONTRADICTION check: same number already confirmed (live or earlier-retro)
  on overlapping frames in the window -> merge REFUSED + loud flag (an
  upstream-error detector, not an error).
- Writes {clip}_player_events_merged.json (canonical); stage5's raw artifact
  is never mutated. stage6 now persists an identities registry for the check.
First real merge (TEST1 wide): w0 id5 #5 @ 1.00 -> 10 candidate frames
re-credited; ledger exact (candidate 1587->1577, retro +10); 0 contradictions.

## 6. JERSEY-KEYED BOX SCORE — built 2026-07-07 (Phase 2, unit 2)
`phase2/stage8_box_score.py` (+ CSV), wired into run_clip after the merge.
Numbers are the stable player key (identity_ids are per-window internals);
live `confirmed` and `confirmed_retroactive` seconds counted separately AND
jointly; per-player ZONE time from the on-court cache's stored court_feet (a
free JSON join — no new geometry). 7 aggregation tests written first (suite
44). Honesty rules enforced+tested: unnamed-confirmed bucket surfaced (never
dropped/guessed); dual-roster numbers flagged team-AMBIGUOUS (HARD #3 case);
candidates/unknowns to review counts only; "presence-seconds, not game stats"
stamped on every output.
First real outputs: TEST1 — #5 (LM) 1.9s total = 1.6 live + 0.3 RETRO (the
merged span visible in a player line), top zone TOP_OF_KEY; #13 (Milford)
0.9s PAINT; unnamed 101 identities/167.4s. HARD — empty player table by
construction (no hypotheses), unnamed 141/182.5s: the before-picture that
click-seeding (next unit) converts into names.

## 7. CLICK-SEEDING — built + FIRST REAL RUN 2026-07-10/11 (Phase 2, unit 3)
`phase2/make_review_bundle.py` (self-contained HTML, crops as data URIs) +
`roster.load_decisions()` (human labels override legacy seed_labels;
off-roster labels REFUSED loud, 3 tests). A human click is a seed -- same
`set_confirmed` gate, no new path to CONFIRMED.

FIRST REAL LABELING RUN (TEST1, user): 19 of 28 labelable tracks labeled from
jersey crops alone, ZERO code assistance. Result: OCR auto-confirms 0 -> 4 (all
eyeball-verified correct: #5 x2 @1.00, #32 @0.88), 0 disagreements. Merge
restamped 104 candidate frames as confirmed_retroactive. **BOX SCORE: 10/10
roster players NAMED** (was 0/10, 101 unnamed identities/167.4s) -- the
"coach clicks players, gets stat lines" product moment, achieved.
Unnamed remainder: 36 identities/62.7s (harder cases: distant, turned away,
or user-marked unsure) -- correctly NOT guessed.

FINDING -- track-splice defect (t49): user visually caught a track whose 3
crops showed TWO different jersey numbers (#44, #44, #13) -- the ByteTrack
id jumped players mid-track (likely a player collision/crossing). Correctly
labeled unsure (not either number). NOT YET FIXED. Planned fix: an automated
purity check (run read_jersey across a track's full lifespan; >=2 confident
DIFFERENT numbers on one track_id -> flag as spliced, quarantine from
labeling / split at the divergence frame) + spread the bundle's 3 crops
across early/mid/late track life (currently biggest-3, which can miss a
mid-track swap) so a human catches it even without the automated check.

### 7a. HARD click-seeding + the first CAUGHT-AND-CORRECTED label (2026-07-11)
User labeled 18 tracks (+12 refs) on HARD. The safety machinery caught a
mislabel two independent ways: #24's line totaled 21.9s in a 20s clip
(impossible => double credit) AND the merge contradiction check refused to
re-credit #24 (overlapping frames). Crop diagnosis (HARD_check24.png,
early/mid/late per track): t6=#24 ✓, t1496=#24 ✓, t7=**#23** (mislabel) — and
white/red #23 was missing from HARD's Milford roster entirely. User confirmed
both fixes: t7->23, Milford roster += 23. Corrected board: #24 19.5s (legal),
#23 7.2s; **#3 AND #23 are now dual-team => AMBIGUOUS lines** until the color
tiebreak; all 9 distinct roster numbers named. REMAINING: t6/t1496 overlap 37
frames though both genuinely #24 => a splice tail (~1.2s inflation); the
contradiction flag now points at exactly that and keeps refusing the merge —
resolved by the planned track-purity check, next build.

## 8. TRACK-PURITY CHECK — built 2026-07-11/12 (two detectors + spread crops)
A) INTRA-track OCR sweep (phase2/purity.py + cache_purity.py, once per clip):
   >=2 confident DIFFERENT numbers on one track_id => SPLICED => quarantined
   in the review bundle + seed labels REFUSED (roster.resolve_label).
   HONEST RECALL FINDING: 0 convictions on both clips — including t49, which
   IS spliced — because conviction needs confident reads of BOTH numbers, and
   at ~9%-confident-per-crop with <=12 attempts/track that's rare. Verdicts:
   TEST1 20 no_evidence/8 consistent; HARD 53/3. The detector is safe-
   direction but LOW-RECALL on this footage; recorded, not hidden. (Possible
   later: a separate, lower DETECTION threshold — quarantine is safe-direction
   — but that is a dial decision for another day, made on evidence.)
B) INTER-track DISPUTED-frames accounting (stage8): a frame where 2+
   identities claim the SAME number is excluded from the line and surfaced as
   disputed_seconds (one body is wrong — abstention). CAUGHT THREE REAL CASES
   on first run:
   - HARD #24: 0.7s disputed (the known t6/t1496 splice tail; line 19.5 ->
     18.2s honest; the merge-contradiction flag stays consistently stuck on it)
   - TEST1 #32: 0.9s disputed — a PREVIOUSLY UNKNOWN collision among the four
     user-labeled #32 tracks (line 13.6 -> 11.8s; needs a look eventually)
   - HARD #23: 2.2s disputed vs 2.7s counted — NOT an error: TWO DIFFERENT
     players legitimately wear #23 (dual-team) and are on court SIMULTANEOUSLY;
     the number-keyed ledger can't split them. Strongest concrete case yet for
     the jersey-color tiebreak / team-scoped numbers.
C) Review bundle: crops now spread early/mid/late per track (mid-track jersey
   changes always visible to the human — the defense that actually caught t49)
   + red quarantine banners when detector A convicts.
Suite: 55 tests. Every disputed second is surfaced, never counted.

## 9. POSSESSION WINDOWS + LATE SEEDING — Phase 3 v1 (2026-07-12)
`phase2/possessions.py`: windows = detected possessions (mean on-court court-x
side, +/-6ft dead zone, 1.5s hold hysteresis, 4s min length; degenerate ->
LOUD fixed-window fallback). Signal is free (oncourt cache). Inspection
artifact {clip}_possessions.json written every run. Both clips detect as ONE
long L possession (80/82% side agreement; matches left-heavy zone data --
user to eyeball at the printed timestamps). windows.WindowedIdentity accepts
boundary lists; fixed-window path unchanged (stage3 containment demo stays
fixed). Suite 64.

LATE SEEDING (forced by real data): with one long window, window-start-only
seeding lost labeled tracks appearing mid-span (#44 vanished). Rule added: a
human label vouches for the TRACK, so a labeled ON-COURT track seeds at FIRST
APPEARANCE -- but only onto a FRESH (UNKNOWN) identity; a relinked CANDIDATE
carries continuity history the human never vouched and is NEVER late-seeded
(tested). Same seed() gate; no new path to CONFIRMED.

MEASURED RESULTS (2.0s windows -> possessions):
- TEST1: queue 24 -> 9 items (93 -> 35 per min of film, 2.7x); confirms 3;
  RETRO recovery 104 -> 669 frames (3.5 -> 22.3s, 6.4x): one read now
  recovers a whole broken span (#5 = 1.6 live + 13.5 retro). 9/10 roster
  named; #44 honestly in queue (relink candidates are never label-vouched;
  OCR never reads a 44). New contradiction caught+refused (#32 multi-label
  overlap, 48 frames).
- HARD: queue 55 -> 15 (165 -> 45/min, 3.7x); 1 confirm (#24 @1.00, merge
  again REFUSED on the known 3-frame splice overlap -- correctly stuck);
  7/9 numbers named, totals LOWER than the 2s-window board. HONEST READING:
  the 2s regime pumped confirmed seconds by RE-VOUCHING everyone every 2s --
  weak provenance. Possession windows demand one seed + survival + OCR/queue
  for breaks: a STRICTER provenance standard. On HARD's low read rate the
  queue (15 items, each now worth a SPAN) carries what OCR can't.
CONSEQUENCE: queue-resolution v2 (human resolves a candidate -> retroactive
credit through the SAME gate mechanics as OCR agree) is now the highest-value
missing piece, especially for distant footage. Next unit.

### 9a. USER-CAUGHT DETECTOR FAILURE + FIX (2026-07-12, same day)
The "1 possession per clip" output was WRONG on both clips (user eyeball):
TEST1 contains an L->R change mid-span; HARD goes right near the end and the
possession continues past the span. TWO lessons recorded:
1) My corroboration was bogus: zone stats are FOLDED to nearest basket and
   say nothing about court half. Never corroborate side with folded zones.
2) The SIGNAL was right all along (per-second mean/median/half-counts showed
   the user's structure exactly); the BUG was the min-length merge erasing
   possessions TRUNCATED by the span edge (their visible tails were < 4s).
   A truncated possession cannot prove its length -- deleting it is
   confidently-wrong. FIX: edge segments are exempt from the min-length merge
   and flagged partial_start/partial_end (3 new tests; suite 67).
Corrected: TEST1 = L 4.0-16.7s + R 16.7s-end (partial); HARD = L 20-36.5s +
R 36.6s-end (partial). Both 100% side agreement. User to spot-check the two
boundary timestamps (~16.7s / ~36.6s = ball crossing half court).

Final 2-possession boards (same day): TEST1 all 10 numbers named (the R-
possession seed point restored #44/#24/#5-in-w1; a NEW confirm fired 3 frames
after the boundary: #24 @1.00 f=504); queue 14; 3 confirms, 3 merges, 0
contradictions. INSTRUCTIVE HONESTY CASE — #5: her recovery read (f=535) now
falls in possession 2, and retro-credit CANNOT cross a possession boundary
(containment), so her possession-1 broken span correctly waits in the queue
instead of being auto-credited: 15.2s -> 4.3s counted + queue item. That is
the containment principle costing us seconds to protect correctness — the
queue-resolution v2 click recovers it with human authority. HARD: 7/9 named,
queue 23, the #24 splice contradiction still correctly stuck.

## 10. QUEUE-RESOLUTION v2 — built + first real session (2026-07-12)
Human queue resolutions ride the SAME merge machinery as OCR agrees
(stage7._restamp: shared contradiction check + credited-set bookkeeping;
merge records carry source "ocr"|"human"). The reviewer sees crops spanning
the identity's WHOLE span, so the click vouches for everything credited; a
resolution restamps candidate+unknown events of that identity. Guards, all
tested (suite 73): stale resolutions refused when window boundaries changed
(identity ids shift); off-roster numbers refused; OCR-vs-human conflicts
flagged never resolved; rejects recorded, never credited; contradiction check
applies to humans exactly as to OCR.

FIRST REAL SESSION (TEST1): 10 user resolutions -> 6 merged (924 frames =
30.8s recovered), 2 REJECTED as crowd, and 2 REFUSED by the contradiction
check (#13 and #32 spans overlapping already-credited time -- the ledger
defending itself against plausible human error; flagged for re-review, not
counted, not lost). Final board: 10/10 named, 8 players at near-full
coverage; 6 lines carry retro time from OCR + human combined.

BUG FOUND BY REAL USE (fixed + tested same session): stage8 looked numbers up
only in the identities registry, so human-resolved identities WITHOUT a prior
hypothesis (registry number None) dropped ~17s into unnamed despite the merge
stamp on their events. Fix: event-level merge.number takes precedence over
the registry hypothesis.

SCALING ANSWER (user asked; recorded): clicking scales with TRACKER
FRAGMENTATION, not game length (122 fragments/15s currently). Levers ranked:
(1) re-ID tracker (BoT-SORT + appearance embeddings; config-swap experiment,
measurable by fragment count) -- NEXT BUILD; (2) footage zoom/4K -> OCR
resolves queue items before a human sees them; (3) span-length-prioritized
queue with a good-enough cutoff. Bar = "labeling pass + short queue on the
bus ride home," not zero clicks (Hudl uses 24h of humans).

### 10a. HARD QUEUE SESSION — 9/9 NUMBERS NAMED + a third real-use bug (2026-07-12)
User resolved 15 of HARD's 23 queue items (8 named, 7 rejected as
refs/coaches/crowd, 8 honestly left unsure). ALL 8 named resolutions merged
through the shared contradiction-checked path: 642 frames (21.4s) re-credited
as confirmed_retroactive [human]; 0 human refusals; the ONLY contradiction
flag remains the known #24 OCR splice (correctly stuck). Safety counters
clean: 0 continuity confirms, 0 OCR disagreements, no line exceeds the 20s
span (max 15.1s). BOARD: **all 9 distinct roster numbers named** (was 7/9);
#3 and #23 stay team-AMBIGUOUS (dual-team; color tiebreak pending); disputed
seconds surfaced, never counted (#24 1.1s, #3 2.7s); unnamed remainder 15
identities/61.7s.

BUG FOUND BY REAL USE (third one; fixed same session): the review page's
download dropped 17 prior labels — dec/qdec started empty and only rendered
rows could populate them, so labels for tracks NOT shown on a regenerated
page (the shown set changed when possession windows replaced fixed windows)
silently vanished from the re-download. Late-seed vouchers were among them:
running with that file would have UN-named players. Fix: dec/qdec now
initialize from the presets (all saved labels carry through any re-download);
queue-resolution presets carry only when window_boundaries match (identity
ids shift otherwise — stale presets would pre-select wrong rows). Data
repaired by merging the backup (ALWAYS back up decisions.json before a
session). Suite 73 green.

## 11. RE-ID TRACKER PROBE — NEGATIVE RESULT (2026-07-12)
The §10 lever-#1 hypothesis ("BoT-SORT + appearance re-ID = ~4x fewer
fragments from a config swap") was MEASURED and is REFUTED on TEST1.
Read-only protocol: same 461-frame span, same detector, BoT-SORT output to a
separate json (spikes/reid_fragment_probe.py); the real tracks cache, user
labels, and queue resolutions untouched. Two runs:
- v1 stock BoT-SORT + with_reid (model auto = native detector features,
  gmc sparseOptFlow): 122 -> **131** distinct ids; mean lifespan 105.8 ->
  105.6 frames (unchanged); mean tracks/frame 28 -> 30 (keeps more
  low-confidence bodies = more ids).
- v2 ONE variable changed, track_buffer 30 -> 120 (1s -> 4s relink window,
  so re-ID could act across real occlusion gaps): 122 -> **128**. A 4x
  longer window recovered 3 fragments.
VERDICT: no adoption. bytetrack.yaml stays the pipeline tracker. Fragment
count is whole-frame (incl. crowd) — but with mean lifespan flat and a 4s
window doing nothing, appearance relinking simply isn't firing usefully here.
LIKELY WHY (recorded, not further chased): teammates wear IDENTICAL uniforms —
appearance embeddings separate player-vs-crowd, not teammate-vs-teammate, so
the exact splices that cost us clicks (players crossing/colliding) are the
ones re-ID is structurally worst at. A wrong-teammate relink would also be a
silent-swap pressure our identity layer would have to absorb (it would, as
CANDIDATE — but that's added risk for no measured gain).
Options NOT taken (dial decisions for another day, on evidence only):
dedicated re-ID weights instead of model:auto; appearance_thresh below 0.8.
Both fight the same-uniform problem.
REMAINING LEVERS from §10, now the ranking: (2) footage zoom/4K -> OCR
resolves queue items before a human sees them; (3) span-length-prioritized
queue with a good-enough cutoff. Clicking bar stays "labeling pass + short
queue on the bus ride home."

## 4. KNOWN DEBT (logged, not fixed)
- **stage2_generate_events reads clip identity from TWO objects.** Frame range +
  output path come from `ACTIVE_CLIP` (clip_config), but the video path for frame
  extraction (`st.s2.VIDEO_PATH`) and the clip label written into team_events
  (`st.s2.cfg.ACTIVE`) still come from `spikes/clips_config.ACTIVE`. Currently
  reconciled by `run_clip.py` setting `clips_config.ACTIVE = config.name` **plus a
  loud assertion** that the two agree before any pipeline work (a desync would
  extract team_events from the wrong video). **Standalone stage2 invocations bypass
  that sync** — they inherit whatever `clips_config.ACTIVE` happens to be. Root fix
  (later): make stage2/calibration read the video + clip name from `ACTIVE_CLIP`
  (single source of truth). Logged as debt, not fixed now.
