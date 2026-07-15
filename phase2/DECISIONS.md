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

### 9b. USER VALIDATION SESSION — boundaries are court-SIDE transitions, not possession starts (2026-07-12)
Joint sanity check (dad-demo substitute): user independently scrubbed both
source videos against 5 system claims. Results:

1. **BOUNDARY SIGNAL real, but the word "possession" overclaims it.** User
   confirmed the ball genuinely crosses half-court at both flagged moments
   (TEST1 16.7s, HARD 36.6s) — the underlying court-side classification is
   measuring the right thing. But user correctly identified that crossing
   half-court is NOT the same as a possession starting: in real play a
   possession begins at the rebound/steal/inbound (usually in a team's own
   defensive end), and the half-court crossing is mid-possession ball
   advancement, not a boundary. **This is not a bug** — it is exactly the
   documented scope of `phase2/possessions.py` (ROADMAP Phase 3: "possession
   detection v1, NO BALL NEEDED"): it detects COURT SIDE as a windowing/
   identity-containment stand-in, not true ball-based possession semantics
   (that's Phase 5, gated on the ball layer). The mechanism is still valid
   for its actual job (a sane point to reset the identity machine) — but the
   word "possession" in code/prints/docs oversells it. **RULE for all future
   coach-facing material: never present window/boundary counts as "N
   possessions" without this caveat.** The box score already uses the
   neutral `windows_present` — keep it that way; do not rename to
   "possessions" anywhere user-facing until Phase 5 makes it true.
2. **OCR ground truth CONFIRMED.** User verified both #24 crops (TEST1
   green/Little Miami, HARD white/red/Milford) — correct number AND correct
   team color on both.
3. **Pure human-click credit CONFIRMED — the queue-resolution v2 bet pays
   off.** HARD identity 103 → #44 (5.3 of 5.8s credited from a single
   queue-resolution click, ZERO OCR backing) — user visually confirmed #44
   by means other than reading the jersey number in the still (which wasn't
   legible at that resolution). First real-world proof that a human's
   whole-span click, unsupported by any second signal, produced a correct
   label — validates the core assumption queue-resolution v2 was built on.

VERDICT: identity/tracking foundation holds up under independent scrutiny
with zero errors found. The dad-demo deferral (2026-07-12, this session)
is reinforced, not reversed: even "possession 1/2" language would need a
caveat a real demo shouldn't require — one more reason the first user-facing
demo should wait for Phase 5's actual stat line.

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

## 12. JERSEY-COLOR TIEBREAK — built + first real run (2026-07-13)
`phase2/color_tiebreak.py` + wiring in `stage8_box_score.py`. Resolves HARD's
two dual-roster numbers (#3, #23) into correct per-team lines, replacing one
blended AMBIGUOUS line, WITHOUT ever guessing.

Design: per-clip TEAM COLOR CENTROIDS built automatically from crops the
system already trusts (every CONFIRMED frame whose number is on exactly one
roster -- HARD's #24/#10/#44/#20/#1/#0/#13 -- is free labeled training data
for that team's jersey on THIS footage; no hardcoded RGB, no new human
input). Mean-BGR crop signature (reuses `ocr_reader.jersey_crop`'s torso
region); `classify_team()` requires the nearest centroid to be >=1.4x closer
than the next-nearest or it ABSTAINS; `classify_identity()` majority-votes
across ~6 sampled frames per (window, identity, number) claim-group, ties
abstain. `build_box_score()` gained an optional `identity_team` override
(default `None` = byte-identical to pre-tiebreak behavior, verified on
TEST1). Disputed (simultaneous conflicting-claim) frames are NEVER
team-attributed by color -- they surface on their own AMBIGUOUS row,
unchanged; color tiebreak only splits SOLE (uncontested) claims. 14 unit
tests first (synthetic solid-color crops, incl. a 50/50-blend abstention
case and a tie-vote abstention case); suite 73 -> 87.

BUG CAUGHT DURING BUILD (fixed before shipping): the first cut of
`_identity_occurrences` collapsed one identity to a single "last event
wins" number, silently EXCLUDING a claim-group whenever the same identity
carried two different numbers across different merge-stamped spans (found
on HARD identity (window 1, id 10): 88 frames fell back to a stale registry
hypothesis of #3 while a separate 13 frames were human-resolved to #13 --
the collapse made the #3 portion invisible to color sampling instead of
attempting it). Fixed: occurrences are now keyed by the (window, identity,
number) TRIPLE, each claim-group sampled and classified independently.
Caught before the feature shipped, not after -- exactly the point of
eyeballing before trusting.

FIRST REAL RUN (HARD): all 6 ambiguous claim-groups resolved by color (0
abstained). Eyeball-verified against real footage (4 of the 6, incl. the
identity with the most floor time): every classification correct --
white/red crops -> Milford, black/green crops -> Winton Woods, no
misattributions found. Board: #3 -> "Milford 6.9s" (was one 9.8s AMBIGUOUS
line); #23 -> "Milford 4.4s" + "Winton Woods 2.7s" (was one 7.1s AMBIGUOUS
line). TEST1 regression check: box_score.json and .csv BYTE-IDENTICAL
before/after (zero ambiguous numbers there, zero-cost path confirmed --
no video read, no [color tiebreak] print).

TWO FINDINGS surfaced by building this, NEITHER silently acted on:

1. **Disputed dual-team-number frames could ALSO be resolved by color, but
   currently aren't -- real credit sitting at zero.** HARD identity (window
   0, id 7)'s entire 81-frame #3 claim is 100% disputed against identity
   4's simultaneous #3 claim (same exact frames, 600-680) -- yet color
   correctly and unambiguously classifies id 4 as Milford (white/red,
   verified) and id 7 as Winton Woods (black/green, verified) in that SAME
   disputed window. This is the #23 pattern from section 8 repeating: two
   REAL players, different teams, same number, on court simultaneously --
   not a tracking error. Because color tiebreak was deliberately scoped to
   leave the disputed-frame mechanism untouched, id 7's 2.7s of genuine
   Winton Woods floor time currently shows as unattributed disputed time
   instead of real credit. NOT fixed this session (scope decision, not a
   bug) -- flagged as the clear next unit if the color tiebreak proves out:
   when a disputed number's simultaneous claimants classify to DIFFERENT
   teams, they are not actually in conflict and both could be credited;
   when they classify to the SAME team (or either is unresolved), the
   dispute is real and must stay excluded exactly as today.
2. **A genuine label contradiction, unrelated to color -- FOUND, VERIFIED,
   AND FIXED same session (2026-07-13):** HARD track 2475
   (window 1, identity 10) was given TWO DIFFERENT roster numbers by two
   DIFFERENT human inputs on the SAME track: Part-1 track-labeling said #3
   (`decisions.json track_labels["2475"] = 3`), Part-2 queue-resolution
   separately said #13 (`{"window":1,"identity_id":10,"number":13}`). Both
   numbers are Milford's, so team attribution is unaffected either way (the
   color tiebreak's own output is correct regardless) -- but the SPECIFIC
   jersey number credited for ~2.9s of floor time is genuinely uncertain.
   ROOT CAUSE: Part-1 track-labels and Part-2 queue-resolutions are two
   independently-gated human-input channels that never cross-check the SAME
   underlying track/identity against each other -- a real gap in the
   contradiction net, LOGGED AS DEBT below (not fixed -- the two channels
   still don't cross-check each other; only this one instance was resolved).

   RESOLVED same session: 4 stills spread across track 2475's full 37.7s-
   39.8s span (incl. right at the #3/#13 seam, 39.5s vs 39.6s) show ONE
   visually continuous player throughout -- same build, same hair, no
   jersey-number change -- ruling out a t49-style splice. A 5x-upscaled
   crop at frame 1155 (38.5s) shows a legible **"3"** on her back. User
   confirmed independently from the same evidence plus a footage check.
   FIX: `HARD_decisions.json` backed up
   (`HARD_decisions.backup-2026-07-13-pre-num-fix.json`), the Part-2
   queue-resolution corrected 13 -> 3, full run_clip HARD rerun. Verified
   effect, isolated and exactly as predicted: #3 6.9s -> 7.3s (+0.4s), #13
   1.3s -> 0.8s (lost its incorrect 0.4s retro credit), every other line
   byte-unchanged. Suite 87 green.

## 13. BALL SPIKE — Phase 5 step 1, POSITIVE RESULT (2026-07-13)
`spikes/ball_spike.py` (new isolated probe, zero edits to pipeline code).
The ROADMAP-mandated measurement BEFORE building any ball code: can stock
YOLOv8m (the SAME yolov8m.pt already doing person detection) see the ball
at all on this footage? Prior "flickery but arc-detectable" note was a
hypothesis to re-verify, not a result — now it's a result.

Protocol: user identified a real shot attempt in HARD.mp4 at ~35-45s;
probe ran raw per-frame detection (COCO class 32 "sports ball", conf=0.05
floor, imgsz=1280 — same as the validated person config) on frames
1020-1380, NO tracker, NO persistence. Output: overlay video + per-frame
JSON log (`spikes/out/HARD_ball_spike_overlay.mp4` / `_log.json`).

MEASURED: 759 raw detections / 360 frames. First read of the data said
"confidence separates ball from junk" (every conf>=0.5 spot-checked still
was the real ball: dribble 0.82, mid-bounce 0.80, loose-ball 0.67, in
flight 0.81). THE USER'S FRAME-BY-FRAME EYEBALL CORRECTED THIS — on the
actual shot arc the boxes are glued to the ball but confidence NEVER
crosses 0.5. The log confirms: the arc at 39.6-40.4s (frames 1188-1211)
is a TEXTBOOK PARABOLA — center rises smoothly to an apex and descends,
~12 px/frame, near-continuous for ~24 frames — at conf 0.05-0.33 the
whole way. Confidence tracks apparent SIZE/BACKGROUND (ball low in frame,
large, plain wall behind -> 0.5-0.8; ball high in frame, small, crowd
behind -> 0.05-0.3), NOT ball-ness. Meanwhile floor-glare false positives
share that same low band but are POSITIONALLY STATIC across frames — the
true ball at low conf moves smoothly, junk doesn't.

DESIGN CONSEQUENCE (the real finding of this spike): a confidence
threshold CANNOT gate ball claims — any threshold high enough to kill
glare also kills the entire rising shot arc, the single most shot-relevant
segment. The trajectory layer must consume ALL low-conf detections and
let PHYSICS CONSISTENCY (smooth parabolic motion frame-to-frame) be the
confidence, exactly as ROADMAP step 2 prescribes — the spike upgrades
that from philosophy to measured necessity. Same shape as identity:
the raw signal (OCR read / ball det) is only evidence; the SYSTEM-level
consistency check is what's allowed to make a claim.

VERDICT: GO for Phase 5 step 2 (trajectory layer). The detector's
POSITIONS are good enough to fit arcs through (user-verified glued-on
boxes + log-verified parabola); its CONFIDENCES are not a gate and must
not be used as one. No custom ball detector needed at this stage, per
ROADMAP's "do NOT build yet" list.

ONE TRAP HIT AND FIXED during the build (same class as §9b's naming trap):
the first run imported `run_tracking` BEFORE setting
`clip_config.ACTIVE_CLIP`, so the temp subclip was named `TEST1_span_*`
while correctly containing HARD footage (the video path is passed
explicitly; only the tempfile NAME came from the stale ACTIVE_CLIP
binding). Data verified correct; script fixed to set ACTIVE_CLIP before
the import, same pattern as reid_fragment_probe.py. The lesson stands:
anything reading module-level clip state must bind AFTER the clip is set.

## 14. TRAJECTORY LAYER — Phase 5 step 2, built + first real run (2026-07-13)
`spikes/ball_trajectory.py` + `tests/test_ball_trajectory.py` (13 synthetic/
regression tests written first; suite 87 -> 100). Turns the step-1 raw
detections into honest BALL-IN-FLIGHT claims. Zero pipeline edits; the ball
layer stays beside the spine, writes nothing into team_events.

Design (each gate justified by a §13 measurement): CHAIN detections by
position (glare is static, ball moves ~12px/frame; MAX_STEP_PX=40, gaps
<= 3 frames) -> de-junk (travel < 30px = static_junk; < 6 points =
too_short) -> FIT short quadratics (>= 8 points), claim ARC only when
physics passes: downward accel in a measured band, mild x curvature,
RMS <= 3px, and >= 25px of real vertical travel. Fail = no_claim,
surfaced never dropped. ALL confidences consumed (per §13, no conf gate).

FIRST RUN (HARD spike log, 759 dets/360 frames -> 140 chains): ground
truth PASSED — the user-verified shot arc claimed exactly (frames
1188-1211, accel 0.886, rms < 1px), and the post-shot descent claim
STOPPED at the floor bounce (the gate refusing to fit through a bounce =
abstention working). User eyeballed the overlay: shot-arc curve glued to
the ball; ~65% of the shot covered (apex + descent — the detector went
blind for a few frames at RELEASE, so the claim starts near apex; logged
below as a real gap for steps 3-4, not papered over).

TWO FALSE-CLAIM CLASSES CAUGHT BY EYEBALL + DATA, BOTH FIXED BY MEASURED
GATES (not by hand-tweaking until the answer looked right):
1. Camera-pan glare drift: 6 first-run "arcs" moved 100-200px horizontally
   but 4-50px vertically at floor height, accel 0.10-0.15 (the band edge),
   all drifting in pan lockstep; user confirmed = floor glare dragged by
   the pan. Fix: ACCEL_Y_MIN 0.1 -> 0.3 (every real arc measured >= 0.89).
2. One 8-frame slice of a glare chain then squeaked in at accel 0.309 with
   only 11px of vertical travel — curvature fit to noise. Fix:
   MIN_Y_RANGE_PX=25 (real arcs span 48-351px of y). That exact chain is
   now a literal-data regression test.

FINAL BOARD: 8 arcs / 7 chains, all real (shot arc + descent, dribble
bounces, the 37.9s pass, 42-44s bounces), zero glare claims, 16
static_junk + 16 no_claim + 101 too_short honestly surfaced.

KNOWN GAPS carried to steps 3-4 (logged, not hidden): (a) release-point
blindness — arc claims start near apex when the detector misses the ball
leaving the shooter's hands; "shooter = nearest identity at release" and
"shot location = arc origin" must handle a claim that starts mid-flight
(back-extrapolation of the fitted parabola is available but is a CLAIM
EXTENSION and needs its own gate + eyeball). (b) Non-shot flight (dribbles,
passes) is correctly claimed as flight; step 3 must select shots by "arc
terminating at hoop region," never by assuming every arc is a shot.
(c) v1 does not model camera pan; pan showed up as the glare-drift false
class, killed by the accel+y-range gates, but a fast pan during a real arc
could distort a fit — revisit only if a real arc fails on other footage.

## 15. SHOT ATTEMPTS — Phase 5 step 3, built + first real run (2026-07-13)
Two new pieces, both tests-first, zero pipeline edits (ball layer stays
beside the spine): `spikes/hoop_anchor.py` (carries a rim pixel through
the pan) + `spikes/shot_attempts.py` (arcs -> shot claims + shooter join).
Suite 87 -> 115 across this session's three units.

HOOP ANCHOR: no calibration landmark gives an elevated rim pixel (every
COURT_MODEL tag is a floor point). Design: mark the rim ONCE in a keyframe
still (user-confirmed, click-seeding philosophy), carry it to every frame
via the SAME machinery that already draws the court overlay (Hs_opt
keyframe->ref900 transforms + per-frame best-match SIFT). Valid because
these are ROTATION-ONLY camera homographies -- one homography relates
every scene point regardless of depth for a panning/tilting (non-
translating) camera, elevated points included. Marked HARD's far rim at
keyframe-1100 px (1855,228); user confirmed the marked still. Carrying
run: 360/360 frames matched (100%, zero abstentions), including frames
matched to a DIFFERENT keyframe (1200) than the anchor -- user-verified
"glued to the hoop" against stills, confirming the CARRY math, not just
the anchor point. TRAP HIT + FIXED: same class as ball_spike.py --
`spikes/clips_config.ACTIVE` binds stage1/2/4/5 at IMPORT time; the first
run silently computed TEST1's keyframes because only `clip_config.
ACTIVE_CLIP` had been set, not `clips_config.ACTIVE`. Fixed at the top of
hoop_anchor.py with a comment pointing at this exact trap.

SHOT CLASSIFIER: a claimed arc (already physics-gated, DECISIONS 14) is a
SHOT ATTEMPT iff, at or after its apex, it passes within HOOP_RADIUS_PX
(100px -- slack for the ~20-40px anchor offset already observed plus ball
size/motion blur) of the carried hoop position at that SAME frame.
Floor-level flight (dribbles) fails this by geometry alone -- the hoop
sits high in the frame, a dribble apex sits near the floor -- no special
case to keep in sync. SHOOTER: nearest tracked body (feet pixel) to the
arc's FIRST claimed point (release -- already an approximation per 14's
release-point blindness), joined to that track's identity_state; no data
(frame outside the tracks-cache span, or the track has no identity event)
-> an honest review item, never a guessed shooter.

FIRST REAL RUN (HARD, 8 claimed arcs): GROUND TRUTH PASSED. The
user-verified ~40s shot (1188-1211) claimed correctly, min_dist=54.6px;
shooter = track 16, identity_state=candidate -> correctly surfaced as a
review_item, NOT auto-attributed (an unconfirmed shooter must never be
silently credited, same rule as the identity layer everywhere else). The
other 6 arcs (dribbles, passes, bounces) correctly scored not_shot.

REAL FINDING, NOT SILENTLY FIXED: a SECOND arc (1217-1250) also passed
the hoop-proximity gate (min_dist=83.7px). Chain data shows why: arc 1
ends at frame 1211 moving (+14px/frame, +11px/frame); arc 2 begins 6
frames later at a position inconsistent with that velocity extrapolated
forward -- a real, measurable velocity change. USER EYEBALLED THE OVERLAY
AND RESOLVED IT: one shot, ball clips the rim (rim-out miss) around
40.4s, falls to the floor by 41.7s -- NOT a second attempt. The
trajectory layer is working exactly as designed (a rim deflection is a
genuine new physics segment, correctly re-fit as a new arc per DECISIONS
14) -- the gap is that "one claimed arc" and "one shot attempt" are NOT
the same concept once a shot deflects off iron. shooter for arc 2 =
no_identity_data (frame 1217 is outside the 600-1200 tracks-cache span --
also correctly abstained, and notably this arc having NO independent
shooter is further evidence it isn't a real second attempt).

KNOWN GAP, LOGGED NOT FIXED: shot attempts can be OVERCOUNTED when a miss
deflects off the rim/backboard into a second physics-consistent segment
that also happens to pass near the hoop again (a rebound arc, a
backboard-then-rim carom, etc.). Root fix options for later (deliberately
not built now, timeboxed like everything else in Phase 5): (a) merge
chains whose gap is small in both time and space before hoop-proximity
classification; (b) require a shot's descending segment to ORIGINATE
outside the hoop region (a segment that STARTS near the hoop is more
likely a deflection continuation than a fresh release) -- (b) is the
more principled version of (a) and is the leading candidate if this
becomes a real accuracy problem on more footage. NOT fixed today because
Phase 5's stated policy is measure-first, timebox everything, and this is
exactly one real example, not yet a measured error RATE.

VERDICT: GO for step 4 (shot location) with this limitation carried
forward explicitly -- a coach-facing shot count from this pipeline must
not be presented as exact until (a)/(b) above is addressed or measured
negligible on more footage.

## 16. SHOT LOCATION — Phase 5 step 4, built + first real run (2026-07-13)
Two pieces, both tests-first, zero pipeline edits: a release-back-
extrapolation fix inside `spikes/shot_attempts.py`, and new
`spikes/shot_location.py` (oncourt join + shot-chart render). Suite
115 -> 128 across this unit.

OPENING FINDING, diagnosed before building: step 3's shooter hint (nearest
tracked body to the arc's FIRST claimed point) was WRONG on the real HARD
shot. It pointed at track 3317, a bystander on the far baseline, not the
shooter. Cause: the arc claim starts near apex (DECISIONS 14's
release-blindness), several frames after the real release, so "nearest
body to the first point" finds whoever stands under the apex. Safety was
never compromised (it was a review_item, never auto-credited) but the
hint would have sent a human reviewer to the wrong player.

FIX: `find_release()` extrapolates the claimed arc's OWN fitted quadratic
BACKWARD a bounded window (<=10 frames), at each step measuring distance
to tracked bodies' BBOXES (release happens at the hands, not the feet).
The closest match within a distance gate becomes the shooter hint +
release-frame estimate; nothing under the gate -> honest
no_confident_shooter. This is a genuine CLAIM EXTENSION (extrapolating
past measured data) so it is bounded, gated, and only ever produces a
review hint -- never an auto-attribution. Re-run: new hint = track 1502,
release_frame=1178 (39.27s). USER CONFIRMED against a marked still: "Yes
it is" -- the white/red player mid-shooting-motion, arms up in
follow-through, correctly picked over the old wrong bystander hint. User
also noted she's slightly airborne at the estimated release frame (a jump
shot) -- logged as a small honest caveat, not a fix: a jump shot's
horizontal court position barely drifts between takeoff and release, so
this is not expected to meaningfully move the location, but it is NOT
zero and is not corrected for.

SHOT LOCATION: read directly from the oncourt cache (court_feet at the
shooter's release frame) -- a free join against data the identity layer
already trusts, NOT a re-projection of the ball's own elevated pixel
through the floor homography (that would smear an elevated point to the
wrong floor spot, same reason the hoop needed its own anchor in section
15). Trusts the oncourt classifier's own on/off-court abstention rather
than second-guessing it. First real result: (68.7, 42.3) ft, ~20.0 ft
from the right hoop center (78.75, 25) -- right at the 3pt line (radius
19.75 ft). The rim-deflection arc (section 15) correctly resolved to
location_unknown (no confident shooter -> no location, never guessed).

REAL BUG CAUGHT BY USER EYEBALL, ROOT-FIXED: the FIRST shot-chart render
was mirrored top-to-bottom relative to the real shot location. Root
cause: `phase1/stage3_heatmap.py` (already validated, Phase 1) draws with
matplotlib `origin='lower'` -- near-sideline y=0 renders at the BOTTOM of
the image, far-sideline y=W at the TOP. The new cv2-based shot chart
plotted `court_feet` y straight into image rows (cv2/numpy: row 0 = top),
silently flipping near/far relative to the established convention. The
underlying court_feet DATA was never wrong (sourced from the trusted
oncourt cache, unchanged) -- this was a render-only bug in the brand-new
script, caught because the user checked the picture against their memory
of the footage instead of trusting the number. Fixed with one small flip
helper (`_court_feet_to_diagram_px`, y' = COURT_WID - y before scaling)
applied uniformly to both the court geometry and the shot dots, plus a
regression test asserting a near-sideline point renders in the bottom
half of the image. Re-rendered; USER CONFIRMED correct.

VERDICT: GO for step 5 (make/miss, timeboxed per ROADMAP Gate 4) with the
airborne-at-release caveat carried forward as a documented, unfixed,
believed-negligible approximation -- same honesty standard as every prior
step. Two real defects were found and root-fixed this session (wrong
shooter hint from release-blindness; mirrored chart from an orientation
mismatch with established code) -- both caught by an eyeball check against
real footage, neither by the test suite, which is exactly why the eyeball
gate stays mandatory at every step rather than becoming optional once
tests are green.

## 17. MAKE/MISS — Phase 5 step 5, timeboxed, GATE 4 UNMEASURABLE (2026-07-13)
`spikes/shot_outcome.py`, tests-first (12 synthetic), zero pipeline edits.
Suite 128 -> 140. Two geometric signals over data steps 2-4 already
produced: MAKE evidence (raw, unclaimed detections falling through a
narrow corridor below the hoop, y increasing -- uses raw detections
because a net-drop may be too short to earn ball_trajectory's own arc
claim) and MISS evidence (a chain whose earliest point in the post-shot
window starts near the hoop and whose later points clearly escape it --
the exact rim-out shape from section 15/16, now read as a real signal
instead of an accidental second "shot attempt"). Both signals present ->
unknown (conflict, never resolved by guessing); neither -> unknown (no
evidence). Every output is a CANDIDATE label per ROADMAP -- this module
never claims a stat, only proposes one for review.

GROUND TRUTH: the known HARD shot (1188-1211) classified candidate_miss,
matching the user-verified rim-out from sections 15/16 exactly. Evidence
traced to the SAME deflection chain already found in section 15 (starts
26.2px from the carried hoop position at frame 1217, ends 344.2px away
at frame 1257) -- a satisfying closure: the thing that was a measured
FALSE POSITIVE for "shot attempt" in step 3 is the CORRECT signal for
"this shot missed" in step 5. Zero make evidence found, correctly -- the
ball never entered the below-rim falling corridor because it deflected
away instead of dropping through.

GATE 4, STATED HONESTLY: ROADMAP's gate asks whether automatic outcome
accuracy on eyeballed samples clears ~85%; that requires an accuracy RATE,
which requires more than one sample. This session produced exactly ONE
real shot with a user-verified outcome (a miss) and the classifier got it
right -- 1/1 is not a rate, it is an anecdote. Gate 4 is explicitly logged
as UNMEASURABLE this session, not passed and not failed. NOT built as a
workaround: synthetic shots, assumed accuracy, or shipping this as
"validated." The honest next step (handed to the user, not decided here)
is harvesting more real shots to get an actual sample -- e.g. a full-clip
HARD ball-detection run (the ~91s clip vs. today's 12s span) would surface
every shot in the game for real eyeballed accuracy scoring, at the cost of
a ~90min background run (same order as the existing tracking caches).

VERDICT: the discriminator design is sound and ground-truth-consistent,
but per ROADMAP's own instruction ("do not chase make/miss for weeks"),
this is where step 5 stops for now without a real sample to validate
against. Attempts + locations + this reviewed (never auto-trusted)
outcome are the shippable unit, exactly as the ROADMAP anticipated for
the low-accuracy case -- except here the honest state isn't "measured low
accuracy," it's "not yet measured at all."

## 18. GATE-4 HARVEST — full clip, second hoop, origin gate (2026-07-14)
User chose to harvest more shots after step 5 (section 17) left Gate 4
unmeasurable at n=1. Multi-part session, each fix verified before the
next: full-clip processing exposed a real geometry bug (fixed), then a
real scoping limit (fixed by adding a second hoop), then a real
double-counting bug the user caught by eyeball (fixed + regression-
tested against all real data found).

BACKED UP the verified 12s-span artifacts before any overwrite
(`.backup-2026-07-13-pre-fullclip.json` suffix), per standing practice.

FULL-CLIP BALL DETECTION: `ball_spike.py 0 2746` (whole 91.5s clip, CLI
span override added, default unchanged) -> 3102 raw detections, 65.5% of
frames with >=1 detection (consistent with the original 12s sample's
rate). `ball_trajectory.py` (unchanged) found 30 candidate arcs across
the full game, up from 2 in the original slice.

BUG FOUND + FIXED: `hoop_anchor.py 0 2746` FIRST run reported "hoop pixel
found in 2746/2746 frames" -- but 1089 of those (all matched to
keyframes 600-1000, outside the previously-validated 1100/1200 range)
were geometrically absurd (e.g. x=42625 in a 1920px-wide frame). Root
cause: a SIFT match can clear MIN_INLIERS on OTHER visible features
(floor lines, bleachers) while still being poorly conditioned for
extrapolating specifically to the rim, if the rim sits outside the
convex hull of matched keypoints for that view -- a near-zero
perspective-divide denominator, NOT a sign bug (verified: negating a
homogeneous matrix cannot change its projective-divide output, since
numerator and denominator flip together). Fixed: `in_plausible_bounds()`
rejects a hoop position wildly outside the frame, treated as an honest
no-match. 5 tests; suite 140->145.

SCOPING FINDING: re-run clean (1702/2746 far-hoop matches, zero
out-of-bounds), covering ~33s-91s -- BETTER than expected (the camera's
framing stayed close to keyframe 1200's view for most of the second
half, not just the originally-calibrated 20-40s pan). But shot-attempt
count STAYED AT 2 (both re-confirming the same known shot) despite 30
candidate arcs across a 58s-covered window. Diagnosis: several
non-qualifying arcs DID have a valid hoop position at their check frame
but sat 295-1043px away -- the signature of shots at the OTHER basket,
which was never anchored. Correct abstention (never guessed those were
misses), but it capped the real sample regardless of how much footage
got processed.

SECOND HOOP ADDED: user confirmed a near-hoop anchor at keyframe-600 px
(633,190) after 3 rounds of on-image fine-tuning. `hoop_anchor.py`
generalized to carry BOTH anchors through the SAME per-frame SIFT match
(one extra point projection per frame, zero extra SIFT cost). Coverage
turned out clean and complementary: near hoop 0-40s (100% for 0-30s),
far hoop ~33-91s -- together spanning nearly the whole clip. Zero
out-of-bounds leaked through for either hoop. `shot_attempts.py` /
`shot_outcome.py` orchestration updated to try both hoops per arc and
use whichever matches (the already-tested pure classify_shot/evidence
functions were NOT touched -- only main()'s hoop-lookup construction
changed). Suite unaffected (145 green). Result: shot-attempt count
DOUBLED to 4 (2 new candidates at the near hoop, ~12.7s and ~14.6s).

USER EYEBALL ON THE 2 NEW CANDIDATES, TWO FINDINGS, NEITHER SILENTLY
PATCHED:
1. **356-381 (real shot): confirmed real, but "the camera moved mid arc
   [so] the arc was messed up."** First CONFIRMED real instance of a
   risk DECISIONS 14 had only flagged hypothetically (no pan model in
   v1). Measured: the carried hoop position (a fixed world point) drifts
   ~300px in image-pixel space across this 25-frame shot, purely from
   camera motion -- proof the ball's own fitted curve was distorted by
   the same pan. Practical impact stayed CONTAINED here (still correctly
   classified as a shot attempt, min_dist 19.4px; its outcome correctly
   abstained as unknown rather than guessing through the distortion) --
   but this is now a measured, not hypothetical, gap. Root fix (camera-
   motion-compensated fitting) logged as KNOWN DEBT, not built --
   bigger than today's timebox; revisit if an arc fails OUTRIGHT rather
   than just distorted, or on user request.
2. **418-438: user identified this as "a shot falling down after a
   shot"** -- i.e. NOT a fresh attempt, and asked "is there a way
   around this?" YES: this is the exact double-counting pattern from
   section 15 (rim deflection re-fit as a new arc), now observed a
   SECOND independent time. Measured across all 4 arcs found this
   session, a clean and consistent split: real shots (356-381, 1188-
   1211) originate 285-338px from the hoop and arrive 19-55px away;
   both false positives (1217-1250, 418-438) start 26-69px away and
   END 374-384px away -- moving OUT, not arriving. FIXED:
   `classify_shot` gained an ORIGIN GATE -- an arc whose first point is
   already within HOOP_RADIUS_PX of the hoop at that frame is rejected
   as a continuation, regardless of what the apex-based descent check
   finds. Skips the gate (no rejection) when the first-frame hoop
   position is unknown, so absence of data can never manufacture a
   rejection. 6 new tests, 4 of them literal-data regressions built
   from the exact real chains (both directions: real shots must survive,
   both known deflections must be rejected); suite 145->151. ACCEPTED
   TRADE-OFF, logged not hidden: a genuine close-range layup released
   near the rim would also fail this gate -- not yet observed on real
   footage; revisit if it is.

FINAL RESULT after both fixes: shot-attempt count is 2 -- but a
DIFFERENT, CLEANER 2 than section 17's: two independent genuine shots at
two different baskets (356-381 near, 1188-1211 far), both correctly
surviving the origin gate, both correctly NOT auto-attributing an
unconfirmed shooter. The new near-hoop shot's outcome is honestly
`unknown` (no evidence either way) -- plausibly a downstream consequence
of the same camera-pan distortion noted above, not investigated further
this session.

GATE 4 STATUS: still UNMEASURABLE. n=2 genuine, independent shots is a
real improvement in QUALITY over section 17's n=1-plus-a-duplicate, but
it is still not a rate. The harvest's actual yield was two found-and-
fixed bugs (implausible extrapolation, arc over-counting) and one
found-and-logged limitation (camera pan), not a bigger accuracy sample
-- the single-basket coverage gap turned out to be the binding
constraint, now removed, but this specific clip apparently has few
enough clearly-resolved shots in view to still leave the sample thin.
Next honest step for a real Gate-4 rate: more games/clips, not more
processing of this one.

## 19. TEST1 + THE APEX-RULE FIX — second clip, first full attribution (2026-07-14)
User feedback opened this unit twice over: (1) chose to grow the Gate-4
sample by running Phase 5 on TEST1 (second real game, already calibrated
+ identity-resolved, zero new footage needed); (2) explicitly reaffirmed
priorities -- "ship ASAP" means defer polish, NEVER correctness: "if its
not working then we dont even have an MVP" (recorded in auto-memory as
standing feedback).

MULTI-CLIP GENERALIZATION: ball_spike/hoop_anchor/ball_trajectory/
shot_attempts/shot_outcome/shot_location all take an optional clip name
(default HARD, behavior unchanged). Real bug found DURING the refactor:
reading sys.argv at hoop_anchor's module level broke pytest's import of
it (pytest's own argv misread as a clip name) -- guarded by __main__.
TEST1's two rims marked + user-confirmed (far: kf-120 px (582,143);
near: kf-580 px (1377,233)). Full-clip runs: 472 raw ball detections
(31.9% of frames -- HALF of HARD's 65.5% rate; harder footage for the
ball detector), hoop coverage complementary near-full-clip, zero
out-of-bounds, 13 candidate arcs.

FIRST RESULT: 0 shot attempts -- while the user counted ~4 real ones
(incl. layups). User's hypothesis: the origin gate (added from their own
deflection feedback in section 18) over-restricted. DIAGNOSIS SAID
OTHERWISE, and this matters: the origin gate rejected ZERO arcs on TEST1.
The real defect was the AT/AFTER-APEX rule from section 15's original
design: on truncated ascent-only arcs (TEST1's sparse detection loses the
ball near the rim, in backboard/body clutter) the last observed point
computes as the "apex", excluding nearly the entire arc from the
proximity check -- arc 315-327's genuine 101px approach at f=324 was
ignored as "pre-apex".

REDESIGN, measured on ALL 43 arcs across both clips (the same
data-first discipline as the origin gate): real shots' closest approach
-- observed, or via a bounded forward extension of the arc's own
already-physics-gated fit -- is <= 110px; every non-shot arc is >= 163px.
A clean separation gap. Three changes, tests-first (suite 151 -> 156,
incl. 2 literal-data TEST1 regressions):
  1. ALL observed points count for arrival. The origin gate provably
     covers the launch-proximity case the apex rule guarded (the existing
     synthetic test now rejects via the origin gate instead -- verified,
     not assumed).
  2. HOOP_RADIUS_PX 100 -> 125: inside the measured 110/163 separation
     gap, consistent with section 15's measured anchor slack (~20-40px)
     + ball size. NOT gate-loosening-to-chase-a-case: the same radius
     drives the origin gate where BIGGER = STRICTER, and both known
     deflections stay rejected.
  3. ARRIVAL FORWARD EXTENSION: when no observed point arrives, extend
     the fitted quadratic forward <= 15 frames (~0.5s, half a flight),
     DESCENDING portion only (past the fit's own apex), hoop position
     required per predicted frame. Same bounded-claim-extension pattern
     as the release finder, opposite direction. Arrivals found this way
     are stamped "arrival": "extrapolated" (vs "observed") -- review
     always sees the evidence class.

VERIFIED: HARD ground truth BYTE-STABLE (exactly the same 2 shots, same
distances). TEST1: 2 shots claimed -- 59-71 (2.0-2.4s, extrapolated
arrival 87.2px) and 315-327 (10.5-10.9s, observed 101.5px). The second
is the project's FIRST FULL-CHAIN ATTRIBUTION: ball -> arc -> hoop ->
release extrapolation landing at 0.0px on a tracked body's bbox ->
track 87 -> identity 40 -> jersey #14 (Little Miami), an identity the
user's own review session had confirmed (confirmed_retroactive, source
human). Status still review_item by policy (confirmed_retroactive is
deliberately not auto-"attributed" -- an open policy question, noted
below). Location: (-0.6, 21.0) court-ft -- a baseline shot ~7ft from
the far hoop; x=-0.6 is nominally behind the baseline, within the
calibration's ~1ft error for a baseline release, flagged not hidden.

HONEST LIMITS, logged:
- The user counted ~4 TEST1 attempts incl. layups; the pipeline found 2.
  The other ~2 never formed arcs AT ALL: raw at-rim detections exist
  (e.g. 51px from the hoop at 5.8s, 23px at 37.5s) but the flights are
  too short/sparse to pass the physics gate. That is a DETECTOR-COVERAGE
  limit on this footage (31.9% detection rate), not a gating problem --
  no gate change can recover a flight that was never tracked. Known
  lever remains footage quality (zoom/4K, DECISIONS 4c), or later a
  rim-region detection-sensitivity unit.
- Outcomes for both TEST1 shots: unknown (no evidence either way) --
  the same sparse detection that truncates the arcs also starves the
  outcome discriminators. Honest unknowns, not guesses.
- Policy question deferred to the user: should a confirmed_retroactive
  shooter be "attributed" like a live confirmed one (box score already
  trusts both, reported separately)? Conservative review_item for now.

GATE-4 TALLY after two clips: 4 genuine shot attempts (2 HARD + 2
TEST1), 1 with a user-verified outcome, 2 with honest unknowns, 1
unverified. Still not an accuracy rate; the sample grows by clips, and
each clip is contributing ~2 scoreable shots at current footage quality.

## 20. BALL-SEEING RESOLUTION SWEEP — NEGATIVE RESULT (2026-07-14)
User raised the "ball seeing problem": TEST1's ball-detection rate is
half of HARD's (32% vs 66% of frames). Root cause confirmed with data
BEFORE proposing fixes: detection tracks apparent ball SIZE -- HARD's
ball is 39px wide (median), TEST1's is 24px (camera further back). Same
root cause as jersey OCR (section 4c: distance, not contrast).

Hypothesis tested: stock detection runs at imgsz=1280, DOWNSCALING the
1920px frame before inference and shrinking the already-small ball --
so a HIGHER imgsz should see the ball more. `ball_spike.py` gained a
configurable imgsz (suffixed output, never clobbers the canonical log --
measure-first discipline, same as the reid probe section 11). Swept
1024 / 1280 / 1920 on TEST1 frames 0-450 (both known shots + play).

RESULT, measured on PHYSICS-GATED ARCS (not raw coverage -- section 13
already established raw count is misleading):
  imgsz | raw coverage | arcs formed | both known shots?
  1024  |     30%      |     3       | NO  (shot B collapses)
  1280  |     34%      |     6       | YES (only resolution with both)
  1920  |     44%      |     5       | NO  (loses BOTH shots)

1280 is the CLEAR OPTIMUM, bracketed on both sides. The result is doubly
instructive:
1. Raw coverage is ANTI-CORRELATED with usefulness here: 1920 has the
   HIGHEST coverage (44%) and the WORST arc result (0 known shots). The
   extra coverage is false positives elsewhere in the frame, not better
   ball tracking. Trusting the 44% number would have shipped a
   REGRESSION -- section 13's lesson, now triple-confirmed.
2. WHY higher res hurts: a fast ball is MOTION-BLURRED; at high
   resolution the smear looks LESS like the compact round "sports ball"
   the model learned, so it's rejected (higher conf when it DOES fire,
   but on far fewer frames -- shot A: 10 tracked frames at 1280 -> 5 at
   1920). Downscaling to 1280 compacts the blur into a ball-like blob.

DEEPER FINDING (why no single setting wins everything): the two shots
have OPPOSITE optimal resolutions. Shot A (blurrier/faster) is seen best
at 1024 (14 frames vs 10 at 1280); shot B (cleaner) needs 1280+ (11
frames vs only 3 at 1024, where it collapses). 1280 is the robust choice
because it's the only setting that clears the arc-forming bar for BOTH.

VERDICT: 1280 STAYS. The resolution lever is EXHAUSTED -- swept, optimum
found, both directions measured worse. Do NOT re-run this sweep (same
status as the reid probe, section 11). Tiling / sliced inference (the
former "option 2") is ABANDONED unmeasured: it is essentially
higher-effective-resolution, the direction just shown to hurt via motion
blur. Custom-trained detector stays ROADMAP-gated and now LESS
attractive -- the ceiling here is footage capture (ball size + motion
blur), not model quality on the pixels we have.

THE REAL LEVERS for ball-seeing, going forward:
- Footage quality (zoom / 4K / closer camera) -- the true root fix,
  helps FUTURE recordings only; already guidance for the user's dad
  (section 4c). This is where the actual gain lives.
- (Speculative, NOT built, would need its own measure-first spike):
  a multi-resolution ensemble (detect at 1024 AND 1280, merge) could in
  principle catch shot A's extra frames AND shot B -- but it doubles
  compute for a gain the current 1280 pipeline mostly already gets
  (both shots form arcs at 1280). Deferred unless a real need is
  measured.
- Layups remain UNRECOVERABLE by any resolution setting: the ball is at
  the rim too briefly and too occluded (bodies + backboard) to form a
  trackable flight. This is a fundamentally different problem from ball
  SIZE and is orthogonal to imgsz. Not pursued.

Net for the product: the shots that form flights already work at 1280
(both TEST1 shots, both HARD shots). Ball-seeing is footage-limited, not
tuning-limited, on the clips in hand.

## 21. MODEL-CAPACITY PROBE — yolov8x, NEGATIVE (2026-07-14)
After §20 (input-resolution exhausted), tested the other detection axis --
MODEL CAPACITY -- prompted by the user (rightly) pushing back that
ball-seeing still needs work and the resolution result didn't close the
goal. yolov8x.pt (extra-large, ~68M params) vs yolov8m.pt (medium, ~26M),
both at the proven-best imgsz 1280, TEST1 0-450.

RESULT (arcs, not raw coverage): WASH. Both = 6 arcs, both capture both
known shots; per-shot flight frames essentially unchanged (shot A 10->8,
shot B 11->11). Raw coverage rose 34%->39% -- junk again (§13). A 2.6x
bigger model, 2-3x slower, moved the shot metric by ZERO.

INTERPRETATION (load-bearing for the product's compute/model strategy):
the bottleneck is NOT model capacity. Medium already extracts what's
extractable from these pixels; extra-large finds no more shots because
the missed balls are motion-blurred / tiny / occluded -- i.e. the
limiting factor is INFORMATION IN THE FOOTAGE, not the model's power to
read it. Corollary for spending: a bigger GENERAL detector is not the
lever, and GPUs buy SPEED (throughput on full games + the ability to
TRAIN), not accuracy-by-themselves. The real model lever, if pursued, is
a FINE-TUNED basketball-ball detector (trained on small/blurry/occluded
basketballs -- the exact hard cases stock COCO "sports ball" misses),
which is a different thing from "use the biggest stock model" and is the
ROADMAP's custom-detector path, now evidence-justified. Highest leverage
overall remains FOOTAGE (pixels on the ball): a closer/zoom/4K camera
makes the ball bigger and less blurred at the source, which no model or
GPU can substitute for. yolov8m stays the default; yolov8x not adopted.

## 4. KNOWN DEBT (logged, not fixed)
- **RESOLVED 2026-07-14 (section 18):** the shot-arc over-count from a
  rim deflection (originally logged here) is now caught by classify_shot's
  ORIGIN GATE -- an arc whose first point already starts within
  HOOP_RADIUS_PX of the hoop is rejected as a continuation/deflection,
  not a fresh release. Validated against all 4 real arcs found in the
  full-clip harvest (2 genuine shots, 2 deflections), each as a literal-
  data regression test. See section 18 for the full story and the
  ACCEPTED remaining trade-off (a genuine close-range layup would also
  fail this gate -- not yet observed on real footage, logged not hidden).
- **Camera pan during a shot distorts the ball-trajectory fit.**
  Found 2026-07-14 (section 18, HARD 356-381): the trajectory layer fits
  parabolas in raw IMAGE-pixel space with no camera-motion compensation
  (DECISIONS 14 flagged this as a hypothetical risk; this is the first
  confirmed real instance). User caught it by eyeball: "the camera moved
  mid arc [so] the arc was messed up." Measured: the carried hoop
  position (a fixed WORLD point) drifts ~300px in image-space over this
  25-frame shot, purely from camera motion -- the ball's fitted curve is
  distorted by the same motion. Practical impact so far: CONTAINED, not
  silent-wrong -- the shot was still correctly classified as an attempt
  (min_dist 19.4px) and its outcome correctly abstained (unknown) rather
  than guessing through the distortion. Root fix (not built, bigger than
  today's timebox): compensate the per-frame ball position by the SAME
  camera-motion transform already computed for the hoop-carrying
  (Hs_opt @ Hfk) before fitting, so the trajectory layer fits in a
  camera-STABILIZED frame. Revisit if a future arc fails outright (gets
  dropped as no_claim) rather than merely distorted, or if the user asks.
- **Part-1 track-labels and Part-2 queue-resolutions never cross-check the
  SAME track/identity against each other.** Found 2026-07-13 (section 12,
  finding 2): a human can label a track in Part 1, then separately resolve
  a queue item for the SAME underlying track in Part 2 with a DIFFERENT
  number, and nothing flags the contradiction (unlike OCR-vs-human, which
  IS flagged -- see section 10). This one instance was found by an eyeball
  check spawned by the color-tiebreak build, not by any automated guard,
  and fixed by hand. Root fix (later): when a Part-2 resolution's track_id
  already has a Part-1 label, refuse or flag loud instead of silently
  layering a second, possibly-conflicting number onto the same track.
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
