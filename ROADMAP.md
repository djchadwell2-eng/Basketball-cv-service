# ROADMAP — where every area actually stands, 2026-08-30

Written after a session that opened a lot of threads at once. Tagging follows
the compute handoff:

- **[MEASURED]** — came out of a real run. Trust it.
- **[ESTIMATE]** — arithmetic on measured numbers. Could be wrong.
- **[UNKNOWN]** — never established. Do not act as if it is.

Order below is by expected value, not by how recently it came up.

---

## PRIORITY 1 — The reader works. Its answers are being thrown away.

### Where this stands right now

`HANDOFF_COMPUTE_2026_08_29` §6 opens with *"23,288 candidates, **0 confident
reads**, 0 named [MEASURED]"* and builds the whole naming pessimism on it.

That number is not evidence about the reader. §2's own timing table, line 56,
labels the stage that produced it:

```
ocr_confirm (reader stubbed) | 184 | 8.40
```

**The reader was not running.** Zero is what a stub returns. The handoff
contradicts itself two sections apart.

What the reader actually does on that same full-game film
[MEASURED, `phase2/out/Full_Game_9eb8bf2a_ocr_confirms.json`, 150-frame window,
reader live]:

| | |
|---|---|
| crops attempted | 103 |
| crops with any read | 17 |
| **crops with a CONFIDENT read** | **9** |
| candidates attempted | 21 |
| **candidates with a confident read** | **5 (24%)** |

And the outcomes for those five:

```
agree 0 | disagree 0 | no_confident_read 16 | no_position_hypothesis 5
```

**All five confident reads were discarded.** Not wrong — unusable, because no
seed existed for them to agree with. `identity.py` returns
`"no_position_hypothesis"` and the read is binned.

This is not new behaviour; it is newly *quantified on real film*. `DECISIONS`
§4a-WIDE measured the same waste on TEST1 (#5 @ 1.00, #24 @ 0.993, both
correctly abstained for want of a hypothesis).

### MEASURED 2026-08-31 — it is worse, and simpler, than described above

Every one of Full_Game's 21 candidates has `position_hypothesis = None`. Not
most — **all 21**. And five of them read at confidence 1.00, **two of those
corroborated by a second crop** (the strongest evidence tier this system has):

```
w0 id2  track 3   : read #3  conf 1.00  single_crop_only
w0 id14 track 15  : read #13 conf 1.00  single_crop_only
w0 id42 track 206 : read #3  conf 1.00  CORROBORATED
w0 id47 track 244 : read #24 conf 1.00  single_crop_only
w0 id50 track 275 : read #33 conf 1.00  CORROBORATED
```

All five binned by one line in `identity.py`:
`if expected is None: return "no_position_hypothesis"`.

**Why every hypothesis is None: Full_Game has no `_decisions.json`.** No human
has clicked it. `roster.seed_number_for` therefore returns None for every track,
so every window-start seed confirms a body with no number attached.

Compare a clip that HAS been clicked:

| clip | candidates | with a hypothesis | agree |
|---|---|---|---|
| Full_Game (no labels) | 21 | **0** | **0** |
| TEST1 | 5 | 3 | 0 |
| HARD | 82 | 25 | **13** |

**The consequence, stated plainly: on a game nobody has clicked, the reader can
never confirm anything — by construction, no matter how well it reads.** Naming
today cannot start from zero. It can only *agree* with a name a human already
supplied.

That is the whole naming problem in one sentence, and it reframes the rest of
this roadmap.

### This also corrects PRIORITY 3 below

Seeding more bodies does **not** unlock naming on its own. A mid-window arrival
seeded with `roster_number=None` is in exactly the position of the 21 above:
visible, reviewable, and permanently unconfirmable by OCR. Re-seeding raises the
BOX-SCORE ceiling (floor time is credited only to confirmed identities) and puts
bodies into the review queue — both real — but the binding constraint on NAMING
is the hypothesis, not the seeding moment.

### What to do about it

Invert the relationship: let a number **establish** an identity instead of only
**confirming** one.

**This needs an explicit decision from DJ, not a silent change.** The proposed
shape, which mirrors what `promote_via_second_signal` already enforces:

> two confident reads, from **different crops at different times**, agreeing
> with each other — the first is the hypothesis, the second is the second
> signal.

Nothing about the CONFIRMED contract loosens: continuity still cannot reach it,
and a single read still cannot.

**Known complication:** dual-roster numbers key two different girls to one
identity — HARD lists #3 and #23 on both rosters, TEST2 lists #1, #4 and #13
[MEASURED, verified against `clip_config`]. `color_tiebreak` must supply the
team split; it is measured 6/6 on HARD (`DECISIONS` §12).

**Cheapest decisive experiment:** re-run stage 6 **unstubbed** over cached
slices and count *how many distinct fragments per player get ≥1 confident read*.
**Kill number:** under ~2 fragments per player per possession, keying on the
number cannot carry a name.

---

## PRIORITY 2 — GMC was only ever tested on the clip with the least camera motion

### Where this stands right now

`DECISIONS` §11 refuted appearance re-ID and, in the same experiment, tested
BoT-SORT's global motion compensation. That verdict has been treated as closed
ever since.

**It ran on TEST1 only, and TEST1 barely pans** [MEASURED, `todo.md`]:

| clip | camera pan |
|---|---|
| TEST1 | 0.8 px/frame |
| HARD | **3.6 px/frame (4.5×)** |

GMC compensates the *camera's own motion* before matching boxes. It was
measured where it had the least to offer.

On TEST1, gmc-only already gives [MEASURED]:

| tracker | ids | mean lifespan | merges |
|---|---|---|---|
| bytetrack (committed) | 122 | 110.7 | — (reference) |
| **botsort gmc-only** | **117** | **122.7** | 1 (on the already-known-spliced t49) |

Machinery is on disk: `phase2/botsort_gmc_only.yaml`, and a previous output at
`spikes/out/TEST1_tracks_botsort_gmconly.json`.

### Why this is priority 2

It attacks fragmentation **at the source**. Every relinking idea downstream is
repair work; fewer breaks means less to repair. And it is one CPU tracker pass —
no GPU, no API, no human.

**A new capability makes this judgeable for free.** There is now a *label-free
ID-switch metric*: match a candidate tracker's boxes to the committed tracker's
by IoU, and if one candidate track absorbs two committed tracks that are alive
at the same time and spatially apart, it merged two people. Provable with no
human labels. The player-tracker plan called ID-switch ground truth the
"HIGHEST VALUE ITEM" and assumed it needed a labelling session — it does not.

Already scored with it [MEASURED, TEST1]: `mt09`'s 24% fragment win costs
**2 merges / 709 merged frames (23.6 s)**. That trade-off now has a number
instead of an eyeball.

**Honest limit:** the metric is one-sided — the committed tracker scores 0 by
construction, so it answers *"is this variant riskier than what we ship?"*, not
*"how good is the baseline?"*

### Next step

Run gmc-only on **HARD** and score with the ID-switch metric.
**Kill number:** if ids do not drop ≥15% with 0 new merges, close it for good.

---

## PRIORITY 3 — Re-seeding: where names are allowed to start

### Where this stands right now

This is structural, and **no relinking idea can fix it.**

`stage4_seed_queue.py` seeds on-court tracks **only at a window start**
[MEASURED, lines 104–118]. A body that walks on mid-window is seeded only if it
already carries a human label (`windows.seed_labeled_newcomers`).

Consequences, all measured:

1. **Naming *every one* of the ~2,043 clicks still covers only ~24% of the
   game.** Everyone who arrives mid-window never enters the naming machinery at
   all.
2. **The first 7.7 minutes of the full game are unnameable.** Window 0 begins on
   the film's ~15 pure black frames, where the tracker sees nobody, so nothing
   is seeded. Confirmed identities start at frame **13,921**. Logged as open in
   the compute handoff §3; still open.
3. **36% of human clicks are already thrown away** — 12 of 33 — because the
   clicked identity spanned several tracks (TEST1 5/10, HARD 7/15, TEST2 0/8).

[ESTIMATE] Fixing *when seeding happens* is a cheaper 3–4× than any relinking
idea on the list, because it raises the ceiling rather than recovering losses
under it.

### What it likely involves

- Seed on-court newcomers **when they arrive**, not only at a boundary. The
  safe-seeding rule already exists in `windows.seed_labeled_newcomers` — it is
  the *gate* (must already be labelled) that needs revisiting, not the mechanism.
- Make window 0 start where there are bodies, or let a window seed late if it
  opened on an empty frame. Generalises beyond black frames: **any** window
  starting on a moment where the tracker sees nobody gets no seeds at all.
- Anything here must keep the `identity.py` contract: seeding is a *hypothesis*,
  not a confirmation.

---

## PRIORITY 4 — Relinking, and what is now closed

### Where this stands right now

Two corrections to the problem statement, both measured this session:

**1. "122 track ids" is mostly crowd.** Only **43 of 122** ever stand on the
court on TEST1; **180 of 469** across three clips. Every fragmentation figure
quoted before this was inflated by spectators.

**2. Jersey number is not a valid identity key.** Of 28 same-number track pairs,
only **17 are real splits**. Eight are **twins** — two different girls wearing
the same number on opposite teams — and 3 are duplicate ids on one body. The
naive count overstates fragmentation by **29%**.
*Any evaluation harness that keys on jersey number will score two opponents as a
successful relink.*

Corrected scale: **~17 real relinks across 3 clips**, ~0.65 per labelled player.
A **floor** — only 50–64% of on-court track-frames carry any label.

**The gaps** [MEASURED, n=16]: median **34 frames**, p90 141.
Only **5 of 16 (31%)** fall inside `identity.MAX_GAP_FRAMES = 30`.
**The current relink window is blind to two-thirds of the problem.**

**Court distance across the gap:** same player median **4.2 ft**, different
player median **16.5 ft**. Real signal, overlapping tails — 10% of *wrong*-player
pairs are under 5 ft.

| rule | nearest candidate correct |
|---|---|
| gap ≤30, 10 ft | **4/5 (80%)**, median 1 competitor |
| gap ≤150, 10 ft | 6/14 (43%), median 4 competitors |

### What is alive

- **R1. Short-gap position relink.** gap ≤30 **and** ≤10 ft **and** exactly one
  candidate. Fixes ~31% of the problem at high precision. **Must produce
  CANDIDATE, never CONFIRMED** — it is continuity, and the contract forbids it.
- **R2. Court-feet as a queue *pruner*, never an answerer.** Candidates per death
  fall from median 6 (TEST1) / 18 (HARD) to **1–2**; true partner survives 6 of
  8. Use it to **rank** what a human is offered. **Never to refuse** — it deletes
  real relinks ~25% of the time (box jitter implies 105 ft/s).
- **R3. Five-on-court lineup exclusion.** See below.

### CLOSED — do not re-propose

| idea | why |
|---|---|
| appearance re-ID | `DECISIONS` §11 — 122 → 131 ids, **worse** |
| court-feet relink **as an answerer** | 37.5% precision (HARD 3/8, TEST1 0/1). `todo.md` listed this as the untried lever; it is now measured dead |
| velocity extrapolation | hurts at long gaps; pushed the correct candidate outside the gate 6× vs 2× |
| naive one-in-one-out counting at a break | fires **0** times of 95 / 235 / 56 |
| gait / height / build | true pairs rank 51st, 58th, 66th of 75 impostors |
| shoes / hair | dead by inheritance — a 40px jersey number already loses to angle |
| colour **at the moment of a break** | flags 0 of 2 real switches — the crop at a break is a blend |
| colour for player-vs-non-player | measured this session, **no safe cut exists** |
| motion for player-vs-non-player | DJ, 2026-08-29 — a coach walked 26 ft |
| better footage | DJ, 2026-08-29 — the film is Hudl, and that is the job |

---

## PRIORITY 5 — Five-on-court exclusion (blocked, with a warning)

### Where this stands right now

**The design is right.** DJ corrected his own earlier version: a roster is not a
closed set, a **lineup** is. Exclusion does not *discover* a name, it **carries**
one through fragmentation — a name would die at the next *substitution* instead
of the next *window*. [ESTIMATE] ~2,043 clicks → ~150–300.

It survives identical uniforms because **it never looks at her.** It looks at who
is already accounted for. That is why it lives where appearance re-ID died.

Independent corroboration: the fragmentation measurement concluded, without
knowing the idea, that what is needed is *"a discriminator that survives a 1–5
second gap, with position as a gate, not the decision."* That is exactly this.

### Two things blocking it

**BLOCKER 1 — player vs non-player.** A referee in the count gets confidently
named. Both automatic routes are now measured dead:

- motion — DJ, 2026-08-29
- colour — this session. Players' distance-to-nearest-team-centroid: median
  22.7. Non-players: median **34.9**, but **min 10.1** — closer than the median
  player. Ranges overlap end to end; **every threshold that catches a ref also
  deletes a real player.**

Routes left:
- **(a) Human ref labels** — already built (`roster.load_ref_tracks`). Refs are
  few and long-lived: TEST1 needed 5, HARD 12, TEST2 4. **~10–20 clicks a game
  against 2,043.** Inelegant, real, ~1% of the budget.
- **(b) Ask the vision model "player, referee, or coach?"** — **UNTRIED**, and it
  plays to the model's *strength*. Reading a number is fine detail on a small
  crop, which it is measurably bad at. Striped shirt vs uniform vs street
  clothes is a coarse semantic call. The same 34-player / 19-non-player ground
  truth is already on disk, so this is one cheap measurement.

**BLOCKER 2 — the precondition may be too rare.** [MEASURED, TEST1, **un-scoped
by team**] "exactly one unnamed on-court body" holds in only **16% of frames**,
and **at the three actual relink moments it was met 0 of 3 times** — two players
were unnamed simultaneously each time.

Per-team scoping should improve this and has not been measured.
**Do that before building.** **Kill number:** if per-team scoping does not lift
it above ~40% of frames, exclusion fires too rarely to change the order of
magnitude.

Also confirmed on disk: HARD's Milford "roster" is **six** numbers for a
five-player team — roster ≠ lineup, exactly as DJ said.

**BLOCKER 3 — substitution boundaries.** A stale lineup produces a confident
wrong name for minutes. Dead balls are the honest boundary and we already read
the scoreboard — but the cached `FULL_GAME_scoreboard_timeline.json` carries
**scores only, no clock**. [UNKNOWN] whether the clock reads as reliably as the
score.

### The safety rule (DJ's, non-negotiable)

Assign only when the arithmetic is **exactly forced**: exactly 5 of that team on
court, exactly 4 CONFIRMED (never candidate), exactly 1 unnamed, no substitution
crossed. Anything uncertain → she stays unknown and goes to the queue.
Provenance is **inherited from the seed**, and it goes through `stage7`'s
existing contradiction check — which already refuses when one number would be in
two places.

---

## PRIORITY 6 — Shots and the basket-window harness (paused, working)

### Where this stands right now

This is the thread we were on before naming took over, and **it works.**

**The problem it solved.** A full-game shot run is [ESTIMATE] ~2,200 CPU-hours
because the rim tracker is CPU SIFT against all five keyframes per frame
(compute handoff §5). So shots could never be measured at game scale.

**The way round it.** The scoreboard is ground truth and reading it needs none of
the expensive machinery:

1. `spikes/scoreboard_timeline.py` — walked all 95 minutes, **47 confirmed
   scoring plays, final 54–68**, 100 of the game's 122 points accounted for
   (82%). 32 minutes of CPU, no GPU. Four independent checks that it is real:
   zero non-monotonic steps across 47 plays; a basketball point distribution
   (12 ones, 22 twos, 12 threes); **8 of the 12 one-pointers fall after minute
   75** — the end-game fouling stretch, which nothing in the reader knows about;
   and only 2 plays show both teams scoring, both across a readable gap.
2. `spikes/refine_basket_times.py` — bisects each basket from a 10 s window to
   under a second. **45 of 47 refined, 31 pinned to ≤2.5 s, median 1.3 s.**
3. `spikes/run_basket_windows.py` — runs the full pipeline on ~5 s around each
   known basket. **3,000 frames instead of 171,120.**

### Results so far [MEASURED, 10 windows completed before it was stopped]

| | |
|---|---|
| **Shot detection recall** | **6/10 = 60%** |
| touches | 22 |
| possessions | 19 |
| **named** | **0** (matches the naming problem exactly) |
| possession endings | 9 other-team, 8 end-of-clip, **2 out-of-bounds** |

**Out of bounds fired on real film for the first time.** DJ's rule from
2026-08-02 had never once fired — all three original clips have zero off-court
touches, so it existed only in unit tests.

### What this measures, and what it cannot

Every window is a **make** by construction — the scoreboard is what found them.
So this measures **recall**, never a shooting percentage. A real percentage needs
misses, and the scoreboard cannot point at those.

### Two bugs found and fixed here, worth remembering

1. **Import-time span binding.** `phase2/run_tracking.py` reads
   `SPAN_START`/`SPAN_LEN` as module-level constants, so a second
   `import run_tracking` in one process silently re-tracks the first window
   forever. Fixed by one window per process — which DJ's own repo note
   (*"one clip per process"*) had said all along.
2. **`TaskStop` does not kill the Python process.** Three runs were alive at once
   for two days, all writing the same artifact files. Any long run must be
   launched with a recorded PID and killed by PID.

### To resume

- 21 usable windows remain unrun (31 usable, 10 done).
- ~30 min per window on CPU → [ESTIMATE] ~10 h for the rest.
- The rim anchors are saved and **verified**: `hoop_anchor` now carries a rim
  marked on **any** frame (SIFT-matched to its nearest keyframe, refuses a weak
  match), which was needed because the right-hand basket appears in **none** of
  this game's five calibration keyframes. Verified by eye — the rim carried from
  f166842 lands on the ring at f151200, f158700 and f171000, 15,000+ frames away.

---

## Immediate next steps (agreed with DJ, 2026-08-30)

1. **GMC on HARD**, scored with the label-free ID-switch metric. One CPU pass.
2. **Re-seeding** — mid-window arrivals and the empty-window-0 bug.

Then return to Priority 1 (the discarded reads), which needs a contract decision
from DJ before any code is written.
