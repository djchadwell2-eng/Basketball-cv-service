# Individual Tracker — the standing context doc

**Purpose:** this file is the shared context for the **naming / clicking /
per-player identity** workstream — handed to outside reviewers (ChatGPT etc.)
and read by anyone picking the thread up. It is about **knowing WHICH GIRL is
which, at what click cost, at what accuracy.** Dollars-per-game and
minutes-per-game are a separate workstream in a separate chat
(`TIME_AND_COST.md`); do not propose compute/cost work here.

Last updated 2026-09-09.

**Tagging convention, used throughout and taken seriously:**
**[MEASURED]** came out of a real run · **[ESTIMATE]** arithmetic on top of
measured numbers · **[UNKNOWN]** never established, do not act as if it is.
An estimate is never silently promoted to a fact. This convention exists
because a confidently-stated wrong number sent this workstream down a dead
end for weeks — see §8.

---

## 1. The target, in the owner's own words

> *"Im not shipping anything until the individual tracker is working and
> shippable. Meaning a low amount of clicks with a high accuracy rate."*
> — DJ, 2026-09-03, unprompted

**Both conditions are joint.** A route that names girls accurately but costs
~2,000 clicks fails exactly as hard as a cheap route that names the wrong
girl. **When judging any idea here, report BOTH numbers — clicks per game AND
accuracy — or the answer is incomplete.**

| target | status |
|---|---|
| Low clicks per game | **NOT MET — ~2,043 today** [ESTIMATE, §2] |
| High naming accuracy | **partially — the reader is accurate when it fires, but fires rarely** [MEASURED, §3] |
| Names produced with **zero** clicks | **NOT MET — 0, on every clip ever run** [MEASURED] |

**Shipping the team-level product first was PROPOSED AND REJECTED.**
Possessions, shot locations, zones, ~60% shot recall and make/miss all work
today without a single player name. That is not the product; per-player
advice is. Do not re-pitch "ship team-level now, add names later."

**Standing rule that governs this workstream:** a wrong name is worse than no
name. Every layer **abstains** rather than guesses, and "not sure" routes to
a human. Any proposal that raises coverage by lowering the confirmation bar
is rejected by default. (But see §9 — abstention does *not* buy as much as
we claimed.)

---

## 2. Where the clicks go [ESTIMATE, from measured counts]

A name is currently **destroyed at every window boundary**. Identity resets
per window *by design* (`phase2/windows.py`): a wrong name inside one window
must not be able to corrupt the next. That containment is a real safety
property — and it is also the direct cause of the click bill.

```
147 windows per game  ×  ~13.9 people on court per window  ≈  2,043 clicks
```

**The trap in the obvious fix:** seeding fires ONLY at a window start, so
using fewer/longer windows means fewer clicks **and less coverage** at the
same time. It is not a free dial.

**And even paying in full does not finish the job:** naming all ~2,029
covers only ~232 player-minutes, roughly **24% of the game** [ESTIMATE].
Clicking harder is not a path to the target.

---

## 3. The jersey reader — accurate, and starved

**When it reads, it is right.** [MEASURED, TEST1, reader live] All 8 reads
that had a human label to check against agreed with it, at confidence 1.00:

```
t3 #13   t5 #23   t6 #5   t8 #14   t9 #24   t10 #30   t17 #32   t67 #32
```

**How often it reads** [MEASURED, per clip, from the saved run artifacts]:

| clip | crops tried | confident reads | per-crop | candidates | ≥1 read | **per-candidate** |
|---|---|---|---|---|---|---|
| TEST1 | 291 | 10 | 3.4% | 45 | 10 | **22%** |
| HARD | 534 | 22 | 4.1% | 79 | 22 | **28%** |
| TEST1_REG | 346 | 23 | 6.6% | 44 | 8 | **18%** |
| TEST2 | 166 | 13 | 7.8% | 28 | 12 | **43%** |

**Correction to how this has been quoted:** "the read rate is 3.4%" has been
repeated in handoffs as if it were *the* number. It is **TEST1's** number and
the worst of the set; the real per-crop range is **3.4–7.8%**. More
importantly the per-crop figure is the wrong headline — the pipeline
accumulates across a whole window, so **per-candidate (18–43%)** is the rate
that decides whether a girl gets named. Quote that one.

**The design that exists but has never fired:** `identity.establish_via_reads`
lets a jersey name a girl **nobody clicked** — requiring two agreeing reads
from *different* crops at *different* times, never overriding a human, never
on a relinked candidate, never on a number both teams share. It is built,
tested (418 tests green at the time), and has **fired 0 times, on every clip**
[MEASURED]. Cause is arithmetic, not footage: corroboration draws ~3 alternate
crops, and at these read rates the chance any of them reads is small. **The
second-opinion step is the starved one** — see §7 lead #1.

---

## 4. The experiment that FAILED this session — read before proposing crop work

**Hypothesis:** crops are picked by BOX SIZE, which is a proxy for *close to
camera*, not *facing it*. Pick face-on crops instead and the reader gets more
readable pictures.

**Stage A — retrospective, on cached data, $0.** For every confident read on
disk, compare its crop against the *other* crops picked for that same
candidate, scoring each with a pose model's apparent shoulder width.

| | n | mean facing | separation |
|---|---|---|---|
| WIN (crops that read) | 23 | 0.156 | |
| LOSS (other picked crops) | 189 | 0.124 | **0.684** |

Checked three ways before being believed: held in the **same direction on all
4 clips** separately; **not a size proxy** (correlation of facing with box
height = 0.068); and eyeballed on rendered crops.

**Then a flaw was found in Stage A itself.** The LOSS pile mixed *"tried and
failed"* with *"never attempted"* — stage6 exits early once a candidate reads,
so later crops are never submitted. Re-scored using **only crops the reader
genuinely attempted**:

| | n | mean facing | separation |
|---|---|---|---|
| WIN | 23 | 0.156 | |
| LOSS (tried and failed) | 34 | 0.102 | **0.785** |

The flaw had been *hiding* the signal, not creating it. Stronger number,
smaller honest sample.

**Stage B — the live A/B. Same code, same day, only the flag differs.**
TEST2, ~25 min, ~900–1,000 real reader calls. [MEASURED]

| metric | facing OFF | facing ON | |
|---|---|---|---|
| crops giving ANY read | 27 | 22 | worse |
| crops giving a CONFIDENT read | 12 | 11 | worse |
| **candidates named** | **11** | **11** | **same** |
| auto-confirmed | 3 | 3 | same |
| corroborated | 2 | 2 | same |
| review queue after | 32 | 32 | same |

**125 crops were swapped and not one product outcome changed.**

**VERDICT: no measurable benefit. The flag ships OFF (its default).** The
per-crop dips are inside the noise of a non-deterministic reader
(unanimous-of-3), so this is "no effect", not "harmful".

**The lesson, which is worth more than the result:**
**A retrospective correlation does not license an intervention on the same
variable, when intervening also changes something else.** Stage A only ever
compared crops that had *already* been chosen for being big — it never varied
size. The intervention **trades size away to buy facing**, and size is itself
a legibility signal. Those cancelled. Only the live A/B could see that.

**Do not re-propose facing-based crop picking on the strength of 0.785
alone.** Re-open it only in a form that does not sacrifice size — facing as a
**tie-break among equally-large crops**, or a wider shortlist.

*(The code and its 8 tests stay in the tree: they cost nothing while off, and
the pose plumbing is reusable for lead #1 and #2 below.)*

---

## 5. The blocker that got SOLVED: player vs referee

Five-on-court exclusion needs an accurate count of players per team. The
count read **1 to 9 bodies**, six and seven common — impossible in basketball.
Root-caused:

- **NOT the geometry margin** — median body sits **0.00 ft** outside the
  painted lines; known bench tracks sit 0.16–0.51 ft out, *inside* the slack
  a real player needs for stepping on a line.
- **NOT duplicate detections** — **99.8%** of excess bodies do not overlap an
  already-counted player. They are separate people.
- **It IS that referees, coaches and bench bodies are genuinely inside the
  court rectangle**, and nothing told them apart from players.

**Asking a vision model works** (`spikes/ask_is_player.py`) — on **whole-body**
crops, because a striped shirt, a tracksuit and a kit are separated by the
silhouette, not by a torso rectangle. Colour was tried first and is dead:
players scored 4.7–63.7, non-players 10.1–63.4, ranges overlapping end to end.

**RULE B, and the asymmetry is the whole point:** only a **unanimous REFEREE
or COACH** vote may delete a body. `OTHER` means *cannot tell*; a split vote
is not evidence. Keeping a referee leaves the count one too high → the gate
refuses → costs a click. Deleting a real player turns a five into a four →
**forces a wrong name.**

[MEASURED] **35 of 37 real players kept — and the 2 apparent errors were
rendered and looked at: the model was right and the human labels were
wrong.** One "player #30" is a referee in a striped shirt; one "player #10"
is an adult in a jacket, seated, holding a clipboard.

---

## 6. Exclusion: measurably better, still fires 0 of 9

Applying Rule B to every on-court body [MEASURED]:

| clip | exactly five | impossible (>5) |
|---|---|---|
| HARD before | 14.1% | **74.9%** |
| HARD after | **31.3%** | **32.8%** |
| TEST2 before | 25.6% | 64.0% |
| TEST2 after | **49.6%** | **15.0%** |

Large, real improvement. **And exclusion still fires 0 of 9** at the real
relink moments on HARD — counts there read 7, 6, 6, 4, 6, 6, 5, 6, 6.

Two things this established:

1. **Exclusion has a FOOTAGE PRECONDITION nobody had stated.** TEST1's mode is
   **three** bodies per team; exactly-five happens 2.3% of the time,
   *unchanged by filtering*, because the camera pans and never holds five of
   one team in frame. **The camera must hold all five of a team at the moment
   the name would be lost.** DJ predicted this before it was measured.
2. **What still inflates the count is BENCH PLAYERS.** The model says PLAYER
   and it is right — she simply is not *playing*. Geometry cannot separate her
   (0.16–0.51 ft outside). The open question is not *"is this a player"* but
   **"is this player on the floor or on the bench"** — and a seated body is a
   different **pose**, not a different appearance. **UNTRIED, targets the
   residual 33% directly.**

---

## 7. Open leads, ranked (nothing here is built)

**#1 — Fix which crops get the SECOND OPINION.** [VERIFIED flaw in code]
`phase2/stage6_ocr_confirm.py:497` sorts corroboration candidates by
`-abs(g - f)` — **time distance only**. It ignores size *and* facing, both of
which the main picker uses. This is the step gating `establish_via_reads`
(§3), i.e. the only path to naming a girl with **zero clicks** — the single
most valuable thing in this workstream. Cheapest of these, machinery already
in the tree.

**#2 — Frame the torso by pose landmarks, not a fixed percentage.**
`ocr_reader.jersey_crop` takes a fixed 15–85% × 15–50% box. Rendered against
a real player leaning forward, **the number is sliced in half by the crop edge
in 3 of 4 frames** — visible in `spikes/out/`-rendered montages. This is a
crop-geometry defect independent of any angle question. Also unhandled:
whether the number in the crop belongs to the tracked girl or to an
overlapping opponent.

**#3 — Roster/lineup constraints that do NOT require five visible.**
Instead of counting to five, keep a candidate set per fragment and *eliminate*:
two bodies visible simultaneously cannot be the same girl; a confident team
assignment separates opposing players sharing a number; a verified jersey
conflict blocks a link; confirmed substitutions constrain the lineup, while
merely leaving frame does not. **This removes the footage precondition that
killed §6** and makes partial views useful. Bigger build, highest ceiling.
Note `stage7` already has a "one girl cannot be in two places" check, used
**defensively** — nobody has used it to *conclude*.

**#4 — "Playing or sitting?"** Same trick that solved §5, aimed at the
residual 33% of bench bodies. Pose question, not appearance.

**#5 — Sports-specific re-ID (e.g. SoccerNet).** Generic re-ID was trained on
pedestrians in *different clothes* and fails on identical uniforms by
construction; sports re-ID datasets are built for exactly that case. Prior
still low, but it is not the same experiment that already failed.

---

## 8. Closed — do not re-propose

| idea | why it's closed |
|---|---|
| **Appearance re-ID (generic)** | BoT-SORT+reID made it **worse**: 122 → 131 ids. Teammates wear identical uniforms, so embeddings separate player-from-crowd, not teammate-from-teammate — and fragments happen exactly when teammates collide |
| **GMC / camera-motion compensation** | **16 merges** across 76 s on the pan-heavy clip (HARD, 3.6 px/frame), for only −3.1% ids. Earlier "looks fine" verdict came from a probe with TEST1 **hardcoded**, ignoring its caller's clip |
| **Court-feet relink as an answerer** | **37.5% precision** (HARD 3/8, TEST1 0/1) — no better than the ~70%-wrong pixel version. Survives only as a queue *pruner* (6–18 candidates per death → 1–2) |
| **Velocity extrapolation** | Pushed the correct candidate *outside* the gate 6× vs 2× at long gaps |
| **Naive one-in-one-out counting** | Fires **0 times of 95 / 235 / 56**. Fragmentation is bursty, never tidy |
| **Gait / height / build** | True pairs rank 51st, 58th, 66th of 75 impostors — anti-informative |
| **Colour for player-vs-non-player** | Ranges overlap end to end; no threshold catches a referee without deleting a real player |
| **Facing-based crop picking** | Tested live, **no effect** (§4). Only re-open in a form that does not trade away size |
| **Higher resolution / tiling** | Resolution sweep exhausted, 1280 optimal |
| **Better camera / footage** | Hudl is a fixed product constraint |

---

## 9. Where our own reasoning was wrong

**Abstention does NOT make the layers independent.** A previous handoff argued
that stacking specialists is safe because every layer abstains, so "coverage
compounds rather than adds." An outside review pushed back and **it is
right**: several stages can inherit the *same* upstream mistake. If the
tracker glues two girls into one track, the reader, the team assignment and
exclusion all reason about a body that is already wrong — abstention prevents
each layer from being *confidently* wrong on its own terms, but does nothing
to decorrelate them. This codebase already proves the point: `purity` /
`_spliced()` exists precisely because one track can hold two players.
**Treat per-layer coverage numbers as correlated, not independent.**

---

## 10. Traps that have cost real time here

- **A STUBBED READER LOOKS EXACTLY LIKE AN IMPOSSIBLE PROBLEM.** The claim
  "23,288 candidates, 0 confident reads" drove months of naming pessimism; the
  same document's own timing table labelled that stage *"reader stubbed."*
  **There is a second one still on disk right now:** `SCALETEST_ocr_confirms.json`
  reports **117,887 crops attempted, 0 confident reads — and 0 reads of *any*
  kind**, which is not a footage outcome at 117k attempts (live clips give
  4–19% *some* read). Treat any run with `crops_any_read == 0` as broken until
  proven otherwise, and **never average read rates across saved files** — that
  one file drags a real 3–8% down to 0.1%.
- **EARLY EXIT MAKES UNREAD CROPS LOOK LIKE FAILED CROPS.** stage6 stops
  attempting once a candidate reads. A crop that was never submitted is *not*
  evidence the reader failed on it. This produced a wrong finding in this very
  session ("a legible number went unread 4 times" — it was never *tried* 4
  times) and it silently biased a measurement. **Always separate "tried and
  failed" from "never tried."**
- **THE GROUND TRUTH IS NOT CLEAN.** `{clip}_decisions.json` is what the
  jersey reader, team assignment, the exclusion precondition and every
  fragmentation number are scored against — **and it has a referee recorded as
  player #30, and a seated adult with a clipboard as #10** (§5). Anything
  scored against those labels inherits that error.
- **LOOK AT THE PICTURES.** Five separate times, rendering crops corrected a
  conclusion that the measurements agreed on — including discovering the
  ground truth itself was wrong, and finding the crop box cuts numbers in half.
- **ONE CLIP PER PROCESS.** `phase2/roster.py` binds the active clip at
  *import* and caches decisions/refs/spliced tracks in globals that never
  invalidate; `run_tracking.py` binds its span at import too. Two clips in one
  interpreter silently score clip B against clip A's roster.
- **`TaskStop` does not kill the Python process.** Three runs were once alive
  at once writing the same artifacts, which made a fixed bug look unfixed.
  Launch long runs with a recorded PID.
- **CACHE THE EXPENSIVE HALF.** Model votes cost money and minutes; the
  scoring *rule* is what actually gets iterated. Every spike here caches votes
  to disk so re-scoring is free.
- **WRITE THE KILL NUMBER DOWN FIRST**, and if the metric turns out to be the
  wrong one, say so out loud rather than quietly re-scoring. (The exclusion
  "40% of frames" number was the wrong metric; per-relink-moment was right.)

---

## 11. What would actually help from an outside reviewer

1. **Lead #3 in §7 is the one we are least sure how to build.** Given that
   five-visible is a dead precondition on real pan-heavy footage, what is the
   right structure for *eliminating* identities from a candidate set using
   constraints that hold on partial views? What breaks first at 147 windows?
2. **Is there a way to get names at ZERO clicks that we have not considered?**
   `establish_via_reads` is the only such path today and it is starved by the
   read rate. Everything else on the board still assumes a human seeds.
3. **Challenge §2's premise.** Windows exist to contain wrong names. Is there
   a containment design that does not destroy a *verified* name at every
   boundary — keeping the safety property while dropping the click bill?
4. **Sanity-check §9.** If per-layer coverage numbers are correlated rather
   than independent, how should we be combining them honestly?

**What is not useful here:** dollars-per-game or minutes-per-game ideas —
different workstream, different chat (`TIME_AND_COST.md`). Anything already
closed in §8. And any proposal that reaches its accuracy by lowering the
confirmation bar (§1).
