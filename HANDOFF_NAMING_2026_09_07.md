# Handoff — the naming/clicking problem (session ending 2026-09-07)

Tagging as in the compute handoff:
**[MEASURED]** came out of a real run · **[ESTIMATE]** arithmetic on measured
numbers · **[UNKNOWN]** never established, do not act as if it is.

---

## 0. THE PRODUCT GATE — read this before proposing anything

DJ, 2026-09-03, explicit and unprompted:

> *"Im not shipping anything until the individual tracker is working and
> shippable. Meaning a low amount of clicks with a high accuracy rate."*

Shipping the team-level product first **was proposed and rejected.** Possessions,
shot locations, zones, 60% shot recall and make/miss all work today without a
single name — that is not the product.

**Both conditions are joint.** A naming route that is accurate but costs ~2,000
clicks fails the gate exactly as hard as a cheap one that names the wrong girl.
When judging any idea, report **clicks per game AND accuracy**, or the answer is
incomplete.

---

## 1. THE ONE-PARAGRAPH STATE OF PLAY

A name currently dies at every window boundary — **147 of them a game** — which
is why clicking costs ~2,043 [MEASURED, windows × people]. Everything this
session was aimed at making a name **survive**, or at making one **appear
without a click**. Four routes were closed with measurements, one blocker was
solved, and the next experiment is set up and unrun.

---

## 2. THE BIGGEST FINDING: "0 confident reads" WAS A STUBBED RUN

`HANDOFF_COMPUTE_2026_08_29` §6 opens with *"23,288 candidates, 0 confident
reads, 0 named [MEASURED]"* and builds the naming pessimism on it. **§2's own
timing table, line 56, labels that stage `ocr_confirm (reader stubbed)`.** The
reader was not running. Zero is what a stub returns.

**The reader is not the weak link.** [MEASURED, TEST1, reader live] all 8 reads
that had a human label to check against agreed with it at confidence 1.00:

```
t3 #13   t5 #23   t6 #5   t8 #14   t9 #24   t10 #30   t17 #32   t67 #32
```

**8 for 8. When it reads, it is right.** The problem is that it rarely reads
(3.4% per crop) and that its answers were being binned.

### Why a fresh game could never be named
[MEASURED] All 21 of Full_Game's candidates carry `roster_number = None`, because
nobody has clicked that game, so `promote_via_second_signal` returns
`"no_position_hypothesis"` for every read however good — four at confidence 1.00,
two corroborated, all discarded.
**On a game nobody has clicked, the reader could not confirm anything, by
construction.**

**FIXED (built this session):** `identity.establish_via_reads` + a third
provenance `read_established`, so a jersey can name a girl nobody clicked. It
requires TWO agreeing reads from DIFFERENT crops at DIFFERENT times, never
overrides a human's number, never fires on a relinked CANDIDATE, and never on a
dual-roster number. The lock still holds — continuity has no valid provenance
and one read still confirms nothing. 418 tests pass.

**IT HAS NEVER FIRED, and the reason is arithmetic, not footage.** At a 3.4%
per-crop read rate, sampling 3 corroborating crops gives a 10% chance any of them
reads at all → expected 1.0 corroborations from 10 reads, observed 0.
**30 crops would give 65%.** [MEASURED]
So Option C is **starved, not wrong**. It becomes viable the moment reads are
cheap — which is the other thread's GPU work, not a research problem.

---

## 3. WHAT WAS CLOSED THIS SESSION (do not re-propose)

| idea | evidence |
|---|---|
| **GMC / camera-motion compensation** | Judged on TEST1 (0.8 px/frame pan) because `reid_fragment_probe.py` had TEST1 **hardcoded** and overwrote any caller's clip. Run properly on HARD (3.6 px/frame): **252 ids vs 260 (−3.1%) and SIXTEEN merges across 76 s of film.** Kill number was ≥15% fewer ids with 0 merges. Closed. |
| **Court-feet relink with a speed limit** | `todo.md` listed this as *the* untried lever. Measured: **37.5% precision** (HARD 3/8, TEST1 0/1), indistinguishable from the 70%-wrong pixel version. Closed as an *answerer*; alive as a **queue pruner** (candidates per death 6–18 → 1–2). |
| **Velocity extrapolation** | Hurts at long gaps — pushed the correct candidate outside the gate 6× vs 2×. |
| **Naive one-in-one-out counting** | Fires **0 times of 95 / 235 / 56**. Fragmentation is bursty, never tidy. |
| **Gait / height / build** | True pairs rank 51st, 58th, 66th of 75 impostors. Anti-informative. |
| **Colour for player-vs-non-player** | Players 4.7–63.7, non-players 10.1–63.4 — ranges overlap end to end. **No threshold catches a referee without deleting a real player.** |
| Appearance re-ID | `DECISIONS` §11, 122→131 ids. Was already closed; re-confirmed. |

---

## 4. THE BLOCKER THAT GOT SOLVED: player vs referee

Exclusion needs an accurate count of players per team. The count read **1 to 9
bodies**, with six and seven common. Chased to the root:

- **NOT the geometry margin** — median body sits **0.00 ft** outside the painted
  lines, and known bench tracks sit **0.16–0.51 ft** out, *inside* the slack a
  real player needs for stepping on a line.
- **NOT duplicate detections** — **99.8%** of excess bodies do not overlap an
  already-counted player.
- **It IS that referees, coaches and bench bodies are genuinely inside the court
  rectangle**, and nothing told them from players.

**Asking the vision model works** (`spikes/ask_is_player.py`), on whole-body
crops — a striped shirt, a tracksuit and a kit are told apart by the silhouette,
not by a torso rectangle.

**RULE B, and the asymmetry is the whole point:** only a **unanimous REFEREE or
COACH** may delete a body. `OTHER` means *cannot tell*; a split vote is not
evidence. Neither may ever delete.
*Keeping a referee* leaves the count one too high → the gate refuses → costs a
click. *Deleting a real player* turns a five into a four → **forces a wrong
name.**

[MEASURED] 35 of 37 players kept, and **the 2 "errors" were rendered and looked
at: the model was right and the human labels were wrong.** TEST1 t10 (labelled
"#30") is in a black-and-white striped shirt. TEST1 t15 (labelled "#10") is an
adult in a jacket, sitting, holding a clipboard.
**So: zero real players deleted, and it caught two non-players the labels
missed.**

### ⚠️ A WARNING THAT REACHES BEYOND THIS EXPERIMENT
`{clip}_decisions.json` is the ground truth used to score the jersey reader,
team assignment, the exclusion precondition and every fragmentation number —
**and it has a referee recorded as player #30.** Those labels are not
unimpeachable. Anything scored against them carries that uncertainty.

---

## 5. EXCLUSION: measured, improved, still does not fire

Applying Rule B to every on-court body [MEASURED]:

| clip | exactly five | impossible (>5) |
|---|---|---|
| HARD before | 14.1% | **74.9%** |
| HARD after | **31.3%** | **32.8%** |
| TEST2 before | 25.6% | 64.0% |
| TEST2 after | **49.6%** | **15.0%** |

Real, large improvement. **And exclusion still fires 0 of 9** at the real relink
moments on HARD — counts there read 7, 6, 6, 4, 6, 6, 5, 6, 6.

### Two things this established
1. **EXCLUSION HAS A FOOTAGE PRECONDITION nobody had stated.** TEST1's mode is
   **three** bodies per team; exactly-five happens 2.3% of the time, *unchanged
   by filtering*, because the camera pans and never holds five of a team in
   shot. **The camera must hold all five of a team in frame at the moment the
   name would be lost.** HARD and TEST2 (wider) pass far more often. DJ predicted
   this.
2. **What still inflates the count is BENCH PLAYERS.** The model says PLAYER and
   it is right — she simply is not on court. Geometry cannot separate her
   (0.16–0.51 ft outside). The remaining question is not *"is this a player"* but
   **"is this player playing or sitting"** — another coarse semantic question,
   and a seated body is a different **pose**, not a different appearance.
   **UNTRIED, and it targets the residual 33% directly.**

---

## 6. THE NEXT EXPERIMENT — set up, unrun

**Pose-guided crop selection.** The strongest untried idea, because it attacks
the measured root cause of everything upstream.

**The reasoning:** the read rate is **3.4% per crop**, and the recorded cause is
**ANGLE** — backs and side-ons. Crops are currently chosen by **box size**, which
is a proxy for *close to camera*, not *facing the camera*. A pose model gives
shoulder keypoints, and **shoulders say which way she is facing.**

**If that lifts 3.4% to even 15%, everything downstream moves at once:**
corroboration stops being a lottery, Option C fires, and the reader can name
people with no click.

**The cheap test, on cached data, no new footage:**
- POSITIVE set: the `(read_frame, read_bbox)` of every confident read — they are
  already recorded in `{clip}_ocr_confirms.json` outcomes
- NEGATIVE set: other picked frames from the same tracks
- Compute a facing score from shoulder keypoints, compare the distributions
- **Kill number:** if facing does not separate read-from-unread, the idea is dead
  and crop selection stays size-based

**Weights are ALREADY ON DISK** — `yolo11x-pose.pt` and `yolov8n-pose.pt` in the
repo root, and `ultralytics 8.4.75` is installed. No download, no new dependency.

**Second, smaller bet:** sports-specific re-ID (SoccerNet). Generic re-ID was
trained on pedestrians in *different clothes* and fails on identical uniforms by
construction; sports re-ID datasets are built for exactly that. Not the same
experiment §11 ran. Prior still low.

---

## 7. DJ'S FRAMING, AND WHY THE EVIDENCE SUPPORTS IT

> *"I feel like there wont be 1 ultimate fix for the clicking problem but rather
> many small fixes that work in different scenarios."*

**The measurements agree.** Every mechanism that works covers a different case
and fails where another succeeds:

| fix | works | fails |
|---|---|---|
| short-gap relink | 31% of gaps (≤30 frames) | long gaps |
| the reader | when her number faces the camera (8/8 right) | backs, side-ons |
| player/ref filter | any framing | a seated player |
| exclusion | wide shots holding five | pans like TEST1 |

They are **independent**, so coverage compounds rather than adds — and this
codebase is already built for it, because every layer **abstains** instead of
guessing, so specialists can be stacked without compounding error.
What to reject is not "many small fixes" but **"one small fix that sometimes
lies."**

---

## 8. NEW TOOLS THIS SESSION

| file | what |
|---|---|
| `spikes/tracker_switch_metric.py` | Scores a tracker's **merges with NO human labels** — if one candidate id absorbs two committed tracks alive at the same time in different places, it glued two people. The player-tracker plan called this the "HIGHEST VALUE ITEM" and assumed it needed a labelling session. One-sided by design: the committed tracker scores 0. |
| `spikes/ask_is_player.py` | Player / referee / coach, whole-body crops, votes cached to `spikes/out/is_player_votes.json` so rescoring never costs an API call. |
| `spikes/exclusion_precondition.py` | Per-team five-on-court arithmetic, plus `--relinks` for the metric that matters. Also gives **team-by-colour for every on-court body: 27/30**. |
| `spikes/exclusion_with_filter.py` | The two combined; per-clip verdict caches. |
| `spikes/player_vs_nonplayer.py` | The colour measurement that closed that route. |
| `spikes/scoreboard_timeline.py`, `refine_basket_times.py`, `run_basket_windows.py` | Earlier in the session: 47 baskets from the scoreboard for 32 CPU-min, timings refined to ~1 s, pipeline aimed only at them. |

---

## 9. SHOTS — paused and working, if you want a win

Not touched since the naming work took over. **60% shot-detection recall over 10
windows** [MEASURED], out-of-bounds fired on real film for the first time, 22
touches, 19 possessions. **21 usable windows remain unrun**, ~30 min each.
Every window is a make by construction, so it measures **recall, never a
shooting percentage.**

---

## 10. OPERATIONAL RULES THAT COST TIME TO LEARN

1. **`TaskStop` does not kill the Python process.** Three runs were alive at once
   for two days, all writing the same artifact files, which made a fixed bug look
   unfixed. Launch long runs with a recorded PID and kill by PID.
2. **One clip per process.** `phase2/run_tracking.py` binds `SPAN_START`/
   `SPAN_LEN` at *import*, so a second `import` in one process silently re-tracks
   the first window forever.
3. **Cache the expensive half.** Model votes are the cost; scoring rules are what
   you iterate on. Every tool here caches votes to disk.
4. **Look at the pictures.** Five times this session, rendering crops corrected a
   conclusion that measurements agreed on — including discovering the ground
   truth itself was wrong.
5. **Write the kill number down first**, and if the metric turns out to be the
   wrong one, say so rather than quietly re-scoring (the exclusion 40%-of-frames
   number was the wrong metric; per-relink-moment was the right one).
