# WHERE THIS PROJECT STANDS

**Last updated: 2026-07-30** · Plain English on purpose. Technical detail lives in
[tasks/todo.md](tasks/todo.md) and evidence in [TEST_LOG.md](TEST_LOG.md).

> ## ✅ SOLVED: calibrate a full game FROM SCRATCH
> DJ, 2026-07-30, after watching a real video of it: *"utter perfection
> everything is right!"*
>
> **A 95-minute game, calibrated from 4 marked spots and 56 clicks total** (was
> 570+ spots / 2,850+ clicks before this session). This is the first time this
> project has proven a from-scratch full-game calibration end to end, checked
> three separate ways instead of trusting one number:
> 1. The 4 spots were checked to actually CONNECT to each other, at full video
>    quality, scoreboard covered, *before* anyone was asked to click a single
>    point. (This exact check got skipped the first time this was tried, and
>    that attempt failed at 15.45 ft off. Second time, it was run first.)
> 2. The real calibration math: **0.21 ft average error** — clears the "glued"
>    bar (0.3 ft), not just "not obviously broken" (0.94 ft, DJ's own eye test).
> 3. DJ watched a real video — actual gameplay, court lines drawn on top, at
>    each of the 4 spots — and confirmed it by eye.
>
> **What's still unproven:** this is one game, one gym, one camera position.
> Whether 4 spots is still enough on a *different* gym, or a shakier camera, is
> untested. The video check covered 10 seconds near each spot, not the long
> stretches of real gameplay in between. Full detail, including the exact
> recipe to repeat this: [tasks/todo.md](tasks/todo.md) (search "REVIEW --
> SESSION SUMMARY, 2026-07-30").

**What the system does today:** you give it a short clip you've set up, and it
tells you who was on the floor, where they stood, where shots came from, and
now who had the ball. It runs start to finish. 285 automated tests pass.

**What it cannot do yet:** work on a video you just uploaded, cover a whole
game, or tell you whether a shot went in.

---

## ✅ WORKS, AND YOU'VE SEEN IT WORK

| Thing | State |
|---|---|
| **Who's on the floor** | Named box score, per-player floor time and zones |
| **Court measuring** | Solved from your clicks, not assumed. Test1 0.15 ft, Test2 0.29 ft |
| **Shot locations** | Shot chart from real court positions |
| **Who has the ball** ("touches") | You confirmed on video: "on the right girl the whole time" |
| **Seen vs guessed, kept apart** | Every touch reports both. An assumption can't pose as a measurement |
| **Referees excluded** | Your ref labels now block refs from holding the ball or taking shots |
| **Flicker protection** | A change of hands must last ≥6 frames to count |
| **Runs end to end** | One command does the whole pipeline |
| **Touches reach the app** | The app's data file now carries them |

### Fixed along the way
- Court length wrong for Test2 (94 ft, not 84) — was the real cause of "nothing lines up"
- Renderer silently deleting entire court overlays (your "missing lines")
- Identity merging different girls into one person
- Your queue clicks pointing at the wrong bodies
- A referee being credited with taking two of your verified shots
- Jersey numbers being confused with internal ID numbers

---

## 🔨 BUILT BUT NOT TURNED ON

These work in testing. Nobody has eyeballed them, so they're not live.

| Thing | Why it's waiting |
|---|---|
| **Smart click planning** | Cuts your clicking ~50%. You haven't looked at a court built this way |
| **Shooter from touches** | Better than the current guess, but no ground truth for who shot what |
| **Better ball model (v3)** | Finds more shots but claims some non-shots |
| **Tracker setting (mt=0.9)** | 35% better at keeping players, but can swap two players |
| **Dead-ball / timeout detector** | Works (151/155 seconds correct on one clip). Not wired in |
| **Pose "was that really a shot"** | Passed a real holdout 9/9. Not wired in |

---

## ❌ THE FOUR REAL BLOCKERS

### 1. Clicking — ✅ RESOLVED 2026-07-30 (for one game; see caveats above)
You click ~10 spots per 3 seconds of footage to teach it the court.

| | marked frames | your clicks |
|---|---|---|
| A 15-second clip | 6 | ~58 |
| A quarter of a game | ~144 | **~1,440** |
| With the new smart spacing | ~48 | **~480 (≈40 min)** |

**Automatic detection: tested 2026-07-29 — doesn't work off the shelf** (2–35 ft
error vs your 0.15). It *does* find the court, just imprecisely, having never
seen a high school gym. Retraining on your footage is the open option.

**Hybrid (detector + a few clicks): tested, dead.** Worse than clicking alone on
every gym, at every weighting.

**But those tests found the real win — and it needs no AI at all:**

| | error | clicks per frame |
|---|---|---|
| What you do now | 0.16 ft | ~10 |
| **5 spread-out spots** | **0.29 ft** | **5** |
| 4 spots | 0.48 ft | 4 |

**You've been clicking about twice what the maths needs.**

### And then a real full game was measured (2026-07-29)

`Full_Game.mp4` — 171,120 frames, **95.1 minutes**. Marks are only needed once
per distinct *camera view*, and a game reuses very few:

| | marks | clicks | time |
|---|---|---|---|
| Old convention | ~1,711 | ~17,000 | ~12 h |
| Best before this | ~570 | ~2,850 | ~3.2 h |
| Attempted | 5 | 63 | ~4 min |

**DJ clicked those 5 frames. The court came out RIGHT — but the calibration
FAILED.** Corrected 2026-07-30:

```
court identified   0.23 ft   84 ft floor, runner-up 3.4x worse   ✓ clean
full calibration  15.45 ft   (0.94 ft is "broken by eye")        ✗ failed
```

**Why:** two of the five frames are 28 minutes apart and barely match each other
— 9 matching points where healthy pairs have 116–352. The calibration chains
frames together; one broken link wrecks the whole thing.

**My error:** I measured whether each frame could *reach* a mark. Calibration
needs the marks to reach *each other*. The frame-picking algorithm was actually
selecting for maximum difference — the opposite of what a chain needs. And the
project already had a guardrail that would have caught it in seconds; I didn't
run it before asking for the clicks.

**Still true:** DJ's clicks are good, the floor is 84 ft, and 7–8 *chainable*
frames stays inside his stated budget. The fix is to pick frames with
`spikes/plan_keyframes.py` (built and validated in TEST 29, then not used here)
and print the chain health before asking anyone to click. See TEST 36.

**Caveat that matters:** this measures whether a frame can *reach* a mark, not
whether the court then lands *accurately*. Those are known to diverge. Turning
~75 clicks from an estimate into a result needs one 5-minute clicking session
from you. Details and the two errors I made getting here:
[TEST_LOG.md](TEST_LOG.md) TESTs 29–34.

**Then it was actually done (2026-07-30).** The chain plan's 4 frames (600 /
127200 / 151200 / 171000) were checked for real — full video quality,
scoreboard covered — and held. You clicked 56 points across those 4 frames.
Real calibration math: **0.21 ft average error.** You watched a real video of
it and called it *"utter perfection."* This is proven now, not estimated —
details in [tasks/todo.md](tasks/todo.md), "REVIEW — SESSION SUMMARY,
2026-07-30".

**Also tested and dead:** the tripod/pure-rotation maths that promised 40 clicks
— premise failed measurement (TEST 32).

### 2. Real tendencies need longer clips
Your actual goal is *"she takes 60% of her shots from the left wing."*
Test1 has **5 touches**. There's no tendency in 5. The measuring works — there
just isn't enough footage yet, and footage is gated by blocker #1.

### 3. Make or miss
The system says "candidate miss," never "miss." The scoreboard reader works
when the graphic style matches what it learned, and 3 clips have had 3
different styles. Rule in force: **the scoreboard may confirm, never deny.**

### 4. Speed
Ball detection is **hours on CPU**. The court matching is another slow pass.
You're renting a serverless GPU — not hooked up yet, not scoped.

---

## 🚫 KNOWN GAPS

- **Fresh uploads don't work.** A clip must be set up by hand first
- **Test2 has no ball data** — hours of computing, so no touches there
- **HARD's court is set to 84 ft but measures 94.** Probably why its accuracy
  (0.7–0.8 ft) is 3–5× worse than Test1's (0.15–0.29 ft)
- **No "who shot it" ground truth.** You've confirmed *which* shots are real,
  never *who* took them
- **Player labelling stalled** at ~60 of 200 frames
- **Everything is measured on 2–3 short clips.** Generalization mostly unknown

---

## 📋 WHAT ONLY YOU CAN DO

- [ ] Look at a court built from the thinned clicks — is it still glued?
- [ ] Say who took the shots where the two methods disagree
- [ ] Tell me what GPU you're renting
- [ ] Decide: fix HARD's 94 ft court, or leave it?
- [ ] Watch HARD's touch video (only Test1 was checked)

---

## 🎯 THE HONEST BOTTOM LINE

**Ready now:** a demo on a clip we've set up. "Watch it analyze this game."
Also now ready: calibrating a brand new full game from scratch with ~4 spots
and ~56 clicks instead of 570+ spots and 2,850+ clicks — proven once, on one
gym (2026-07-30). Exact clicking TIME is not yet measured for real; the click
*count* is what's proven.

**Not ready:** "upload any video and it just happens, zero manual steps."
Someone still has to pick the spots and click them — just ~4 spots and ~56
clicks instead of 570+ spots and 2,850+ clicks. Fully automatic court
detection (no clicking at all) was tried and isn't accurate enough yet.

**Not close:** turning a calibrated full game into finished stats *fast*.
Ball detection alone is still hours on a normal computer — that's blocker #4,
separate from calibration, and still unsolved.

The quality work is largely done and it's honest — it abstains instead of
guessing. Calibration-at-scale (clicking) is now solved for one proven case.
What's left is generalizing it beyond one gym, and compute speed (#4).
