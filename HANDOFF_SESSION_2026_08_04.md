# Session handoff — 2026-08-03/04 (CV quality session)

For cross-checking against the other session. Every claim here is measured, and
where something was tried and **rejected** it says so, so nobody rebuilds it.

My commits, oldest first:

| commit | what |
|---|---|
| `6b9983f` | real team possessions from the ball, not the court half |
| `823935d` | load `.env.local` so the Gemma reader actually runs |
| `a2aa309` | stop the scoreboard reader inventing makes |
| `a893d77` | bound shooter attribution by possession, not a stopwatch |
| `4c9ea71` | read jersey numbers with the vision model, not EasyOCR |
| `e535045` | a read must survive a second crop before it counts |

395 tests pass (was 297 at session start).

---

## 1. TEAM POSSESSIONS — new layer, fully wired

**New files:** `phase2/touch_teams.py`, `phase2/team_possessions.py`
**Wired into:** `ball_stages.stage_team_possessions()` → called from `run_clip.py`
right after `stage_ball_touches`. Output `spikes/out/{clip}_team_possessions.json`.
**Reaches the app via:** `measured_stats.py` → `possessions`, `possession_summary`,
`meta.possessions_available`, `meta.possession_note`, and `possession_index` on
every shot.

**The rules (DJ's, from basketball):** a possession is one TEAM holding the ball.
It ends when the other team touches it, or when the ball goes out of bounds —
which **restarts it even for the same team** (a film-room cut, deliberately NOT
the stat-sheet rule). A team rebounding its own miss stays one possession. A
touch whose jersey colour can't be read is SKIPPED, never treated as a turnover.

**Teams come from jersey colour, costing zero new human input** — `clip_config.py`
and the web-app registry have both been storing `jersey_color` all along.

**Results (team labels verified by eye against the video):**
TEST1 2 possessions, 4/4 tracks right · HARD 4, 8/9 right · TEST2 2, 5/5 right.

### Deleted, because it was lying
`src/team_stats.estimate_possessions()` and `pace_per_minute()`, plus the
"approx possessions / pace" lines in `process_game.py` and `render_heatmaps.py`.
They guessed possession from which half of the floor bodies stood on. Both files'
own TODOs already admitted they were placeholders.

### Renamed, because it was doing a real job under a wrong name
`phase2/possessions.py` → **`phase2/window_boundaries.py`**, and its output
`{clip}_possessions.json` → `{clip}_id_windows.json`. It cuts the clip into
chunks so the identity layer can reset; five stages import it
(`stage4_seed_queue`, `stage5_player_events`, `stage6_ocr_confirm`, `purity`,
`make_review_bundle`). Deleting it would have broken identity. **If the other
session still imports `possessions`, that's the break.**

---

## 2. THE GEMINI KEY WAS NEVER REACHING THE CODE

**New file:** `env_local.py`. Loads both `.env.local` files (this repo first,
then `../Basketball Analysis App/`) into the environment without overwriting a
real env var.

Every consumer read `os.environ.get("GEMINI_API_KEY")` and **nothing ever put it
there**, so command-line runs silently used the weaker OCR fallback. It never
errored — a missing key and a working one behaved identically, which is why it
hid for a whole session.

**Note for the other session:** the two `.env.local` files hold **different
keys**. This repo has `AQ.Ab8RN6...` (53 chars, the working Gemma one); the app
folder has `AIzaSyBxYd...` (39 chars, older). The repo one wins in `env_local`.
Your `d629665` supplies the key to the worker as an endpoint env var, which is
the right shape — just make sure it's the `AQ.` one.

---

## 3. SCOREBOARD MAKE/MISS — four bugs, all shipping wrong answers

In `spikes/gemma_make_miss_fast.py`:

1. **Any score change counted as a make** — including a score going DOWN. Real
   output: `MAKE [1,0]->[0,0]`. Now `is_scoring_play()` requires exactly one team
   UP by 1, 2 or 3.
2. **The prompt never said which side was home.** Measured on TEST1 f300 (truth
   2-0), 3 runs each: terse prompt gave `0-2, 2-0, 2-0` — swapped 1 in 3. Naming
   LEFT/RIGHT gives 2-0 every time. A swap looks exactly like a score change.
3. **Errors were swallowed** as a bare `E` with the exception discarded.
4. **The clip was assumed to start 0-0.** HARD really starts **15-12**, TEST2
   **2-2** — so the first reading looked like a 15-point basket. The baseline is
   now read from before the first shot.

Also: the crop is now the clip's **marked scorebug rectangle** (`exclude_regions`)
upscaled 3x, instead of a hardcoded frame fraction. The old crop was **cutting
TEST2's board in half** — the away score was outside it.

**Result:** TEST1 one real make (0→2, was reporting 0→1); HARD and TEST2 now
correctly report *unknown* — their scores never move. **Fewer makes than before,
and the removed ones were not real.**

**Still true:** a misread that happens to look like a legal jump for that zone
still passes. These guards remove the impossible, not the merely wrong.

---

## 4. SHOOTER ATTRIBUTION — verified for the first time ever

`shooter_attribution_verified` had been `false` since the feature was built.
All six shots on TEST1/HARD/TEST2 were checked against the video (filmstrips,
ball circled, credited player boxed).

**5 of 6 right, 1 wrong, 0 wrongly abstaining.** When a touch exists at the
right moment the body is right **3 out of 3**.

**Rule changed** (`measured_stats.attribute_shooter`): the hard 2-second ceiling
is gone. The shooter is the last player seen holding the ball, bounded by the
**possession** the shot happened in. A shot in no detected possession keeps the
2s fallback — matching `None` to `None` is not evidence two events are related.

No second "flicker" threshold was added: `ball_touch.build_touches` already
requires a handover to be sustained, and taking the latest touch means anyone
who really holds it in between is credited on their own merit.

**Still wrong:** TEST2 f110 credits the wrong girl. The real shooter held the
ball for 3 frames — under `MIN_TOUCH_FRAMES` — so no touch existed.

---

## 5. JERSEY NUMBERS NOW READ BY THE VISION MODEL

`phase2/ocr_reader.py` — the swappable engine seam its own docstring always
described. Gemma when a key is present, EasyOCR otherwise, and it **prints which
one it is using**. `JERSEY_ENGINE=easyocr` forces the old reader.

**Measured on identical crops, same selection, same closed-set filter:**

| reader | correct | wrong |
|---|---|---|
| EasyOCR @0.85 | 1 | 0 |
| Gemma unanimous-of-3 | **12** | 1 |

**Confidence = agreement fraction, so no new dial.** Three reads; confidence is
agreed/asked. Unanimous = 1.00, 2-of-3 = 0.67, so the existing
`OCR_CONFIRM_THRESHOLD` of 0.85 enforces unanimity by itself. It earned that bar
twice: a real "23" reading `[3, 30, 3]` was refused, and a **REFEREE** reading
`[13, 10, 10]` was refused — majority-of-3 would have named him "10".

**Then a second crop must agree** (`e535045`). The two real mistakes came back
UNANIMOUS at 1.00 and were still wrong (a jersey plainly reading 44 read as 14
three times). Asking the same clipped picture again cannot fix it. Every read now
carries `corroboration`: `corroborated` / `single_crop_only` / `conflict_A_vs_B`
(the last is NOT confirmed and goes to the human queue).

**Honest limit:** corroboration succeeded on only 1 read out of 45. A girl
usually has exactly one moment where her number is legible, so the meter mostly
says "only one clear photo" rather than "checked twice".

### `stage6_ocr_confirm.py` restructured for a network reader
A plain nested loop over 43 candidates × 10 crops × 3 network reads is over an
hour — the first run timed out having written nothing. Now **parallel rounds
(6 workers; 16 rate-limited) with early exit**. The attempt budget and threshold
are untouched — only how many of the same crops get spent.

**Also fixed a thread-safety bug:** the engine choice was unlocked, so with
workers running some threads fell back to EasyOCR while others built the Gemma
client — one run silently using two readers.

**Full-run results (auto-confirmed / review queue):**
TEST1 5→**8**, 46→38 · HARD ~0→**13**, 82→69 · TEST2 **4**, 35→31.
Gemma also **fixed a false swap flag** EasyOCR had raised (misread #32 as "3").

**Note:** the model is non-deterministic — TEST1 gave 9 names on one run and 8 on
the next. Expect ±1–2.

---

## 6. THINGS BUILT, MEASURED, AND REJECTED — do not rebuild

1. **Carrying a track's name across identity windows.** Recovers **zero**
   touches (TEST1 2→2, HARD 6→6, TEST2 4→4). The unnamed players were never read
   anywhere.
2. **Crediting brief sub-touch ball contacts to the shooter.** Fixed TEST2 f110
   and broke three others: **3 right/3 wrong vs 5 right/1 wrong**. A ball passing
   *near* a body reads as "held" for 2–3 frames.
3. **The scorebug guard** (skip boxes inside `exclude_regions`). **Would have
   deleted real players** — those rectangles are SIFT masks; TEST2's covers the
   BENCH, and it skipped tracks 16 and 22 whose "24" and "13" are the most
   legible numbers in that clip. Motion doesn't separate them either (graphic
   0.67 px/frame, a player standing on the bench 1.95). Reasoning is recorded in
   `ocr_reader.py`. **Use the existing `roster.load_ref_tracks` ref/bench label
   instead — one click.**
4. **A layout-aware scoreboard prompt** (naming LEFT/RIGHT, ignoring
   period/fouls/clock). Scored **identically** on all three test frames.

---

## 7. PERFORMANCE — the note that was wrong twice

Measured cold on the laptop, on the real full-game film:

| stage | per frame (CPU) |
|---|---|
| ByteTrack + YOLOv8m@1280 | 3.1 s |
| SIFT camera anchor | 3.6 s |
| ball detection | 2.5 s |
| **total** | **~9.2 s → ~440 h (18 days) per game** |

Previous belief was "the anchor alone, 3.35 s/frame, and the GPU doesn't help".
**Both wrong** — tracking costs nearly as much, and `f86ddc3` shows the GPU makes
the anchor 80× faster. Overnight on CPU buys ~1.7 minutes of film, so **CPU is
not a route to full-game validation**.

**Anchor subsampling: settled, DO NOT SUBSAMPLE.** DJ's three runs are now
recorded in `tasks/todo.md`. Headline: where the camera is settled skipping costs
hundredths of a foot; in live play up to ~1 ft — but **even N=2 costs 0.755 ft
there, so that error is the anchor's own jitter, not the skipping**. Anchoring
every frame is 4.1 GPU-hours ≈ 25 min across 10 machines, so the lever isn't
needed.

**Separate finding worth keeping:** a player's court position carries ~0.1 ft
mean / 0.75 ft worst-case anchor noise during roaming play even with no
subsampling. Zones are feet wide so zone calls are safe; anything finer than a
foot must reckon with this floor.

---

## 8. NEW TOOL

`spikes/full_game_possession_run.py` — runs tracks → on-court → identity → ball →
touches → possessions on a slice of a real game, and prints the three unknowns
(possession count, whether out-of-bounds fired, cost per frame). Smoke-tested end
to end on `Full_Game_9eb8bf2a`; it correctly **abstained** on a 60-frame slice
(one touch, one player — you cannot split one player into two teams).

It uses **two configs on purpose**: `ClipConfig.validate()` refuses a ball span
without human-clicked rim pixels (correctly — shots are placed against those
rims), so the shot layer is skipped rather than given fabricated rims.

---

## WHAT IS NOT DONE / STILL OPEN

- **Nothing has run on a full game.** Every result above is from 15–40 second
  clips. Possession COUNT is unverified.
- **The out-of-bounds rule has never fired on real film** — all three test clips
  have zero off-court touches. It is unit-tested only.
- **TEST2 f110 still credits the wrong shooter.**
- **The scoreboard track is still readable as a player** — needs the one-click
  ref/bench label.
- **`Full_Game_9eb8bf2a` has no rim clicks**, so no shots/make/miss on it yet.
- **Cost of the jersey reader at game scale is unmeasured** — extrapolates to
  roughly $0.50/clip, never checked against real billing.
