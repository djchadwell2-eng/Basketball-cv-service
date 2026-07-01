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
- **~15s fixed window** (containment boundary stand-in) → replace with **real
  possession detection**. Neither is final possession logic.
