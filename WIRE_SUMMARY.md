# Make/Miss Pipeline Wiring - Summary

## What Was Done

Wired the dense scoreboard make/miss detection directly into the main `measured_stats.py` pipeline so that make/miss data is automatically computed for every clip.

## Files Changed

### measured_stats.py (3 changes)

**Change 1: Updated function signature**
- Added `make_miss_results=None` parameter to `build_measured_stats()` function
- This parameter receives the results from the scoreboard reader

**Change 2: Merged make/miss data into shots**
```python
# For each shot, if make/miss data exists, add these fields:
- make_miss_outcome: "candidate_make" or "unknown"
- make_miss_score_from: [home, away] score before shot
- make_miss_score_to: [home, away] score after shot
- make_miss_score_change_frame: frame where score changed
- make_miss_score_change_time_sec: timestamp of score change
```

**Change 3: Auto-run scoreboard detection in generate()**
```python
# In generate() function:
- Try to load and run detect_makes_by_scoreboard()
- Only runs if both shot_attempts.json and scoreboard_ocr.json exist
- Safely handles missing files (no crash)
- Passes results to build_measured_stats()
```

**Change 4: Updated make_miss_available flag**
```python
"make_miss_available": bool(make_miss_results)  # True only if data exists
```

## How It Works Now

### Before (Manual)
1. User runs ball detection pipeline
2. User manually runs `spikes/dense_shot_score_match.py` to get make/miss
3. User somehow merges it into the output

### After (Automatic)
1. User runs ball detection pipeline
2. `measured_stats.generate(clip_name)` is called (by analyze_clip.py or web app)
3. Automatically calls `detect_makes_by_scoreboard()` 
4. Returns measured_stats.json with make/miss data already included
5. Web app reads it and displays

## What This Enables

- **Web app uploads**: Any new clip uploaded through the web app gets make/miss automatically
- **GPU runs**: The serverless handler calls `measured_stats.generate()`, so GPU jobs get make/miss
- **Local development**: Local runs of analyze_clip.py now include make/miss without extra steps
- **Scoreboard styles**: Currently works for broadcast-overlay style (HARD, TEST1). Future Gemini API will add OHSAA and LED boards

## What Still Needs To Happen

1. **Full end-to-end test**: Verify TEST1 measured_stats.json now has make_miss_outcome fields
2. **TEST2 test**: Verify it works on a different clip (currently scoreboard reading doesn't work on OHSAA style, but that's a separate issue)
3. **Web app verification**: Confirm the frontend can display the new fields
4. **Future**: Implement Google Gemini vision reader for universal scoreboard reading

## Implementation Details

The wiring is conservative:
- Only adds make/miss fields when data actually exists (not None)
- Never force-guesses a miss (DJ's hard rule: "silence is not proof")
- Outcome is only "candidate_make" (score changed) or "unknown" (no change)
- If scoreboard data is missing, just skips - doesn't block the pipeline

## Code Quality

✓ Syntax validated
✓ Unit test passed (merge logic verified)
✓ Integration points confirmed (serverless_handler also uses measured_stats.generate)
✓ Error handling in place (try/except on the detection call)
✓ Minimal diff (only ~40 lines changed, no unnecessary refactoring)
