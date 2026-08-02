# False Positive Rejection - Path to 90% Accuracy

## Current State
- TEST1: 3 attempts → 3 marked as "shot_attempt" (player-signal rejects 10 others as "not_shot")
- HARD: 2 attempts → 2 marked as "shot_attempt" (player-signal rejects 51 others as "not_shot")

The player-signal check IS working - it filters out most false positives. The question is: can we improve it further?

## What Are Current False Positives?

The "not_shot" items are likely:
1. **Rebounds** - Ball caught after bounce, player's hand close to ball
2. **Passes** - Ball thrown between players
3. **Dribble holds** - Player holding the ball during dribble
4. **Missed catches** - Ball thrown but not caught cleanly
5. **Loose balls** - Contested ball handling

## Improvement Strategies (Ranked by Feasibility)

### Strategy 1: Enhanced Hand/Rim Distance (EASY)
**Current logic:** Ball closer to hand than rim = HAND (false positive)

**Improvements:**
- Use hand VELOCITY, not just position
  - Shooting: hand ACCELERATES away from ball (release)
  - Pass/catch: hand DECELERATES toward ball (receiving)
  - Rebound: hand moves SIDEWAYS/DOWN (boxing out)
- Use multiple hands: are BOTH hands involved?
  - Shot: often one hand dominant (release hand)
  - Pass/catch: both hands often involved (receiving)
- Check hand HEIGHT relative to head
  - Shot: hand above head (release point)
  - Pass: hand at chest/shoulder level
  - Catch: hand catches ball

**Effort:** Medium (add velocity + multi-hand tracking)
**Expected gain:** 5-10% accuracy improvement

### Strategy 2: Ball Trajectory Analysis (MEDIUM)
**Current logic:** Rely only on end-frame position

**Improvements:**
- Fit parabola to ball path
  - Shot: strong upward arc early, gravity dominates later
  - Pass: nearly straight line
  - Rebound: bouncing pattern (V-shaped, multiple bounces)
- Measure arc height relative to shooter
  - Shot from hand: arc height >> hand height
  - Pass/catch: minimal arc, ball stays near hand
- Measure arc duration
  - Shot: 0.5-1.5 seconds flight time
  - Pass: <0.3 seconds
  - Rebound: variable, chaotic

**Effort:** Medium (need ball path for 1 second, not just end frame)
**Expected gain:** 10-15% accuracy improvement

### Strategy 3: Player Motion Analysis (HARD)
**Current logic:** Just check hand position

**Improvements:**
- Pose keypoints: arm angle trajectory
  - Shooting release: arm EXTENDS away from body
  - Pass release: arm FLICKS laterally
  - Catch: arm BENDS toward body
- Elbow angle change
  - Shot: elbow EXTENDS (130° → 180°)
  - Pass/catch: elbow stays BENT or FLEXES
- Shooting form signature
  - Real shot: consistent form (pickup → load → release)
  - False positive: erratic, uncoordinated motion

**Effort:** Hard (complex pose analysis)
**Expected gain:** 10-20% accuracy improvement

### Strategy 4: Multi-Shot Context (HARD)
**Current logic:** Analyze each potential shot independently

**Improvements:**
- Temporal: do shots come in realistic patterns?
  - 2-3 shots per possession
  - Shot every 10+ frames (minimum dribble/movement between)
  - Can't have 3 shots in 1 second
- Spatial: do shooters cluster in realistic zones?
  - Shots from paint + three = realistic
  - 30 shots scattered randomly across half-court = suspicious
- Player identity: is same player shooting repeatedly?
  - Natural: some players shoot more
  - Suspicious: many different players with 1 shot each

**Effort:** Very hard (needs possession/player tracking)
**Expected gain:** 5-10% accuracy improvement

---

## Recommended Approach (SIMPLE PATH TO 90%)

1. **Phase 1 (This week):** Implement Strategy 1 (hand velocity + height)
   - Easy to add to existing code
   - Should get to ~75-80% accuracy
   - Low risk, high confidence

2. **Phase 2 (If needed):** Add Strategy 2 (ball trajectory)
   - Requires 1 second of ball history (we have it)
   - Medium complexity
   - Should get to ~85-90% accuracy

3. **Phase 3 (Polish):** Strategy 3 (shooting form)
   - Only if we need to go above 90%
   - Most complex but most reliable

## Implementation Checklist

### Phase 1: Hand Velocity + Height
- [ ] Track hand position across frames (last 5 frames)
- [ ] Calculate hand velocity (pixels/frame)
- [ ] Add velocity rule: hand ACCELERATING AWAY = shot
- [ ] Check hand height vs head height
- [ ] Test on TEST1, TEST2, HARD
- [ ] Measure accuracy improvement

### Phase 2: Ball Trajectory  
- [ ] Collect ball positions for 1 second post-shot
- [ ] Fit parabola to trajectory
- [ ] Measure arc height, duration, shape
- [ ] Add trajectory rule: strong arc + gravity = shot
- [ ] Test on all clips

### Phase 3: Shooting Form
- [ ] Analyze arm angle sequence
- [ ] Measure elbow angle trajectory
- [ ] Recognize shooting release form
- [ ] Test on all clips

---

## Success Criteria

- **75% accuracy:** Phase 1 only (velocity + height)
- **85% accuracy:** Phase 1 + Phase 2 (trajectory analysis)
- **90%+ accuracy:** All three phases

## Code Location

Current player-signal check: `spikes/pose_shot_check.py`
- `window_verdict()` - main decision function
- `_ends_at_hand()` - position-based logic
- `_nearest_hand()` - hand distance calculation

Next steps: Add velocity and trajectory analysis to these functions.
