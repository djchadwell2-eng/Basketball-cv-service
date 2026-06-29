"""Per-clip configuration for the court-calibration engine.

STEP 0 of the generalization test: everything that was hardcoded to HARD.mp4 is
lifted here so a new gym clip is just a new entry + a different ACTIVE. The ENGINE
(SIFT/RANSAC homographies, landmark assembly, global least_squares fit, sign-fix,
overlay) is UNCHANGED -- the stages just read their per-clip values from here.

Each clip entry:
  video_path       absolute path to the source video
  keyframes        frame indices ~100-150 apart spanning the pan (chain order)
  reference_pos    index into keyframes used as the fixed reference (None = middle)
  exclude_regions  scorebug / burned-in graphic rectangles (x1,y1,x2,y2) native px,
                   masked out of SIFT. DIFFERENT per gym -- wrong box contaminates RANSAC.
  court            court model dims in feet (don't assume HS for every gym):
                     length, width, lane_y0, lane_y1, ft_x, circle_r
  stills           frame indices to dump as preview JPGs from the overlay renderer
  landmarks        clicked landmarks per keyframe { frame: [(tag,x,y), ...] } native px.
                   keyframe absent -> interactive click mode;  [] -> skip that frame.

The canonical landmark *palette* (tag -> court-feet coord) lives in COURT_MODEL
(stage4), derived from each clip's court dims, so it is NOT duplicated here.
"""

# --- HIGH-SCHOOL (NFHS) court, the default model -----------------------------
#   court 84 x 50 ft; 12-ft lane -> y 19..31; FT line 19 ft; center circle r=6.
HS_COURT = {
    "length": 84.0, "width": 50.0,
    "lane_y0": 19.0, "lane_y1": 31.0,
    "ft_x": 19.0, "circle_r": 6.0,
}

CLIPS = {
    # =========================================================================
    # HARD.mp4 -- the validated, human-confirmed baseline (~0.95 ft, arcs glued).
    # These are the EXACT values the engine was validated with; do not change.
    # =========================================================================
    "HARD": {
        "video_path": r"C:\Users\djcha\Downloads\HARD.mp4",
        "keyframes": [600, 700, 800, 900, 1000, 1100, 1200],
        "reference_pos": None,
        "exclude_regions": [
            (7.5, 891.0, 327.0, 1063.5),   # scorebug, bottom-left
        ],
        "court": dict(HS_COURT),
        "stills": [650, 900, 1100, 1500, 2000, 2700],
        "landmarks": {
            600: [
                ('LB_side_far', 943.5, 355.5),
                ('L_lane_base_near', 432.0, 501.0),
                ('L_lane_base_far', 655.5, 436.5),
                ('L_FT_near', 894.0, 598.5),
                ('L_FT_far', 1097.8, 504.2),
            ],
            700: [
                ('LB_side_far', 945.0, 358.5),
                ('L_lane_base_near', 423.0, 502.5),
                ('L_lane_base_far', 663.0, 435.0),
                ('L_FT_near', 893.7, 598.8),
                ('L_FT_far', 1097.0, 503.7),
            ],
            800: [
                ('LB_side_far', 732.0, 348.0),
                ('L_lane_base_near', 259.5, 495.0),
                ('L_lane_base_far', 481.5, 427.5),
                ('L_FT_near', 698.4, 563.7),
                ('L_FT_far', 870.4, 476.2),
                ('center_logo', 1653.0, 628.5),
                ('center_far', 1698.0, 439.5),
            ],
            900: [
                ('LB_side_far', 673.5, 351.0),
                ('L_lane_base_near', 211.5, 496.5),
                ('L_lane_base_far', 432.0, 426.0),
                ('L_FT_near', 648.0, 559.5),
                ('L_FT_far', 802.1, 471.4),
                ('center_logo', 1549.5, 609.0),
                ('center_far', 1587.0, 424.5),
                ('circle_top', 1561.5, 553.5),
                ('circle_bottom', 1533.0, 685.5),
                ('circle_left', 1353.0, 589.5),
                ('circle_right', 1768.5, 639.0),
            ],
            1000: [
                ('LB_side_far', 672.0, 354.0),
                ('L_lane_base_near', 216.0, 496.5),
                ('L_lane_base_far', 433.5, 429.0),
                ('L_FT_near', 635.9, 558.8),
                ('L_FT_far', 798.0, 472.5),
                ('center_logo', 1512.0, 598.5),
                ('center_far', 1546.5, 429.0),
                ('circle_top', 1521.0, 549.0),
                ('circle_bottom', 1494.0, 678.0),
                ('circle_left', 1318.5, 586.5),
                ('circle_right', 1722.0, 627.0),
            ],
            1100: [
                ('center_near', 684.0, 1023.0),
                ('center_logo', 697.5, 549.0),
                ('center_far', 699.0, 381.0),
                ('R_FT_near', 1549.5, 562.5),
                ('R_FT_far', 1396.5, 462.0),
                ('R_lane_base_far', 1809.0, 435.0),
                ('RB_side_far', 1567.5, 345.0),
                ('circle_top', 697.5, 499.5),
                ('circle_bottom', 694.5, 621.0),
                ('circle_left', 511.5, 562.5),
                ('circle_right', 873.0, 540.0),
            ],
            1200: [
                ('center_logo', 27.0, 685.5),
                ('center_far', 10.5, 481.5),
                ('R_FT_near', 1024.5, 585.0),
                ('R_lane_base_near', 1447.5, 499.5),
                ('R_lane_base_far', 1246.5, 436.5),
                ('RB_side_far', 1015.5, 363.0),
                ('circle_top', 22.5, 625.5),
                ('circle_bottom', 34.5, 777.0),
                ('circle_right', 267.0, 652.5),
            ],
        },
    },

    # =========================================================================
    # TEST1.mp4 -- generalization clip #1 ("normal but different gym").
    # Placeholder: keyframes / scorebug / stills / landmarks get filled in during
    # the per-clip workflow (surface frames -> confirm scorebug -> click).
    # =========================================================================
    "TEST1": {
        "video_path": r"C:\Users\djcha\Downloads\Test1.mp4",
        "keyframes": [],
        "reference_pos": None,
        "exclude_regions": [],          # TBD: confirm Test1's scorebug box first
        "court": dict(HS_COURT),        # TBD: confirm Test1's court size
        "stills": [],
        "landmarks": {},                # filled by interactive clicking
    },
}

# Which clip the stages operate on. Set to "HARD" for the regression check.
ACTIVE = "HARD"


def active():
    """The config dict for the currently selected clip."""
    return CLIPS[ACTIVE]
