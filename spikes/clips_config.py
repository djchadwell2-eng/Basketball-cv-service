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
    # TEST1.mp4 -- generalization clip #1: a DIFFERENT GYM from HARD.
    #   HARD  = Winton Woods "Warriors" (blue/green painted court).
    #   TEST1 = Milford "Eagles", Coach Ted Dixon Court (red/black hardwood),
    #           game vs Little Miami. Same broadcast style, HS court.
    # Clean left->center->right pan is frames ~100..585 (then it pans back).
    # Lower-left graphic varies: normal scorebug + a larger player card ~500-585,
    # so the exclude box is widened to cover the card too.
    # Landmarks filled by interactive clicking (stage2 click mode).
    # =========================================================================
    "TEST1": {
        "video_path": r"C:\Users\djcha\Downloads\Test1.mp4",
        "keyframes": [120, 220, 320, 420, 500, 580],
        "reference_pos": None,
        "exclude_regions": [
            (0.0, 810.0, 415.0, 1080.0),   # scorebug + player-card, bottom-left
        ],
        "court": dict(HS_COURT),           # HS 84x50, same as HARD
        "stills": [150, 320, 480, 580],
        "landmarks": {
            120: [
                ('LB_side_far', 840.0, 241.5),
                ('L_lane_base_near', 331.5, 522.0),
                ('L_lane_base_far', 571.5, 391.5),
                ('L_FT_near', 964.5, 640.5),
                ('L_FT_far', 1128.0, 480.0),
                ('circle_left', 1726.5, 663.0),
            ],
            220: [
                ('LB_side_far', 837.0, 241.5),
                ('L_lane_base_near', 333.0, 525.0),
                ('L_lane_base_far', 567.0, 393.0),
                ('L_FT_near', 958.5, 639.0),
                ('L_FT_far', 1128.0, 478.5),
                ('circle_left', 1722.0, 667.5),
            ],
            320: [
                ('LB_side_far', 627.0, 234.0),
                ('L_lane_base_near', 151.5, 519.0),
                ('L_lane_base_far', 375.0, 385.5),
                ('L_FT_near', 762.0, 606.0),
                ('L_FT_far', 910.5, 451.5),
                ('center_logo', 1702.5, 630.0),
                ('center_far', 1723.5, 337.5),
                ('circle_top', 1710.0, 546.0),
                ('circle_bottom', 1692.0, 745.5),
                ('circle_left', 1464.0, 603.0),
            ],
            420: [
                ('LB_side_far', 382.5, 292.5),
                ('L_lane_base_far', 174.0, 430.5),
                ('L_FT_near', 507.0, 586.5),
                ('L_FT_far', 622.5, 454.5),
                ('center_near', 1228.5, 1033.5),
                ('center_logo', 1237.5, 547.5),
                ('center_far', 1237.5, 331.5),
                ('R_FT_far', 1908.0, 525.0),
                ('circle_top', 1237.5, 489.0),
                ('circle_bottom', 1236.0, 637.5),
                ('circle_left', 1054.5, 547.5),
                ('circle_right', 1422.0, 561.0),
            ],
            500: [
                ('L_FT_near', 138.0, 643.5),
                ('L_FT_far', 247.5, 495.0),
                ('center_near', 918.0, 1023.0),
                ('center_logo', 873.0, 537.0),
                ('center_far', 855.0, 327.0),
                ('R_FT_near', 1582.5, 592.5),
                ('R_FT_far', 1461.0, 460.5),
                ('R_lane_base_far', 1918.5, 450.0),
                ('RB_side_far', 1695.0, 309.0),
                ('circle_top', 867.0, 477.0),
                ('circle_bottom', 882.0, 618.0),
                ('circle_left', 702.0, 549.0),
                ('circle_right', 1042.5, 534.0),
            ],
            580: [
                ('center_logo', 357.0, 625.5),
                ('center_far', 312.0, 396.0),
                ('R_FT_near', 1099.5, 592.5),
                ('R_FT_far', 966.0, 469.5),
                ('R_lane_base_near', 1563.0, 516.0),
                ('R_lane_base_far', 1381.5, 409.5),
                ('RB_side_far', 1170.0, 301.5),
                ('circle_top', 342.0, 561.0),
                ('circle_bottom', 376.5, 714.0),
                ('circle_left', 153.0, 658.5),
                ('circle_right', 546.0, 603.0),
            ],
        },
    },
}

# Which clip the stages operate on. Set to "HARD" for the regression check.
ACTIVE = "TEST1"


def active():
    """The config dict for the currently selected clip."""
    return CLIPS[ACTIVE]
