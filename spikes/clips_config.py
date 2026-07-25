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

    # =========================================================================
    # TEST2.mp4 -- FIRST fully NEW gym (different court/camera/teams): Fairfield
    # Indians girls game. 1920x1080 30fps 48s. Panning follow-cam. Landmarks
    # clicked by DJ in the browser court-marking tool (2026-07-22) -- the first
    # clip calibrated through that interface. Keyframes span left->center->right
    # (camera holds the left basket ~0-8s, then pans right). Includes the new
    # 3pt-arc-top tag (L_arc_top/R_arc_top).
    # =========================================================================
    "TEST2": {
        "video_path": r"C:\Users\djcha\New folder\Throw away repos\Basketball Analyer CV System Test\clips\Test2.mp4",
        # v2 keyframes (2026-07-23): dropped redundant 40/140, added 3 bridge
        # frames (275/300/325) across the fast left->center pan to strengthen
        # the SIFT chain (the 240->340 direct pair was weak, arcs ~2ft off).
        "keyframes": [240, 275, 300, 325, 340, 400],
        "reference_pos": None,
        "exclude_regions": [
            (0.0, 810.0, 580.0, 1080.0),   # scorebug, bottom-left
        ],
        # MEASURED from DJ's own marks, not assumed (court_detect.identify).
        # This gym is a FULL-SIZE 94-ft floor with ordinary high-school
        # markings -- and getting that wrong is what broke the calibration for
        # two sessions. It had been given dict(HS_COURT) (84 ft) copied from
        # TEST1, so the engine squeezed a 94-ft court into 84 ft and dragged
        # every mark ~10 ft out of place along the length: mean error 0.94 ft,
        # arcs visibly off the paint, and the marks blamed instead. Detected at
        # 0.20 ft vs 0.62 ft for the 84-ft court (3.1x -- a clear call); with
        # the right court the fit is 0.29 ft, identical to TEST1's glued 0.29.
        "court": "auto",
        "stills": [240, 300, 340, 400],
        # Rims (NOT court landmarks -> not here; carried by hoop_anchor):
        #   left  rim = frame 240 px (509, 129);  right rim = frame 400 px (1473, 205)
        # The 3pt-arc-apex marks (L_arc_top / R_arc_top) are good data -- the
        # arc is the standard 19.75 and they land 0.3-0.4 ft on the right court.
        # They were once removed as "imprecise"; that was wrong.
        "landmarks": {
            240: [
                ('LB_side_far', 750, 206), ('L_lane_base_near', 265, 508),
                ('L_lane_base_far', 477, 363), ('L_FT_near', 889, 616),
                ('L_FT_far', 1050, 461), ('circle_left', 1807, 660),
                ('L_arc_top', 1187, 570),
            ],
            275: [
                ('LB_side_far', 752, 216), ('L_lane_base_near', 283, 510),
                ('L_lane_base_far', 496, 371), ('L_FT_near', 890, 616),
                ('L_FT_far', 1050, 462), ('circle_left', 1785, 658),
            ],
            300: [
                ('LB_side_far', 484, 203), ('L_lane_base_near', 31, 503),
                ('L_lane_base_far', 240, 370), ('L_FT_near', 637, 582),
                ('L_FT_far', 781, 434), ('circle_left', 1477, 587),
                ('circle_top', 1709, 535), ('circle_bottom', 1684, 715),
                ('center_logo', 1699, 606), ('center_far', 1741, 318),
            ],
            325: [
                ('LB_side_far', 122, 220), ('L_FT_near', 295, 580),
                ('L_FT_far', 428, 422), ('circle_left', 1094, 535),
                ('circle_right', 1510, 562), ('circle_top', 1307, 473),
                ('circle_bottom', 1293, 641), ('center_logo', 1300, 548),
                ('center_far', 1323, 272), ('center_near', 1254, 1056),
            ],
            340: [
                ('L_FT_near', 86, 599), ('L_FT_far', 214, 441),
                ('circle_left', 877, 532), ('circle_bottom', 1069, 623),
                ('circle_top', 1077, 447), ('center_logo', 1072, 532),
                ('circle_right', 1273, 543), ('center_near', 1049, 1027),
                ('center_far', 1086, 265),
                ('L_arc_top', 348, 520), ('R_arc_top', 1816, 545),
            ],
            400: [
                ('circle_left', 7, 658), ('circle_bottom', 224, 714),
                ('circle_top', 208, 552), ('center_logo', 218, 624),
                ('center_far', 187, 363), ('circle_right', 417, 599),
                ('R_FT_far', 1030, 459), ('R_FT_near', 1155, 589),
                ('R_lane_base_far', 1502, 403), ('R_lane_base_near', 1676, 520),
                ('RB_side_far', 1286, 259), ('R_arc_top', 917, 538),
            ],
        },
    },
}

# Which clip the stages operate on. Set to "HARD" for the regression check.
ACTIVE = "TEST1"

_RESOLVED = {}          # clip name -> court dims worked out from its marks


def _resolve_court(name, clip):
    """Work out which real court this clip was filmed on, from its own marks.

    Raises if it cannot tell. That is deliberate: a clip whose court can't be
    identified must stop the run, not quietly proceed on a guessed court --
    guessing is exactly what put TEST2 on an 84-ft model in a 94-ft gym and
    made every mark look 10 ft out of place.
    """
    import court_detect
    result = court_detect.identify(clip["landmarks"])
    print(f"[{name}] " + f"\n[{name}] ".join(court_detect.report(result)))
    if not result["identified"]:
        raise ValueError(
            f"{name}: cannot tell which court this is -- {result['reason']}")
    return result["dims"]


def active():
    """The config dict for the currently selected clip.

    A clip whose "court" is "auto" gets its dimensions measured from its own
    clicked landmarks (once, then cached) instead of assuming a court.
    """
    clip = CLIPS[ACTIVE]
    if clip.get("court") != "auto":
        return clip
    if ACTIVE not in _RESOLVED:
        _RESOLVED[ACTIVE] = _resolve_court(ACTIVE, clip)
    return dict(clip, court=_RESOLVED[ACTIVE])
