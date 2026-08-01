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
        # v4 keyframes (2026-07-26): 40 and 140 RESTORED. They were dropped as
        # "redundant" while chasing the arc problem, which turned out to be the
        # court length, not the marks -- and dropping them shrank the calibrated
        # window to 240..400, just 5.3s of a 48s clip. On the corrected 94-ft
        # court they are DJ's two most accurate frames on this clip (0.10 and
        # 0.19 ft), and restoring them more than doubles the usable span to
        # 40..400 (12s, comparable to TEST1's 15s) with no new clicking.
        # 275/300/325 are bridge frames across the fast left->centre pan.
        "keyframes": [40, 140, 240, 275, 300, 325, 340, 400],
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
            40: [
                ('LB_side_far', 749, 210), ('L_lane_base_near', 267, 508),
                ('L_lane_base_far', 484, 373), ('L_FT_near', 892, 618),
                ('L_FT_far', 1052, 461), ('L_arc_top', 1185, 569),
                ('circle_left', 1802, 663),
            ],
            140: [
                ('LB_side_far', 769, 235), ('L_lane_base_near', 326, 513),
                ('L_lane_base_far', 524, 388), ('L_FT_near', 899, 609),
                ('L_FT_far', 1044, 469), ('L_arc_top', 1160, 569),
                ('circle_left', 1731, 648),
            ],
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
    # FULL GAME -- the first REAL full-length clip. 171,120 frames / 95.1 min.
    # Keyframes are NOT on a fixed interval: they are the views
    # spikes/full_game_views.py measured as covering 99% of the game (the first
    # alone covers 50%). The 10 other candidates it found were the pre-game
    # introductions with the house lights OFF -- unlit floor, no basketball --
    # deliberately excluded rather than clicked. See TEST_LOG TESTs 33-35.
    # Court is "auto": the floor is SOLVED from these marks by court_detect, never
    # assumed. A first pass with 25 marks (free-throw lines + centre only) gave a
    # good 0.32 ft fit but could NOT tell an 84 ft floor from a 94 ft one, because
    # nothing was marked at a BASELINE to anchor the length. court_detect refused
    # rather than guess. This is the second pass, with baselines.
    "FULL_GAME": {
        "video_path": r"c:\Users\djcha\New folder\basketball-cv-service\Full_Game.mp4",
        "keyframes": [200, 16000, 65800, 79200, 169000],
        "reference_pos": None,
        "exclude_regions": [(0.0, 830.0, 330.0, 1080.0)],   # bottom-left scorebug
        "court": "auto",
        "stills": [200, 16000, 65800],
        "landmarks": {
            200: [
                ('L_FT_near', 444.6, 469.3),
                ('L_FT_far', 335.9, 606.4),
                ('center_near', 1036.2, 325.6),
                ('center_logo', 1046.2, 529.5),
                ('center_far', 1059.6, 1015.8),
                ('R_FT_near', 1656.3, 484.3),
                ('R_FT_far', 1781.6, 626.4),
                ('LB_side_near', 185.5, 320.6),
                ('RB_side_near', 1918.7, 335.6),
                ('circle_left', 879.1, 531.1),
                ('circle_right', 1213.4, 534.5),
                ('circle_bottom', 1041.2, 476),
                ('circle_top', 1047.9, 616.4),
                ('L_arc_top', 559.9, 531.1),
                ('R_arc_top', 1540.9, 549.5),
            ],
            16000: [
                ('L_FT_near', 1069.6, 471),
                ('L_FT_far', 910.9, 628.1),
                ('center_logo', 1893.6, 680.4),
                ('L_lane_base_near', 531.5, 395.8),
                ('L_lane_base_far', 305.8, 522.8),
                ('LB_side_near', 788.9, 248.7),
                ('circle_left', 1639.6, 637),
                ('circle_bottom', 1900.3, 593.5),
                ('circle_top', 1880.2, 799.1),
                ('L_arc_top', 1201.7, 578.5),
            ],
            65800: [
                ('R_FT_near', 783.8, 490),
                ('R_FT_far', 961, 637.1),
                ('R_lane_base_near', 1307, 401.3),
                ('R_lane_base_far', 1540.9, 518.3),
                ('RB_side_near', 1037.9, 264.3),
                ('circle_right', 224, 672.1),
                ('circle_top', 0, 832.5),
                ('R_arc_top', 651.8, 588.5),
            ],
            79200: [
                ('L_FT_near', 528.1, 457.6),
                ('L_FT_far', 419.5, 596.3),
                ('center_near', 1118.1, 328.9),
                ('center_logo', 1124.8, 532.8),
                ('center_far', 1126.5, 1037.5),
                ('R_FT_near', 1748.2, 501.1),
                ('R_FT_far', 1876.9, 649.8),
                ('L_lane_base_near', 73.5, 451.5),
                ('LB_side_near', 280.8, 312.7),
                ('circle_left', 954.3, 536.7),
                ('circle_right', 1293.6, 543.4),
                ('circle_bottom', 1121.4, 481.5),
                ('circle_top', 1124.8, 620.3),
                ('L_arc_top', 640.1, 530),
                ('R_arc_top', 1622.8, 556.8),
            ],
            169000: [
                ('L_FT_near', 628.4, 459.3),
                ('L_FT_far', 518.1, 586.3),
                ('center_near', 1223.4, 335.6),
                ('center_logo', 1220.1, 544.5),
                ('center_far', 1216.7, 1037.5),
                ('R_FT_near', 1870.2, 522.8),
                ('L_lane_base_near', 188.9, 436.4),
                ('L_lane_base_far', 8.4, 555.1),
                ('LB_side_near', 391.1, 304.4),
                ('circle_left', 1047.9, 545.1),
                ('circle_right', 1395.5, 560.1),
                ('circle_bottom', 1220.1, 489.9),
                ('circle_top', 1218.4, 632),
                ('L_arc_top', 738.7, 523.3),
                ('R_arc_top', 1736.5, 581.8),
            ],
        },
    },
    # Verified-chain frame set (spikes/verify_chain_fullres.py, 2026-07-30):
    # every adjacent pair holds ratio >= 0.6 at full resolution, scorebug masked.
    # DJ's clicks from spikes/out/FULLGAME_chain_landmarks.json. Kept SEPARATE
    # from "FULL_GAME" (the old, broken 5-frame coverage set) so neither
    # overwrites the other.
    "FULL_GAME_CHAIN": {
        "video_path": r"c:\Users\djcha\New folder\basketball-cv-service\Full_Game.mp4",
        "keyframes": [600, 127200, 151200, 171000],
        "reference_pos": None,
        "exclude_regions": [(0.0, 830.0, 330.0, 1080.0)],   # bottom-left scorebug
        "court": "auto",
        "stills": [600, 127200, 151200, 171000],
        "landmarks": {
            600: [
                ('center_logo', 1044.6, 531.7),
                ('center_near', 1037.9, 327.8),
                ('center_far', 1057.9, 1003),
                ('L_FT_near', 444.6, 469.8),
                ('L_FT_far', 334.3, 605.2),
                ('R_FT_near', 1654.6, 483.2),
                ('R_FT_far', 1778.3, 626.9),
                ('LB_side_near', 185.5, 321.1),
                ('circle_left', 879.1, 535),
                ('circle_right', 1215, 538.4),
                ('circle_bottom', 1042.9, 476.5),
                ('circle_top', 1047.9, 616.9),
                ('L_arc_top', 563.2, 526.7),
                ('R_arc_top', 1534.3, 543.4),
            ],
            127200: [
                ('center_logo', 879.1, 538.4),
                ('center_near', 862.4, 327.8),
                ('center_far', 919.2, 1013),
                ('L_FT_near', 255.7, 494.9),
                ('L_FT_far', 143.7, 642),
                ('R_FT_near', 1467.4, 463.2),
                ('R_FT_far', 1591.1, 595.2),
                ('RB_side_near', 1704.7, 312.7),
                ('circle_left', 715.3, 548.4),
                ('circle_right', 1046.2, 536.7),
                ('circle_bottom', 874.1, 478.2),
                ('circle_top', 885.8, 618.6),
                ('L_arc_top', 382.7, 556.8),
                ('R_arc_top', 1358.8, 526.7),
            ],
            151200: [
                ('L_lane_base_near', 147.1, 439.8),
                ('center_logo', 1193.3, 548.4),
                ('center_near', 1195, 331.1),
                ('center_far', 1193.3, 1049.8),
                ('L_FT_near', 595, 456.5),
                ('L_FT_far', 484.7, 590.2),
                ('R_FT_near', 1845.1, 516.6),
                ('LB_side_near', 351, 302.7),
                ('circle_left', 1024.5, 545.1),
                ('circle_right', 1370.5, 556.8),
                ('circle_bottom', 1195, 484.9),
                ('circle_top', 1195, 630.3),
                ('L_arc_top', 707, 526.7),
                ('R_arc_top', 1714.8, 571.8),
            ],
            171000: [
                ('L_lane_base_near', 287.5, 429.7),
                ('L_lane_base_far', 115.3, 533.4),
                ('center_logo', 1305.3, 566.8),
                ('center_near', 1317, 347.8),
                ('center_far', 1300.3, 1049.8),
                ('L_FT_near', 713.6, 458.1),
                ('L_FT_far', 603.3, 583.5),
                ('LB_side_near', 484.7, 302.7),
                ('circle_left', 1134.8, 548.4),
                ('circle_right', 1487.5, 578.5),
                ('circle_bottom', 1307, 501.6),
                ('circle_top', 1301.9, 645.3),
                ('L_arc_top', 825.6, 526.7),
                ('R_arc_top', 1840.1, 603.6),
            ],
        },
    },
    # SECOND GYM -- the multi-gym test. 120-min game, 7 marked frames, 92 clicks
    # (DJ, 2026-07-30). Chain verified at FULL resolution with the graphic masked
    # before any clicking was requested (spikes/out/FULL_GAME2_chain_verify_fullres
    # .json -- every link >= 0.6). Two of those frames (190500, 208500) came from
    # spikes/bridge_gap_fullres.py after the planner's 165000->208800 link failed
    # the real gate at 0.590; a hand-guessed midpoint was worse on both sides.
    "FULL_GAME2": {
        "video_path": r"c:\Users\djcha\New folder\basketball-cv-service\Full_Game2.mp4",
        "keyframes": [0, 5400, 150000, 165000, 190500, 208500, 215400],
        "reference_pos": None,
        "exclude_regions": [(0.0, 870.0, 340.0, 1080.0)],   # bottom-left player overlay
        "court": "auto",
        "stills": [0, 5400, 150000, 165000, 190500, 208500, 215400],
        "landmarks": {
            0: [
                ('R_lane_base_near', 1574.4, 424.7),
                ('R_lane_base_far', 1783.3, 498.3),
                ('center_logo', 461.3, 588.5),
                ('center_near', 456.3, 406.3),
                ('center_far', 461.3, 1059.5),
                ('R_FT_near', 1185, 461.2),
                ('R_FT_far', 1338.7, 553.1),
                ('RB_side_near', 1345.4, 342.5),
                ('circle_left', 252.4, 604.9),
                ('circle_right', 656.8, 569.8),
                ('circle_bottom', 457.9, 524.7),
                ('circle_top', 466.3, 658.4),
                ('R_arc_top', 1104.7, 514.6),
            ],
            5400: [
                ('R_lane_base_near', 1612.8, 422),
                ('R_lane_base_far', 1821.7, 500.6),
                ('center_logo', 556.5, 575.8),
                ('center_near', 551.5, 393.6),
                ('center_far', 548.2, 1002),
                ('R_FT_near', 1258.5, 453.8),
                ('R_FT_far', 1378.8, 542.4),
                ('RB_side_near', 1410.6, 350.2),
                ('circle_left', 344.3, 609.2),
                ('circle_right', 738.7, 554.1),
                ('circle_bottom', 554.9, 512.3),
                ('circle_top', 559.9, 634.3),
                ('R_arc_top', 1163.2, 507.3),
            ],
            150000: [
                ('R_lane_base_near', 1766.6, 420.4),
                ('center_logo', 656.8, 557.4),
                ('center_near', 658.5, 380.3),
                ('center_far', 650.1, 1018.7),
                ('R_FT_near', 1370.5, 458.8),
                ('R_FT_far', 1524.2, 550.7),
                ('RB_side_near', 1547.6, 340.2),
                ('circle_left', 468, 569.1),
                ('circle_right', 835.7, 540.7),
                ('circle_bottom', 656.8, 502.3),
                ('circle_top', 653.5, 624.3),
                ('R_arc_top', 1281.9, 512.3),
            ],
            165000: [
                # L_lane_base_far + LB_side_near re-clicked by DJ 2026-07-30 after
                # this landmark showed 1.28 ft error. DJ confirmed the ORIGINAL
                # click was correct ("that click was perfectly fine") -- and the
                # full frame backs him up. The error is GEOMETRIC, not a mis-click:
                # this point sits at x=229 of 1920 (frame edge) at the most distant,
                # most oblique part of the court, where px->ft scaling is worst, and
                # it is the ONLY landmark clicked in a single frame, so nothing
                # averages its error down. Do not "fix" this by deleting the click.
                ('L_lane_base_near', 436.2, 428.7),
                ('L_lane_base_far', 229, 498.3),
                ('center_logo', 1539.3, 617.6),
                ('center_near', 1577.7, 428.7),
                ('center_far', 1439, 1078.2),
                ('L_FT_near', 820.6, 476.2),
                ('L_FT_far', 653.5, 558.1),
                ('LB_side_near', 678.6, 356.2),
                ('circle_left', 1340.4, 584.8),
                ('circle_right', 1751.5, 631.6),
                ('circle_bottom', 1554.3, 551.4),
                ('circle_top', 1525.9, 686.8),
                ('L_arc_top', 889.1, 531.4),
            ],
            190500: [
                ('center_logo', 1088, 538.4),
                ('center_near', 1106.4, 376.3),
                ('center_far', 1024.5, 1011.4),
                ('L_FT_near', 381.1, 471.5),
                ('L_FT_far', 202.2, 575.1),
                ('R_FT_near', 1873.5, 511.6),
                ('LB_side_near', 217.3, 356.2),
                ('circle_left', 915.9, 535),
                ('circle_right', 1265.2, 546.7),
                ('circle_bottom', 1093, 489.9),
                ('circle_top', 1079.7, 603.6),
                ('L_arc_top', 459.6, 520),
                ('R_arc_top', 1761.6, 553.4),
            ],
            208500: [
                ('L_lane_base_near', 30.1, 449.8),
                ('center_logo', 1148.2, 546.7),
                ('center_near', 1168.2, 381.3),
                ('center_far', 1093, 1004.7),
                ('L_FT_near', 454.6, 466.5),
                ('L_FT_far', 289.1, 565.1),
                ('LB_side_near', 297.5, 357.9),
                ('circle_left', 979.4, 536.7),
                ('circle_right', 1327, 555.1),
                ('circle_bottom', 1153.2, 494.9),
                ('circle_top', 1139.8, 610.2),
                ('L_arc_top', 533.1, 516.6),
                ('R_arc_top', 1846.8, 578.5),
            ],
            215400: [
                ('R_lane_base_near', 1434, 429.7),
                ('R_lane_base_far', 1637.9, 496.6),
                ('center_logo', 312.5, 620.3),
                ('center_near', 304.2, 431.4),
                ('R_FT_near', 1059.6, 471.5),
                ('R_FT_far', 1213.4, 558.4),
                ('RB_side_near', 1228.4, 342.8),
                ('circle_left', 91.9, 652),
                ('circle_right', 516.4, 591.9),
                ('circle_bottom', 310.9, 561.8),
                ('circle_top', 317.5, 693.8),
                ('R_arc_top', 987.7, 530),
            ],
        },
    },
}

# --- Clips created through the WEB APP -------------------------------------
# Games set up in the browser live as JSON in clips/ (see clip_registry.py) so
# a coach never has to edit Python. They are merged in here as ordinary CLIPS
# entries, which is what makes them first-class to every stage.
#
# Hand-written entries above WIN on a name collision: they are the validated
# baselines this engine was proven against, and a stray uploaded clip that
# happened to be named "HARD" must never silently redefine them.
def _merge_registry_clips():
    try:
        import sys as _sys
        import os as _os
        _r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _r not in _sys.path:
            _sys.path.insert(0, _r)
        import clip_registry
    except ImportError:
        return
    for _name, _doc in clip_registry.load_all().items():
        if _name in CLIPS or not clip_registry.has_calibration(_doc):
            continue
        try:
            CLIPS[_name] = clip_registry.to_calibration_entry(_doc)
        except (KeyError, TypeError, ValueError):
            # A half-written clip is expected mid-setup -- skip it rather than
            # break config loading for every other clip.
            continue


_merge_registry_clips()

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
