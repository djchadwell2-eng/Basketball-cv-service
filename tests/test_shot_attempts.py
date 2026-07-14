"""Shot-attempt classification -- pure geometry tests (synthetic arcs +
synthetic hoop positions, no video). A claimed arc (already physics-gated
by ball_trajectory.py) becomes a SHOT ATTEMPT only if, at or after its
apex (the descending half -- a rising ball hasn't reached the rim yet),
it passes within HOOP_RADIUS_PX of the hoop position AT THAT SAME FRAME.
Floor-level flight (dribbles) can never satisfy this by geometry alone,
not by a special case.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from shot_attempts import (  # noqa: E402
    HOOP_RADIUS_PX, RELEASE_BACK_MAX_FRAMES, RELEASE_DIST_GATE_PX,
    apex_index, classify_shot, find_release, nearest_track_feet,
    point_to_bbox_dist,
)


def arc_points(frames, x0, vx, y0, vy, ay):
    """Same ballistic shape helper as test_ball_trajectory.py."""
    t0 = frames[0]
    return [(f, x0 + vx * (f - t0), y0 + vy * (f - t0) + 0.5 * ay * (f - t0) ** 2)
            for f in frames]


def const_hoop(hx, hy):
    return lambda frame: (hx, hy)


# ---------------------------------------------------------------- apex

def test_apex_is_the_min_y_point_since_image_y_grows_downward():
    pts = arc_points(range(0, 10), x0=0, vx=10, y0=300, vy=-20, ay=4)
    i = apex_index(pts)
    ys = [p[2] for p in pts]
    assert pts[i][2] == min(ys)


# ------------------------------------------------------------ classification

def test_arc_passing_through_hoop_region_on_descent_is_a_shot():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    # apex ~ t=9 (y min); pick a hoop position matching a late point closely
    hoop_frame, hx, hy = 18, pts[18][1], pts[18][2]
    out = classify_shot(pts, const_hoop(hx, hy))
    assert out["verdict"] == "shot_attempt"
    assert out["min_dist"] <= HOOP_RADIUS_PX


def test_same_arc_hoop_moved_far_away_is_not_a_shot():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    out = classify_shot(pts, const_hoop(5000, 5000))
    assert out["verdict"] == "not_shot"


def test_floor_bounce_arc_never_reaches_a_rim_high_in_the_frame():
    """A dribble bounce: apex near the floor (y~600), hoop near the top of
    the frame (y~200). Geometry alone rules it out -- no special case."""
    pts = arc_points(range(0, 12), x0=400, vx=3, y0=700, vy=-15, ay=3.0)
    out = classify_shot(pts, const_hoop(400, 200))
    assert out["verdict"] == "not_shot"


def test_proximity_before_apex_does_not_count_as_a_shot():
    """The ball passing near the hoop's (x,y) coordinates on the way UP
    (before apex) must not count -- only descending/at-apex does."""
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    i_apex = apex_index(pts)
    early_frame, ex, ey = pts[2]
    assert 2 < i_apex        # confirm this point IS before the apex
    out = classify_shot(pts, const_hoop(ex, ey))
    assert out["verdict"] == "not_shot"


# ------------------------------------------------------ origin gate (§18)

def test_arc_originating_near_the_hoop_is_rejected_not_a_fresh_shot():
    """A deflection/continuation starts already close to the hoop and
    moves away -- must be rejected regardless of the apex-based check."""
    pts = arc_points(range(0, 15), x0=0, vx=5, y0=200, vy=2, ay=1.0)  # starts AT (0,200)
    out = classify_shot(pts, const_hoop(0, 200))   # hoop is exactly the start point
    assert out["verdict"] == "not_shot"
    assert "originates" in out["reason"]


def test_origin_gate_skipped_when_first_frame_hoop_position_is_unknown():
    """Absence of data at the first frame must not manufacture a
    rejection -- falls back to the ordinary apex-based check."""
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    def sparse_hoop(f):
        return None if f == 0 else (pts[-1][1], pts[-1][2])
    out = classify_shot(pts, sparse_hoop)
    assert out["verdict"] == "shot_attempt"


def _hoop_lookup(pairs):
    table = {f: (x, y) for f, x, y in pairs}
    return lambda f: table.get(f)


def test_real_near_hoop_shot_356_381_survives_the_origin_gate():
    """REGRESSION, real HARD data (DECISIONS 18): user-confirmed real shot
    at the near hoop. Camera pans hard during this arc (hoop x shifts
    73->371px over 25 frames) yet the arc still correctly originates far
    from frame-356's hoop position and arrives close -- must stay a shot."""
    pts = [(356, 358.1, 197.7), (357, 347.4, 186.9), (358, 336.8, 177.0),
           (359, 326.8, 167.9), (360, 318.4, 159.3), (361, 311.2, 152.7),
           (362, 304.6, 146.1), (363, 299.7, 141.3), (364, 295.0, 137.2),
           (365, 291.7, 133.7), (366, 290.1, 131.6), (367, 289.3, 130.4),
           (368, 289.4, 130.1), (369, 290.7, 131.2), (370, 292.8, 132.7),
           (371, 296.7, 135.6), (372, 301.6, 139.7), (373, 307.5, 144.6),
           (374, 314.9, 150.6), (375, 323.1, 157.4), (376, 332.2, 165.1),
           (377, 342.6, 174.3), (378, 354.5, 184.2), (379, 367.1, 195.1),
           (381, 390.5, 213.8)]
    hoop_at = _hoop_lookup([
        (356, 73.0, 201.4), (357, 74.7, 202.0), (358, 76.2, 202.6),
        (359, 78.9, 203.1), (360, 82.6, 203.8), (361, 87.3, 204.2),
        (362, 93.0, 204.6), (363, 99.7, 205.1), (364, 107.4, 205.5),
        (365, 115.9, 205.8), (366, 127.0, 208.6), (367, 135.8, 206.4),
        (368, 147.3, 206.7), (369, 159.5, 207.0), (370, 172.7, 207.4),
        (371, 186.9, 207.7), (372, 202.0, 208.2), (373, 218.0, 208.7),
        (374, 234.6, 209.1), (375, 252.3, 209.7), (376, 271.1, 210.8),
        (377, 290.5, 211.5), (378, 310.8, 212.6), (379, 331.8, 213.7),
        (380, 352.1, 214.9), (381, 371.2, 216.1)])
    out = classify_shot(pts, hoop_at)
    assert out["verdict"] == "shot_attempt"


def test_real_near_hoop_deflection_418_438_rejected_by_the_origin_gate():
    """REGRESSION, real HARD data (DECISIONS 18): user identified this as
    a rebound/continuation, not a fresh shot. Starts 69px from the hoop,
    ends 374px away -- the origin gate now correctly excludes it."""
    pts = [(418, 633.1, 290.9), (419, 626.5, 293.1), (420, 619.2, 297.4),
           (421, 612.8, 302.9), (422, 605.9, 308.3), (423, 600.0, 315.3),
           (424, 593.0, 324.1), (425, 586.9, 332.6), (426, 579.3, 342.6),
           (427, 572.2, 354.1), (428, 563.2, 366.9), (429, 555.5, 380.1),
           (430, 546.4, 394.8), (431, 539.0, 411.6), (432, 529.9, 428.2),
           (433, 519.7, 445.8), (434, 511.5, 466.0), (435, 501.5, 486.0),
           (436, 494.1, 509.0), (437, 484.8, 531.8), (438, 478.3, 538.1)]
    hoop_at = _hoop_lookup([
        (418, 658.3, 226.9), (419, 644.5, 224.7), (420, 661.7, 225.5),
        (421, 663.2, 224.7), (422, 664.7, 223.9), (423, 649.1, 218.9),
        (424, 667.3, 222.3), (425, 668.7, 221.7), (426, 653.0, 216.8),
        (427, 654.0, 216.0), (428, 653.8, 214.8), (429, 653.6, 213.8),
        (430, 653.3, 212.7), (431, 654.6, 214.6), (432, 654.6, 213.6),
        (433, 654.5, 212.6), (434, 654.4, 211.6), (435, 654.2, 210.7),
        (436, 654.1, 209.7), (437, 652.8, 205.8), (438, 653.3, 207.5)])
    out = classify_shot(pts, hoop_at)
    assert out["verdict"] == "not_shot"
    assert "originates" in out["reason"]


def test_real_far_hoop_shot_1188_1211_survives_the_origin_gate():
    """REGRESSION, real HARD data: the original user-verified shot. Camera
    is nearly static here (hoop x drifts only ~4px) -- must stay a shot."""
    pts = [(1188, 942.6, 146.0), (1189, 953.7, 138.5), (1190, 965.1, 131.2),
           (1191, 977.0, 125.2), (1192, 988.4, 119.1), (1193, 1000.2, 114.4),
           (1194, 1012.4, 110.6), (1195, 1024.6, 108.0), (1196, 1036.7, 106.7),
           (1197, 1049.1, 105.9), (1198, 1061.5, 106.0), (1199, 1074.5, 106.7),
           (1200, 1086.4, 108.2), (1201, 1098.9, 110.2), (1202, 1111.7, 115.0),
           (1205, 1150.0, 131.9), (1206, 1162.3, 138.9), (1207, 1176.7, 146.0),
           (1208, 1189.7, 154.7), (1209, 1203.5, 164.3), (1210, 1217.3, 173.9),
           (1211, 1231.1, 185.3)]
    hoop_at = _hoop_lookup([
        (1188, 1268.7, 235.2), (1189, 1268.1, 235.1), (1190, 1267.5, 235.0),
        (1191, 1266.9, 234.9), (1192, 1266.0, 234.8), (1193, 1265.5, 234.8),
        (1194, 1264.9, 234.8), (1195, 1264.4, 234.8), (1196, 1263.8, 234.8),
        (1197, 1263.3, 234.8), (1198, 1262.8, 234.9), (1199, 1262.3, 234.9),
        (1200, 1261.8, 234.9), (1201, 1261.3, 234.8), (1202, 1260.9, 234.8),
        (1205, 1260.0, 234.9), (1206, 1260.2, 234.3), (1207, 1260.9, 233.2),
        (1208, 1261.7, 232.1), (1209, 1262.4, 231.0), (1210, 1263.2, 229.9),
        (1211, 1264.0, 228.8)])
    out = classify_shot(pts, hoop_at)
    assert out["verdict"] == "shot_attempt"


def test_real_far_hoop_deflection_1217_1250_rejected_by_the_origin_gate():
    """REGRESSION, real HARD data (DECISIONS 15/18): the original rim-out
    deflection that motivated this whole gate. Starts 26px from the hoop,
    ends 384px away -- must now be excluded, not double-counted."""
    pts = [(1217, 1258.2, 198.8), (1218, 1266.4, 190.8), (1219, 1275.6, 183.9),
           (1220, 1283.3, 177.8), (1223, 1309.5, 165.0), (1224, 1317.7, 162.5),
           (1227, 1345.9, 162.1), (1228, 1354.8, 164.4), (1229, 1363.9, 167.5),
           (1230, 1373.4, 171.7), (1231, 1381.7, 177.3), (1232, 1391.0, 183.9),
           (1233, 1399.4, 191.5), (1234, 1408.3, 200.7), (1235, 1417.4, 211.5),
           (1236, 1426.3, 223.4), (1237, 1434.9, 236.5), (1238, 1444.5, 251.6),
           (1239, 1454.3, 267.8), (1240, 1461.1, 284.7), (1241, 1469.0, 303.2),
           (1242, 1475.4, 322.7), (1243, 1481.9, 342.7), (1244, 1489.2, 364.2),
           (1245, 1495.9, 386.1), (1246, 1500.8, 408.8), (1247, 1507.5, 434.4),
           (1248, 1513.9, 458.5), (1249, 1519.5, 485.5), (1250, 1526.2, 513.3)]
    hoop_at = _hoop_lookup([
        (1217, 1269.2, 222.5), (1218, 1270.0, 221.5), (1219, 1271.0, 220.4),
        (1220, 1271.9, 219.3), (1223, 1274.7, 216.1), (1224, 1275.6, 215.0),
        (1227, 1278.5, 211.7), (1228, 1279.5, 210.7), (1229, 1280.4, 209.6),
        (1230, 1281.3, 208.5), (1231, 1282.3, 207.4), (1232, 1283.3, 206.2),
        (1233, 1284.3, 205.2), (1234, 1285.3, 204.1), (1235, 1285.9, 203.5),
        (1236, 1287.1, 203.1), (1237, 1288.9, 202.9), (1238, 1291.2, 202.8),
        (1239, 1292.7, 202.9), (1240, 1294.3, 203.1), (1241, 1294.2, 203.6),
        (1242, 1294.6, 204.1), (1243, 1294.7, 204.8), (1244, 1294.7, 205.3),
        (1245, 1294.5, 206.0), (1246, 1294.1, 206.6), (1247, 1293.7, 207.3),
        (1248, 1293.2, 207.9), (1249, 1292.9, 208.6), (1250, 1292.2, 209.1)])
    out = classify_shot(pts, hoop_at)
    assert out["verdict"] == "not_shot"
    assert "originates" in out["reason"]


def test_hoop_lookup_returning_none_for_a_frame_is_skipped_not_a_crash():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    def sparse_hoop(f):
        return None if f < 15 else (pts[-1][1], pts[-1][2])
    out = classify_shot(pts, sparse_hoop)
    assert out["verdict"] == "shot_attempt"


def test_no_hoop_data_anywhere_is_honestly_not_shot_not_a_crash():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    out = classify_shot(pts, lambda f: None)
    assert out["verdict"] == "not_shot"
    assert out["min_dist"] is None


# ------------------------------------------------------------ shooter join

def test_nearest_track_feet_picks_the_closest_body():
    tracks = [{"track_id": 1, "bbox": [0, 0, 20, 100]},      # feet (10,100)
              {"track_id": 2, "bbox": [190, 0, 210, 100]}]   # feet (200,100)
    tid, dist = nearest_track_feet(tracks, 15, 100)
    assert tid == 1
    assert dist < 10


def test_nearest_track_feet_on_empty_frame_is_none():
    assert nearest_track_feet([], 15, 100) is None


# --------------------------------------------------- point-to-bbox distance

def test_point_inside_bbox_is_zero_distance():
    assert point_to_bbox_dist(50, 50, (0, 0, 100, 100)) == 0.0


def test_point_outside_bbox_measures_to_nearest_edge():
    # 10px right of the right edge, 0 vertical offset (inside y-range)
    assert point_to_bbox_dist(110, 50, (0, 0, 100, 100)) == 10.0
    # diagonal: 3-4-5 triangle to the corner
    assert point_to_bbox_dist(103, 104, (0, 0, 100, 100)) == 5.0


# ---------------------------------------------- release back-extrapolation

def _fit_of(pts_by_t):
    """Exact quadratic fit through 3 (t, value) pairs -- lets tests build a
    fit_x/fit_y whose backward extrapolation lands EXACTLY on a chosen point."""
    import numpy as np
    ts = np.array([t for t, _ in pts_by_t], dtype=float)
    vs = np.array([v for _, v in pts_by_t], dtype=float)
    a, b, c = np.polyfit(ts, vs, 2)
    return [float(a), float(b), float(c)]


def test_release_finder_picks_body_near_backward_extrapolated_path():
    # Ball path: cy(t) descends toward the hoop for t>=0; extrapolated
    # backward it should pass through (200,500) at t=-5 (release).
    fit_y = _fit_of([(-5, 500), (0, 300), (10, 200)])
    fit_x = _fit_of([(-5, 200), (0, 250), (10, 400)])
    start_frame = 1188
    tracks_by_frame = {1183: [{"track_id": 42, "bbox": [180, 480, 220, 560]}]}
    out = find_release(fit_x, fit_y, start_frame, tracks_by_frame)
    assert out["status"] == "found"
    assert out["release_frame"] == 1183
    assert out["track_id"] == 42


def test_release_finder_abstains_when_nothing_is_close():
    fit_y = _fit_of([(-5, 500), (0, 300), (10, 200)])
    fit_x = _fit_of([(-5, 200), (0, 250), (10, 400)])
    tracks_by_frame = {f: [{"track_id": 1, "bbox": [1500, 1500, 1550, 1600]}]
                       for f in range(1178, 1188)}
    out = find_release(fit_x, fit_y, 1188, tracks_by_frame)
    assert out["status"] == "no_confident_shooter"


def test_release_finder_never_looks_past_the_frame_bound():
    """A body sitting exactly on the extrapolated path, but only at a frame
    BEYOND RELEASE_BACK_MAX_FRAMES, must be invisible to the finder."""
    fit_y = _fit_of([(-20, 500), (0, 300), (10, 200)])
    fit_x = _fit_of([(-20, 200), (0, 250), (10, 400)])
    start_frame = 1188
    too_far_frame = start_frame - RELEASE_BACK_MAX_FRAMES - 5
    ax, bx, cx = fit_x
    ay, by, cy = fit_y
    t = too_far_frame - start_frame
    x, y = ax * t * t + bx * t + cx, ay * t * t + by * t + cy
    tracks_by_frame = {too_far_frame: [{"track_id": 9,
                                        "bbox": [x - 10, y - 10, x + 10, y + 10]}]}
    out = find_release(fit_x, fit_y, start_frame, tracks_by_frame)
    assert out["status"] == "no_confident_shooter"


def test_release_finder_respects_the_distance_gate():
    fit_y = _fit_of([(-5, 500), (0, 300), (10, 200)])
    fit_x = _fit_of([(-5, 200), (0, 250), (10, 400)])
    far_bbox = [200 + RELEASE_DIST_GATE_PX * 3, 500, 220 + RELEASE_DIST_GATE_PX * 3, 560]
    tracks_by_frame = {1183: [{"track_id": 7, "bbox": far_bbox}]}
    out = find_release(fit_x, fit_y, 1188, tracks_by_frame)
    assert out["status"] == "no_confident_shooter"
