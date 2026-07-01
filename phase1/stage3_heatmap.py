"""STAGE 3c -- top-down court diagram + position heatmap over trusted frames.

Draws a geometrically-correct HS court (boundary, lanes, free-throw arcs, center
circle, 3-pt arcs, hoops) and overlays a 2-D occupancy heatmap of the on-court
detections from the TRUSTED frames, with the zone-guide lines visible.

Reads only the team_event positions (via stage3_team_stats). No new perception.
The court-drawing helpers are reused by the Stage 3d demo artifact.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)

import zones as Z
import stage3_team_stats as S3
from clip_config import ACTIVE_CLIP as CLIP

OUT_PNG = os.path.join(_HERE, "out", f"{CLIP.name}_stage3_heatmap.png")

L, W = Z.LENGTH, Z.WIDTH                 # 84 x 50
CX, CY = L / 2.0, W / 2.0
R3 = 19.75                               # HS 3-pt radius
HOOP_DX = Z.HOOP_DEPTH                    # 5.25 ft from baseline
CIRCLE_R = 6.0


def _arc(cx, cy, r, a0, a1, n=60):
    a = np.linspace(np.radians(a0), np.radians(a1), n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def draw_court(ax, c="black", lw=1.4):
    """Geometrically-correct HS court, top-down, in feet."""
    ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color=c, lw=lw)          # boundary
    ax.plot([CX, CX], [0, W], color=c, lw=lw)                          # center line
    ax.add_patch(plt.Circle((CX, CY), CIRCLE_R, fill=False, ec=c, lw=lw))
    for x0 in (0.0, L):                                                # both ends
        s = 1 if x0 == 0 else -1
        # lane + free-throw line
        ax.plot([x0, x0 + s * Z.FT_X, x0 + s * Z.FT_X, x0],
                [Z.LANE_Y0, Z.LANE_Y0, Z.LANE_Y1, Z.LANE_Y1], color=c, lw=lw)
        # free-throw circle (top half, toward midcourt)
        fx, fy = _arc(x0 + s * Z.FT_X, CY, CIRCLE_R, -90 if s > 0 else 90,
                      90 if s > 0 else 270)
        ax.plot(fx, fy, color=c, lw=lw)
        # 3-pt: corner straights + arc
        hoop_x = x0 + s * HOOP_DX
        ax.plot([x0, hoop_x], [CY - R3, CY - R3], color=c, lw=lw)
        ax.plot([x0, hoop_x], [CY + R3, CY + R3], color=c, lw=lw)
        ax_, ay_ = _arc(hoop_x, CY, R3, -90 if s > 0 else 90, 90 if s > 0 else 270)
        ax.plot(ax_, ay_, color=c, lw=lw)
        ax.add_patch(plt.Circle((hoop_x, CY), 0.75, fill=False, ec=c, lw=lw))  # rim


def draw_zone_guides(ax, c="tab:blue"):
    """Dashed lines marking the zone-depth bands on both halves."""
    for x0 in (0.0, L):
        s = 1 if x0 == 0 else -1
        for depth in (Z.CORNER_DEPTH, Z.ARC_TOP_DEPTH):               # 14, 25
            xd = x0 + s * depth
            ax.plot([xd, xd], [0, W], color=c, lw=0.7, ls="--", alpha=0.5)
    for yl in (Z.LANE_Y0, Z.LANE_Y1):                                 # side splits
        ax.plot([0, L], [yl, yl], color=c, lw=0.7, ls="--", alpha=0.5)


def positions(events):
    pts = [det.court_feet for ev in events for det in ev.detections]
    return np.array(pts, dtype=float) if pts else np.empty((0, 2))


def render(events, path, title):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    pts = positions(events)
    if len(pts):
        heat, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1],
                                      bins=[42, 25], range=[[0, L], [0, W]])
        im = ax.imshow(heat.T, origin="lower", extent=[0, L, 0, W],
                       cmap="hot", alpha=0.85, interpolation="gaussian",
                       aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                     label="body-positions per 2 ft cell")
    ax.set_facecolor("0.12")
    draw_court(ax, c="white", lw=1.4)
    draw_zone_guides(ax, c="deepskyblue")
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c="cyan", alpha=0.5, edgecolors="none")
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(-3, W + 3)
    ax.set_title(title)
    ax.set_xlabel("court length (ft)")
    ax.set_ylabel("court width (ft)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    trusted, skipped, meta = S3.load_trusted(S3.JSON_PATH)
    title = (f"{meta['clip']} -- on-court position heatmap "
             f"({len(trusted)} trusted frames, {len(skipped)} skipped)")
    out = render(trusted, OUT_PNG, title)
    print(f"trusted frames: {len(trusted)}  skipped: {len(skipped)}")
    print(f"saved heatmap: {out}")


if __name__ == "__main__":
    main()
