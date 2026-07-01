"""STAGE 3d -- the coach-legible demo artifact (Phase 1 validation gate).

ONE clean image: the court position heatmap + a small plain-English summary
(frames used, frames skipped, mean on-court count, top occupied zones). This is
the "show my dad" artifact. Deterministic read over the team_event JSON; no new
perception, no identity, no stats beyond occupancy + mean count.
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
import stage3_heatmap as HM
from clip_config import ACTIVE_CLIP as CLIP

OUT_PNG = os.path.join(_HERE, "out", f"{CLIP.name}_phase1_demo.png")


def _pretty(zone: str) -> str:
    return zone.replace("_", " ").title()


def main():
    trusted, skipped, meta = S3.load_trusted(S3.JSON_PATH)
    occ = S3.per_zone_occupancy(trusted)
    total = sum(occ.values())
    mean_ct = S3.mean_on_court(trusted)
    top = sorted(occ, key=lambda k: -occ[k])[:3]

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.0], wspace=0.04)

    # --- left: the heatmap on the court ---
    ax = fig.add_subplot(gs[0, 0])
    pts = HM.positions(trusted)
    heat, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[42, 25],
                                range=[[0, Z.LENGTH], [0, Z.WIDTH]])
    ax.set_facecolor("0.10")
    ax.imshow(heat.T, origin="lower", extent=[0, Z.LENGTH, 0, Z.WIDTH],
              cmap="hot", alpha=0.9, interpolation="gaussian", aspect="equal")
    HM.draw_court(ax, c="white", lw=1.6)
    ax.set_xlim(-3, Z.LENGTH + 3)
    ax.set_ylim(-3, Z.WIDTH + 3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Where the players were  —  team position heatmap",
                 fontsize=14, weight="bold")

    # --- right: plain-English summary ---
    axt = fig.add_subplot(gs[0, 1])
    axt.axis("off")
    lines = [
        f"CLIP:  {meta['clip']}",
        "",
        f"Frames used:     {len(trusted)}",
        f"Frames skipped:  {len(skipped)}",
    ]
    for (fi, reason) in skipped:
        lines.append(f"   - {fi}: {reason.split('(')[0].strip()}")
    lines += [
        "",
        f"Avg bodies on court: {mean_ct:.1f}",
        "   (~10 players + refs)",
        "",
        "Most-occupied zones:",
    ]
    for z in top:
        share = occ[z] / total if total else 0
        lines.append(f"   {_pretty(z):<13} {share:>4.0%}")
    lines += [
        "",
        "Sampled frames only —",
        "proves the map works,",
        "not real season stats.",
    ]
    axt.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", family="monospace",
             fontsize=11, transform=axt.transAxes)

    fig.suptitle("Basketball Team Stats — Phase 1 demo", fontsize=16, weight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"saved demo artifact: {OUT_PNG}")
    print(f"  frames used {len(trusted)}, skipped {len(skipped)}, "
          f"mean on-court {mean_ct:.1f}, top zones {top}")


if __name__ == "__main__":
    main()
