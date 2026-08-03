"""exp_subsample.py -- the anchor-subsampling question, sized to be ANSWERED.

Run through lab.py (uploaded to the volume, executed by a warm worker) rather
than baked into the image, so the cycle is minutes not twenty minutes.

Deliberately small: two spots, few frames. The earlier version asked the same
question with 4 spots x 300 frames and ran for hours, which is how you get
two experiments a day. If the small version shows a clear answer, take it; if
it shows a marginal one, THEN spend the big run.
"""

import sys


def run(starts=None, frames=60, ns=(2, 5, 10, 15, 30)):
    sys.path.insert(0, "/app/spikes")
    sys.path.insert(0, "/app")
    import gpu_anchor_bench
    # One MARKED spot (camera settled) and one deep in a gap (camera roaming).
    # Two is the minimum that can disagree with each other.
    return gpu_anchor_bench.subsample("Full_Game_9eb8bf2a",
                                      starts or [600, 60000],
                                      n_frames=frames, ns=tuple(ns))
