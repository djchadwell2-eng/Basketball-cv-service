"""gpu_anchor.py -- the camera anchor, on the GPU.

WHAT THE ANCHOR IS. For each frame: where is the camera pointing? Answered by
matching the frame to the nearest hand-marked keyframe photo and solving the
homography between them. Everything downstream that needs court feet -- who is
on the floor, where a shot came from -- depends on it.

WHY THIS EXISTS. It was the pipeline's real scaling wall, and nobody had
measured it: 47 s/frame on a full game, which is 2,240 HOURS for one 95-minute
game. The GPU had already made person-detection 131x faster while this stage
sat on the CPU, so the pipeline could only ever handle ~15-second clips.

MEASURED on DJ's own game (spikes/gpu_anchor_bench.py):
    OpenCV CPU SIFT   47 s/frame
    kornia on the GPU  0.588 s/frame      = 80x faster
    agreement with the CPU answer: 0.008 ft mean, 0.11 ft max, 0 failures
0.008 ft is a tenth of an inch. For scale, DJ called a 0.21 ft court "utter
perfection" and a 0.94 ft one "broken". This is not a trade; it is the same
answer, sooner.

TWO THINGS THAT MADE IT FAST, both free:
  - the five keyframe photos are described ONCE, not re-analysed for every
    frame of the game (that alone was 4x)
  - matching runs in torch on the GPU; only RANSAC stays on the CPU, where it
    is cheap and already correct

SAFETY. If torch, kornia or CUDA are missing this module reports unavailable
and the caller keeps the CPU path -- DJ's laptop has no CUDA at all, so local
runs and the test suite are untouched.
"""

from __future__ import annotations

import numpy as np

RANSAC_PX = 3.0
N_FEATURES = 4000
LOWE_RATIO = 0.9
MIN_MATCHES = 10


def available() -> bool:
    """Is a GPU anchor possible on this machine?"""
    try:
        import torch
        import kornia  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class GpuAnchor:
    """Drop-in replacement for stage1_court_roi's anchor(f, frame) closure.

    Returns the same 4-tuple: (T, inliers, reproj_px, keyframe).
    """

    def __init__(self, kf_imgs: dict, keyframes: list, Hs_opt, exclude_regions=None):
        import torch
        import kornia.feature as KF

        self._torch = torch
        self._KF = KF
        self.device = "cuda"
        self.keyframes = list(keyframes)
        self.KF_arr = np.array(self.keyframes)
        self.Hs_opt = Hs_opt
        self.sift = KF.SIFTFeature(num_features=N_FEATURES, device=self.device).eval()
        # Described once, reused for the whole game. This is the difference
        # between 2.4 s/frame and 0.588.
        self._kf = {k: self._describe(self._tensor(img)) for k, img in kf_imgs.items()}

    def _tensor(self, img_bgr):
        import cv2
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return self._torch.from_numpy(gray)[None, None].to(self.device)

    def _describe(self, t):
        with self._torch.inference_mode():
            lafs, _, desc = self.sift(t)
        return self._KF.get_laf_center(lafs)[0], desc[0]

    def __call__(self, f, frame_bgr):
        import cv2
        k = int(self.KF_arr[np.argmin(np.abs(self.KF_arr - f))])   # nearest keyframe
        pos = self.keyframes.index(k)
        pts_k, desc_k = self._kf[k]
        pts_f, desc_f = self._describe(self._tensor(frame_bgr))
        with self._torch.inference_mode():
            _, idxs = self._KF.match_smnn(desc_k, desc_f, LOWE_RATIO)
        if idxs.shape[0] < MIN_MATCHES:
            return None, 0, float("inf"), k
        pk = pts_k[idxs[:, 0]].cpu().numpy()
        pf = pts_f[idxs[:, 1]].cpu().numpy()

        H, mask = cv2.findHomography(pf.reshape(-1, 1, 2), pk.reshape(-1, 1, 2),
                                     cv2.RANSAC, RANSAC_PX)
        if H is None or mask is None:
            return None, 0, float("inf"), k
        keep = mask.ravel().astype(bool)
        inliers = int(keep.sum())
        if inliers == 0:
            return None, 0, float("inf"), k
        proj = cv2.perspectiveTransform(pf[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
        reproj = float(np.linalg.norm(proj - pk[keep], axis=1).mean())

        T = self.Hs_opt[pos] @ H
        T = T / T[2, 2]
        return T, inliers, reproj, k
