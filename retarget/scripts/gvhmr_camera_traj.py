"""Rebuild GVHMR's world root trajectory from the camera for gliding motions (skating) filmed by a panning camera.

GVHMR's world translation integrates a learned root velocity; trained on stepping motions, it reads "legs not
stepping" as "not moving", so glides come out at ~0.2 m/s. A panning (fixed-position) camera keeps the skater
centred, so the motion lives in the camera rotation and the in-camera position instead:
    pelvis_world(t) = R_c2w(t) @ pelvis_cam(t),   R_c2w(t) = R_global(t) R_incam(t)^-1
(R_c2w from the two root orientations GVHMR predicts, smoothed). Horizontal position comes from this; height stays
GVHMR's (gravity-aligned, y-up). Writes a copy of hmr4d_results.pt with smpl_params_global.transl replaced.
Run in the `gmr` env (needs torch, smplx): python retarget/scripts/gvhmr_camera_traj.py --pred .../hmr4d_results.pt --out ...
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import smplx
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
BODY_MODELS = os.path.join(HERE, "..", "third_party", "GMR", "assets", "body_models")


def smooth_rot(rot: R, sigma: float) -> R:
    q = rot.as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return R.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rot_sigma", type=float, default=15.0, help="frames (30 fps); camera rotation smoothing")
    ap.add_argument("--pos_sigma", type=float, default=4.0, help="frames; in-camera position smoothing")
    a = ap.parse_args()
    d = torch.load(a.pred, map_location="cpu")
    g, c = d["smpl_params_global"], d["smpl_params_incam"]
    betas = g["betas"][:1].float()
    bm = smplx.create(BODY_MODELS, "smplx", gender="neutral", use_pca=False, num_betas=betas.shape[1])
    J0 = bm(betas=betas).joints[0, 0].detach().numpy()  # rest pelvis joint (root rotates about it)
    R_c2w = R.from_rotvec(g["global_orient"].numpy()) * R.from_rotvec(c["global_orient"].numpy()).inv()
    R_c2w = smooth_rot(R_c2w, a.rot_sigma)
    pelvis_cam = gaussian_filter1d(c["transl"].numpy() + J0, a.pos_sigma, axis=0, mode="nearest")
    pelvis_w = R_c2w.apply(pelvis_cam)
    old = g["transl"].numpy()
    new = pelvis_w - J0
    new[:, 1] = old[:, 1]  # keep GVHMR's height (y-up)
    new[:, [0, 2]] -= new[0, [0, 2]] - old[0, [0, 2]]
    sp = lambda t: np.linalg.norm(np.diff(t[:, [0, 2]], axis=0), axis=1) * 30  # noqa: E731
    print(f"horizontal speed: GVHMR {sp(old).mean():.2f} m/s (path {sp(old).sum() / 30:.1f} m) -> camera-based {sp(new).mean():.2f} m/s "
          f"(p95 {np.percentile(sp(new), 95):.2f}, path {sp(new).sum() / 30:.1f} m); camera yaw range {np.degrees(np.ptp(R_c2w.as_euler('YXZ')[:, 0])):.0f} deg")
    d["smpl_params_global"]["transl"] = torch.tensor(new, dtype=g["transl"].dtype)
    d["traj_source"] = "camera (retarget/scripts/gvhmr_camera_traj.py)"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(d, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
