"""Rebuild GVHMR's world root trajectory from the camera for gliding motions (skating) filmed by a panning camera.

GVHMR's world translation integrates a learned root velocity; trained on stepping motions, it reads "legs not
stepping" as "not moving", so glides come out at ~0.2 m/s. A panning (fixed-position) camera keeps the skater
centred, so the motion lives in the camera rotation and the in-camera position instead:
    pelvis_world(t) = R_c2w(t) @ pelvis_cam(t),   R_c2w(t) = R_global(t) R_incam(t)^-1
(R_c2w from the two root orientations GVHMR predicts, smoothed). Horizontal position comes from this; height stays
GVHMR's (gravity-aligned, y-up). Writes a copy of hmr4d_results.pt with smpl_params_global.transl replaced.

--video (zoom-aware): GVHMR assumes a fixed focal length and reads depth from the person's apparent size, so when the
operator zooms to keep the skater the same size, the depth stays ~constant and the skater seems to stay put. With
--video the background (person masked with GVHMR's boxes) is tracked frame to frame with SIFT similarity transforms:
the cumulative zoom Z(t) rescales the depth (z_true = z_GVHMR * Z / median(Z); the in-camera x / y offsets do not depend
on the focal length) and the pan / tilt increments (background shift / f(t)) give the camera rotation, replacing
GVHMR's R_c2w after the first frame.
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
from scipy.spatial.transform import Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
BODY_MODELS = os.path.join(HERE, "..", "third_party", "GMR", "assets", "body_models")


def smooth_rot(rot: R, sigma: float) -> R:
    q = rot.as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return R.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))


def background_motion(video, bbx_file, f_med):
    """Per-frame cumulative zoom and camera rotation (yaw about camera y, pitch about camera x) from the background."""
    import cv2

    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    boxes = torch.load(bbx_file, map_location="cpu")["bbx_xyxy"].numpy()
    h, w = frames[0].shape
    sift, bf = cv2.SIFT_create(2000), cv2.BFMatcher()

    def feats(i):
        mask = np.full((h, w), 255, np.uint8)
        x0, y0, x1, y1 = boxes[i]
        pad = 0.15 * (x1 - x0)
        mask[max(int(y0 - pad), 0) : int(y1 + pad), max(int(x0 - pad), 0) : int(x1 + pad)] = 0
        return sift.detectAndCompute(frames[i], mask)

    n = len(frames)
    log_s, tx, ty = np.zeros(n), np.zeros(n), np.zeros(n)
    prev = feats(0)
    for i in range(1, n):
        cur = feats(i)
        (k0, d0), (k1, d1) = prev, cur
        prev = cur
        if d0 is None or d1 is None:
            continue
        good = [m for m, nn in bf.knnMatch(d0, d1, k=2) if m.distance < 0.75 * nn.distance]
        if len(good) < 15:
            continue
        A, _ = cv2.estimateAffinePartial2D(np.float32([k0[m.queryIdx].pt for m in good]), np.float32([k1[m.trainIdx].pt for m in good]),
                                           method=cv2.RANSAC, ransacReprojThreshold=2.0)  # fmt: skip
        if A is None:
            continue
        sc = np.sqrt(abs(np.linalg.det(A[:, :2])))
        log_s[i] = np.log(sc)
        # background shift of the image centre, in frame i-1 pixels
        c = np.array([w / 2, h / 2])
        shift = A[:, :2] @ c + A[:, 2] - c
        tx[i], ty[i] = shift / sc
    Z = np.exp(np.cumsum(log_s))
    Z = Z / np.median(Z)
    f = f_med * Z
    # background moving right in the image (+x) = camera turning left (negative yaw about camera +y, y down)
    dyaw, dpitch = -np.arctan(tx / f), np.arctan(ty / f)
    return Z, dyaw, dpitch


def repair_orientation_flips(c, thresh_deg=35.0, win=21, pad=3):
    """Find and bridge brief front/back flips of GVHMR's in-camera body orientation.

    Low-resolution monocular estimates sometimes flip the body's facing for a few frames and flip back (e.g. a
    -312 deg/s then +180 deg/s yaw spike while the skater only turns ~45 deg). Frames whose in-camera facing deviates
    from its running median by more than thresh_deg (plus pad frames on each side) are replaced by slerping the root
    orientation and every body-pose joint between the neighbouring good frames. Edits c in place; returns the segments.
    """
    from scipy.ndimage import median_filter

    go = R.from_rotvec(c["global_orient"].numpy())
    f = go.apply([0.0, 0.0, 1.0])
    yaw = np.unwrap(np.arctan2(f[:, 0], f[:, 2]))
    dev = np.degrees(np.abs(yaw - median_filter(yaw, size=win, mode="nearest")))
    bad = dev > thresh_deg
    for _ in range(pad):
        bad = bad | np.roll(bad, 1) | np.roll(bad, -1)
    bad[0] = bad[-1] = False
    segs, i, n = [], 0, len(bad)
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    bp = c["body_pose"].numpy().reshape(n, -1, 3)
    gorv = c["global_orient"].numpy()
    for a, b in segs:  # frames a..b-1 are bad; a-1 and b are good
        t = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
        gorv[a:b] = Slerp([0, 1], R.from_rotvec([gorv[a - 1], gorv[b]]))(t).as_rotvec()
        for k in range(bp.shape[1]):
            bp[a:b, k] = Slerp([0, 1], R.from_rotvec([bp[a - 1, k], bp[b, k]]))(t).as_rotvec()
    c["global_orient"] = torch.tensor(gorv, dtype=c["global_orient"].dtype)
    c["body_pose"] = torch.tensor(bp.reshape(n, -1), dtype=c["body_pose"].dtype)
    return segs, dev


LEG_JOINTS = (0, 1, 3, 4, 6, 7, 9, 10)  # SMPL-X body_pose: hips, knees, ankles, feet (left, right)


def repair_leg_swaps(c, betas, thresh=0.2, win=31, pad=3, max_s=0.7, isolate_s=1.5, fps=30):
    """Find and bridge brief left / right leg swaps of GVHMR's body pose.

    A monocular estimate sometimes swaps which leg is in front for a few frames and swaps back: the ankles' fore-aft
    separation in the pelvis frame jumps from -30 to +35 cm and back within 0.4 s while the skater holds the same scissor
    stance (roller40 18.7-19.1 s), and the retarget then shows a support change. Frames whose separation deviates from its
    running median by more than thresh m (plus pad frames on each side) get their leg joints slerped between the
    neighbouring good frames. Only glitch-like segments are repaired, so real crossovers survive: at most max_s long,
    back to the stance it left (separation before and after within thresh; a real change of the leading leg stays),
    and no other candidate within isolate_s (a run of them is more likely a real rhythmic alternation, e.g. strides;
    it is left alone and reported).
    Edits c["body_pose"] in place; returns (repaired segments, skipped segments, separation).
    """
    from scipy.ndimage import median_filter

    bp = c["body_pose"].float()
    n = len(bp)
    bm = smplx.create(BODY_MODELS, "smplx", gender="neutral", use_pca=False, num_betas=betas.shape[1], batch_size=n)
    with torch.no_grad():
        j = bm(betas=betas.expand(n, -1), body_pose=bp, global_orient=torch.zeros(n, 3)).joints.numpy()
    sep = j[:, 7, 2] - j[:, 8, 2]  # left minus right ankle along the body's forward axis (+z)
    bad = np.abs(sep - median_filter(sep, size=win, mode="nearest")) > thresh
    for _ in range(pad):
        bad = bad | np.roll(bad, 1) | np.roll(bad, -1)
    bad[0] = bad[-1] = False
    segs, i = [], 0
    while i < n:
        if bad[i]:
            k = i
            while k < n and bad[k]:
                k += 1
            segs.append((i, k))
            i = k
        else:
            i += 1
    keep, skipped = [], []
    for idx, (a, b) in enumerate(segs):
        returns = abs(sep[min(b, n - 1)] - sep[max(a - 1, 0)]) < thresh  # comes back to the stance it left
        near = any(abs(a - b2) < isolate_s * fps or abs(a2 - b) < isolate_s * fps for j, (a2, b2) in enumerate(segs) if j != idx)
        (keep if (b - a <= max_s * fps and returns and not near) else skipped).append((a, b))
    segs = keep
    pose = c["body_pose"].numpy().reshape(n, -1, 3)
    for a, b in segs:
        t = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
        for k in LEG_JOINTS:
            pose[a:b, k] = Slerp([0, 1], R.from_rotvec([pose[a - 1, k], pose[b, k]]))(t).as_rotvec()
    c["body_pose"] = torch.tensor(pose.reshape(n, -1), dtype=c["body_pose"].dtype)
    return segs, skipped, sep


def floor_fit_rotation(c, R_c2w, iters=15):
    """Rotation Q (about the world origin) that makes the floor, fitted to the lowest foot points of the whole clip, horizontal.

    GVHMR's gravity (camera tilt) has errors of ~10 deg; with a person walking along the viewing direction the depth offset
    between the two feet then becomes a height difference (three_turn: the planted left foot read 7 cm high while the right
    foot trailing behind it was lifted). The lowest of the ankles / toes in each frame stands on the floor, so a plane fitted
    to them gives the true up direction (needs a flat floor and a camera that only rotates). Returns (Q, tilt in degrees).
    """
    n = len(c["body_pose"])
    bm = smplx.create(BODY_MODELS, "smplx", gender="neutral", use_pca=False, num_betas=c["betas"].shape[1], batch_size=n)
    with torch.no_grad():
        J = bm(betas=c["betas"].float(), body_pose=c["body_pose"].float(), global_orient=c["global_orient"].float(), transl=c["transl"].float()).joints.numpy()
    feet = np.stack([R_c2w[i].apply(J[i, [7, 8, 10, 11]]) for i in range(n)])  # world axes, camera at the origin
    up = np.array([0.0, 1.0, 0.0])
    for _ in range(iters):
        low = feet[np.arange(n), np.argmin(feet @ up, axis=1)]
        nrm = np.linalg.svd(low - low.mean(0), full_matrices=False)[2][-1]
        up = nrm if nrm @ up > 0 else -nrm
    Q = R.align_vectors([[0.0, 1.0, 0.0]], [up])[0]
    return Q, float(np.degrees(np.arccos(np.clip(up[1], -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rot_sigma", type=float, default=15.0, help="frames (30 fps); camera rotation smoothing")
    ap.add_argument("--pos_sigma", type=float, default=4.0, help="frames; in-camera position smoothing")
    ap.add_argument("--video", default=None, help="zoom-aware mode: the video GVHMR ran on")
    ap.add_argument("--bbx", default=None, help="zoom-aware mode: GVHMR preprocess/bbx.pt (default: next to --pred)")
    ap.add_argument("--no_flip_repair", action="store_true", help="keep GVHMR's in-camera orientations as they are")
    ap.add_argument("--no_leg_swap_repair", action="store_true", help="keep GVHMR's brief left / right leg swaps")
    ap.add_argument("--floor_fit", action="store_true", help="re-level gravity with a floor plane fitted to the lowest foot points (fixes tilt errors of GVHMR's camera pose)")
    a = ap.parse_args()
    d = torch.load(a.pred, map_location="cpu")
    g, c = d["smpl_params_global"], d["smpl_params_incam"]
    if not a.no_flip_repair:
        segs, dev = repair_orientation_flips(c)
        g["body_pose"] = c["body_pose"].clone()  # the world body pose is the same local pose
        print(f"orientation flips repaired: {len(segs)} segment(s) " + ", ".join(f"{s0 / 30:.2f}-{s1 / 30:.2f} s (max dev {dev[s0:s1].max():.0f} deg)" for s0, s1 in segs))
    betas = g["betas"][:1].float()
    bm = smplx.create(BODY_MODELS, "smplx", gender="neutral", use_pca=False, num_betas=betas.shape[1])
    if not a.no_leg_swap_repair:
        segs, skipped, sep = repair_leg_swaps(c, betas)
        g["body_pose"] = c["body_pose"].clone()
        print(f"leg swaps repaired: {len(segs)} segment(s) " + ", ".join(f"{s0 / 30:.2f}-{s1 / 30:.2f} s" for s0, s1 in segs))
        if skipped:
            print("  NOT repaired (long, not returning, or in a run - check the video): " + ", ".join(f"{s0 / 30:.2f}-{s1 / 30:.2f} s" for s0, s1 in skipped))
    J0 = bm(betas=betas).joints[0, 0].detach().numpy()  # rest pelvis joint (root rotates about it)
    R_c2w = R.from_rotvec(g["global_orient"].numpy()) * R.from_rotvec(c["global_orient"].numpy()).inv()
    R_c2w = smooth_rot(R_c2w, a.rot_sigma)
    pelvis_cam = c["transl"].numpy() + J0
    if a.video:
        f_med = float(d["K_fullimg"][0][0, 0])
        bbx = a.bbx or os.path.join(os.path.dirname(os.path.abspath(a.pred)), "preprocess", "bbx.pt")
        Z, dyaw, dpitch = background_motion(a.video, bbx, f_med)
        Z = Z[: len(pelvis_cam)]
        pelvis_cam = pelvis_cam.copy()
        pelvis_cam[:, 2] *= Z  # depth scales with the true focal length; x / y offsets do not
        R_cam = [R.identity()]
        for i in range(1, len(pelvis_cam)):
            R_cam.append(R_cam[-1] * R.from_euler("YX", [dyaw[i], dpitch[i]]))
        R_c2w = R_c2w[0] * R.concatenate(R_cam)
        R_c2w = smooth_rot(R_c2w, 3.0)
        print(f"zoom-aware: background zoom x{Z.min():.2f}..x{Z.max():.2f} (median 1), depth {np.percentile(pelvis_cam[:, 2], 5):.1f}-"
              f"{np.percentile(pelvis_cam[:, 2], 95):.1f} m (GVHMR {np.percentile(c['transl'][:, 2].numpy(), 5):.1f}-{np.percentile(c['transl'][:, 2].numpy(), 95):.1f} m)")
    pelvis_cam = gaussian_filter1d(pelvis_cam, a.pos_sigma, axis=0, mode="nearest")
    if a.floor_fit:
        Q, tilt = floor_fit_rotation(c, R_c2w)
        R_c2w = Q * R_c2w
        print(f"floor fit: GVHMR's up direction was {tilt:.1f} deg off the fitted floor normal; gravity re-levelled")
    pelvis_w = R_c2w.apply(pelvis_cam)
    old = g["transl"].numpy()
    new = pelvis_w - J0
    new[:, 1] = old[:, 1]  # keep GVHMR's height (y-up)
    new[:, [0, 2]] -= new[0, [0, 2]] - old[0, [0, 2]]
    sp = lambda t: np.linalg.norm(np.diff(t[:, [0, 2]], axis=0), axis=1) * 30  # noqa: E731
    print(f"horizontal speed: GVHMR {sp(old).mean():.2f} m/s (path {sp(old).sum() / 30:.1f} m) -> camera-based {sp(new).mean():.2f} m/s "
          f"(p95 {np.percentile(sp(new), 95):.2f}, path {sp(new).sum() / 30:.1f} m); camera yaw range {np.degrees(np.ptp(R_c2w.as_euler('YXZ')[:, 0])):.0f} deg")
    d["smpl_params_global"]["transl"] = torch.tensor(new, dtype=g["transl"].dtype)
    if a.video:  # the body's world orientation must follow the same camera rotation as its position
        go = (R_c2w * R.from_rotvec(c["global_orient"].numpy())).as_rotvec()
        d["smpl_params_global"]["global_orient"] = torch.tensor(go, dtype=g["global_orient"].dtype)
    d["traj_source"] = "camera" + (" zoom-aware" if a.video else "") + " (retarget/scripts/gvhmr_camera_traj.py)"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(d, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
