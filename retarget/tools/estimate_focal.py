"""Estimate the focal length of a panning (rotation-only) camera from the video background.

For a camera that only rotates, frame-to-frame homographies satisfy H ~ K R K^-1, so K^-1 H K is a rotation only for
the right focal length f. Background homographies (person masked out with GVHMR's boxes) are fitted with SIFT + RANSAC,
and f is the value that makes K^-1 H K closest to orthonormal over all pairs (principal point at the image centre,
square pixels). GVHMR otherwise assumes a ~45 deg field of view; for a telephoto shot that underestimates depth, and with
it how far the skater moves while the camera pans.
Usage (gvhmr env): python retarget/tools/estimate_focal.py video.mp4 --bbx .../preprocess/bbx.pt [--step 5]
Prints f in pixels, the horizontal FOV and GVHMR's --f_mm equivalent.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def ortho_error(H, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ H @ K
    d = np.linalg.det(M)
    if d <= 0:
        return np.inf
    M = M / np.cbrt(d)
    return np.linalg.norm(M @ M.T - np.eye(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--bbx", default=None, help="GVHMR preprocess/bbx.pt to mask the person")
    ap.add_argument("--step", type=int, default=5, help="frame gap between the two images of a pair")
    ap.add_argument("--every", type=int, default=5, help="use every n-th frame as a pair start")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    h, w = frames[0].shape
    cx, cy = w / 2, h / 2
    boxes = None
    if a.bbx:
        import torch

        boxes = torch.load(a.bbx, map_location="cpu")["bbx_xyxy"].numpy()
    sift = cv2.SIFT_create(2000)

    def feats(i):
        mask = np.full((h, w), 255, np.uint8)
        if boxes is not None:
            x0, y0, x1, y1 = boxes[i]
            pad = 0.15 * (x1 - x0)
            mask[max(int(y0 - pad), 0) : int(y1 + pad), max(int(x0 - pad), 0) : int(x1 + pad)] = 0
        return sift.detectAndCompute(frames[i], mask)

    Hs = []
    matcher = cv2.BFMatcher()
    for i in range(0, len(frames) - a.step, a.every):
        k0, d0 = feats(i)
        k1, d1 = feats(i + a.step)
        if d0 is None or d1 is None or len(k0) < 30 or len(k1) < 30:
            continue
        good = [m for m, n in matcher.knnMatch(d0, d1, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            continue
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)
        if H is None or inl.sum() < 30:
            continue
        # only pairs with real panning constrain f
        if np.linalg.norm(H[:2, 2]) < 5.0:
            continue
        Hs.append(H)
    print(f"{len(Hs)} usable background homographies (image {w}x{h})")
    fs = np.geomspace(0.3 * w, 20 * w, 400)
    err = np.array([np.median([ortho_error(H, f, cx, cy) for H in Hs]) for f in fs])
    f = fs[np.argmin(err)]
    # GVHMR create_camera_sensor: full-frame (36 mm wide) equivalent, f_px = f_mm / 36 * max(w, h)
    f_mm = f / max(w, h) * 36.0
    print(f"focal {f:.0f} px  ->  horizontal FOV {np.degrees(2 * np.arctan(cx / f)):.1f} deg,  GVHMR --f_mm {f_mm:.1f}")
    print("error curve (f px : median orthogonality error):", " ".join(f"{fs[i]:.0f}:{err[i]:.3f}" for i in range(0, len(fs), 40)))


if __name__ == "__main__":
    main()
