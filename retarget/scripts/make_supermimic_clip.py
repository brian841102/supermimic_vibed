"""Training clip for SuperMimic from a retargeted nemo_v3 motion: HOME -> clip -> HOME, both ends static.

--hold_s (1 s) at HOME (booster_mjlab HOME_KEYFRAME, the AMP default pose; root z 0.50, yaw of the clip's first frame),
--blend_s (1 s) cosine interpolation to the clip's first frame, the clip from --start_s to --end_s, --blend_s cosine
interpolation back to HOME (yaw of the clip's last frame) and --hold_s at HOME. The root decelerates from the clip's final speed to rest over the
blend (and accelerates from rest into the clip at the start), so a skater still gliding at the cut stops smoothly. Joints are written in SuperMimic's nemo_v3 order
(supermimic/robots/nemov3.py JOINT_NAMES = Isaac Lab's articulation order).
Usage: python retarget/scripts/make_supermimic_clip.py --src data/retarget_out/roller40_hw.csv --end_s 30.57 --out .../roller32.csv
(--start_s cuts the beginning, e.g. while the skater is not yet fully in the frame; --blend_s 2 for 2 s transitions)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = {"left_hip_pitch_joint": -0.26, "right_hip_pitch_joint": -0.26, "left_calf_joint": 0.43, "right_calf_joint": 0.43,
        "left_hip_roll_joint": 0.08, "right_hip_roll_joint": 0.08, "left_hip_yaw_joint": -0.09, "right_hip_yaw_joint": -0.06,
        "joint_lShoulderRoll": 1.22, "joint_rShoulderRoll": 1.22}  # fmt: skip
HOME_Z = 0.50


def nemov3_joint_names():
    """SuperMimic's nemo_v3 joint order (Isaac Lab articulation order), read from supermimic/robots/nemov3.py."""
    import ast

    src = open(os.path.join(HERE, "..", "..", "supermimic", "robots", "nemov3.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "JOINT_NAMES":
            return ast.literal_eval(node.value)
    raise RuntimeError("JOINT_NAMES not found")


def home_like(frame, order):
    h = frame.copy()
    h[2] = HOME_Z
    h[3:7] = R.from_euler("Z", R.from_quat(frame[3:7]).as_euler("ZYX")[0]).as_quat()
    h[7:] = [frame[7 + k] if "wheel" in n else HOME.get(n, 0.0) for k, n in enumerate(order)]
    return h


def blend(a, b, n):
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n + 2)[1:-1])  # cosine ease, endpoints excluded
    out = a[None] * (1 - s[:, None]) + b[None] * s[:, None]
    out[:, 3:7] = Slerp([0, 1], R.from_quat([a[3:7], b[3:7]]))(s).as_quat()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CSV from gvhmr_to_nemov3.py (32 joints, NEMOV3_JOINT_ORDER)")
    ap.add_argument("--start_s", type=float, default=0.0)
    ap.add_argument("--end_s", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold_s", type=float, default=1.0)
    ap.add_argument("--blend_s", type=float, default=1.0)
    a = ap.parse_args()
    fps = 30
    c = np.loadtxt(a.src, delimiter=",")
    order = json.load(open(a.src.replace(".csv", ".json")))["joint_order"]
    sim_order = list(nemov3_joint_names())
    start, end = int(round(a.start_s * fps)), int(round(a.end_s * fps))
    clip = c[start : end + 1][:, list(range(7)) + [7 + order.index(n) for n in sim_order]]
    nh, nb = int(a.hold_s * fps), int(a.blend_s * fps)
    T = (nb + 1) / fps
    # root velocity at the clip ends (the skater may still be gliding): accelerate from rest into the clip and
    # decelerate to rest after it at constant rate over the blend, instead of jumping from / to zero speed
    v0 = (clip[min(5, len(clip) - 1), :2] - clip[0, :2]) / (min(5, len(clip) - 1) / fps)
    v1 = (clip[-1, :2] - clip[-6, :2]) / (5 / fps)
    h0, h1 = home_like(clip[0], sim_order), home_like(clip[-1], sim_order)
    h0[:2] = clip[0, :2] - v0 * T / 2
    h1[:2] = clip[-1, :2] + v1 * T / 2
    b_in, b_out = blend(h0, clip[0], nb), blend(clip[-1], h1, nb)
    tau = np.arange(1, nb + 1)[:, None] / fps
    b_in[:, :2] = h0[:2] + v0 * tau**2 / (2 * T)
    b_out[:, :2] = clip[-1, :2] + v1 * (tau - tau**2 / (2 * T))
    print(f"root speed at the clip start {np.linalg.norm(v0):.2f} m/s, at the end {np.linalg.norm(v1):.2f} m/s (ramped over {T:.2f} s)")
    out = np.concatenate((np.repeat(h0[None], nh, 0), b_in, clip, b_out, np.repeat(h1[None], nh, 0)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savetxt(a.out, out, delimiter=",", fmt="%.6f")
    segs = {"home_hold": [0, nh - 1], "blend_in": [nh, nh + nb - 1], "clip": [nh + nb, nh + nb + len(clip) - 1],
            "blend_out": [nh + nb + len(clip), nh + 2 * nb + len(clip) - 1], "home_hold_end": [nh + 2 * nb + len(clip), len(out) - 1]}  # fmt: skip
    json.dump({"source": a.src, "start_s": a.start_s, "end_s": a.end_s, "fps": fps, "joint_order": sim_order, "segments": segs}, open(a.out.replace(".csv", ".json"), "w"), indent=1)
    nw = [7 + i for i, n in enumerate(sim_order) if "wheel" not in n]
    sp = np.abs(np.diff(out[:, nw], axis=0)) * fps
    print(f"wrote {a.out}: {len(out)} frames = {len(out) / fps:.1f} s; segments {segs}")
    print(f"max non-wheel joint speed: blend_in {sp[nh:nh + nb].max():.2f}, blend_out {sp[segs['blend_out'][0] - 1:segs['blend_out'][1] + 1].max():.2f} rad/s")


if __name__ == "__main__":
    main()
