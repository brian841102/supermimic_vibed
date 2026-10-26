"""Training clip for SuperMimic from a retargeted nemo_v3 motion: HOME -> clip -> HOME, both ends static.

--hold_s (1 s) at HOME (booster_mjlab HOME_KEYFRAME, the AMP default pose; root z 0.50, yaw of the clip's first frame),
--blend_s (1 s) cosine interpolation to the clip's first frame, the clip from --start_s to --end_s, --blend_s cosine
interpolation back to HOME (yaw of the clip's last frame) and --hold_s at HOME. The root decelerates from the clip's final speed to rest over the
blend (and accelerates from rest into the clip at the start), so a skater still gliding at the cut stops smoothly. Joints are written in SuperMimic's nemo_v3 order
(supermimic/robots/nemov3.py JOINT_NAMES = Isaac Lab's articulation order).
--smooth makes the root follow inertia across the junctions instead of the constant-rate ramp: the root translation and yaw
leave the clip with the clip's own velocity and acceleration (fitted over its last / first 0.4 s) and come to rest with a
minimum-jerk profile (a cubic in velocity: v(0) = v1, a(0) = a1, v(T) = 0, a(T) = 0), over the shortest T >= --blend_s whose
peak acceleration stays within a_max: by default the 75th percentile of the clip's own root acceleration, so the transition is
never more abrupt than the motion it joins (floor 0.3 m/s^2); --a_max_out / --a_max_in override. Joints keep their cosine blend
(same length); wheel angles stay fixed.
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


def boundary_state(clip, fps, end, n=12):
    """Root position / velocity / acceleration (xy) and yaw / yaw rate / yaw acceleration at the clip's last (end) or first
    frame: quadratic fit over n frames, expressed at that frame. Positions are the clip's own."""
    seg = clip[-n:] if end else clip[:n]
    t = (np.arange(n) - (n - 1 if end else 0)) / fps
    A = np.stack([np.ones(n), t, 0.5 * t**2], 1)
    yaw = np.unwrap(R.from_quat(seg[:, 3:7]).as_euler("ZYX")[:, 0])
    px, py = np.linalg.lstsq(A, seg[:, :2], rcond=None)[0], np.linalg.lstsq(A, yaw, rcond=None)[0]
    edge = clip[-1] if end else clip[0]
    return edge[:2], px[1], px[2], yaw[-1 if end else 0], py[1], py[2]


def stop_profile(v1, a1, T):
    """Minimum-jerk velocity from (v1, a1) to rest at T: cubic with v(T) = 0, v'(T) = 0. Returns functions v(t), integral."""
    v1, a1 = np.asarray(v1, float), np.asarray(a1, float)
    speed = np.linalg.norm(v1)
    if speed > 1e-9:  # a deceleration stronger than 3 v1 / T would make the cubic overshoot and roll backwards
        u = v1 / speed
        a1 = a1 - u * min(np.dot(a1, u) + 3 * speed / T, 0.0)
    c2, c3 = -2 * a1 / T - 3 * v1 / T**2, 2 * v1 / T**3 + a1 / T**2
    vel = lambda t: v1 + a1 * t + c2 * t**2 + c3 * t**3  # noqa: E731
    pos = lambda t: v1 * t + a1 * t**2 / 2 + c2 * t**3 / 3 + c3 * t**4 / 4  # noqa: E731
    acc = lambda t: a1 + 2 * c2 * t + 3 * c3 * t**2  # noqa: E731
    return vel, pos, acc


def own_accel_p75(clip, fps, floor=0.3):
    """75th percentile of the clip's own horizontal root acceleration (5-frame smoothing, 5 frames dropped at both ends)."""
    from scipy.ndimage import uniform_filter1d

    v = uniform_filter1d(np.gradient(clip[:, :2], 1 / fps, axis=0), 5, axis=0)
    acc = np.linalg.norm(uniform_filter1d(np.gradient(v, 1 / fps, axis=0), 5, axis=0), axis=1)[5:-5]
    return max(float(np.percentile(acc, 75)), floor)


def choose_blend_frames(v1, a1, nb_min, fps, a_max):
    """Smallest nb >= nb_min whose stop profile (T = (nb + 1) / fps) keeps the peak translational acceleration <= a_max."""
    for nb in range(nb_min, 4 * fps):
        T = (nb + 1) / fps
        _, _, acc = stop_profile(v1, a1, T)
        tt = np.linspace(0, T, 60)
        if max(np.linalg.norm(acc(t)) for t in tt) <= a_max:
            return nb
    return 4 * fps


def smooth_blend(clip_edge, b, fps, nb, sim_order, end):
    """Blend frames (nb of them) between the clip's edge frame and HOME. b = boundary_state(...). The root xy follows the
    minimum-jerk stop profile from the edge state, the yaw likewise (rate -> 0), roll / pitch and the joints cosine-blend
    to HOME. For the start side the profile is run backwards in time. Returns (frames in time order, hold frame)."""
    xy, v, a, yaw, w, al = b
    T = (nb + 1) / fps
    sgn = 1.0 if end else -1.0  # start side: time runs backwards from the clip's first frame
    vel, pos, acc = stop_profile(v, a if end else -a, T)  # start side: running backwards in time, the acceleration changes sign
    _, ypos, _ = stop_profile(np.array([w]), np.array([al if end else -al]), T)
    tk = np.arange(1, nb + 2) / fps  # nb blend frames + the first hold frame at T
    xyk = np.array([xy + sgn * pos(t) for t in tk])
    yawk = np.array([yaw + sgn * ypos(t)[0] for t in tk])
    home = home_like(clip_edge, sim_order)
    tilt = R.from_euler("Z", -yaw) * R.from_quat(clip_edge[3:7])
    ease = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, nb + 2) / (nb + 1))
    frames = []
    for k in range(nb + 1):
        f = clip_edge * (1 - ease[k]) + home * ease[k]  # joints, z, wheel angle (equal at both ends)
        f[:2] = xyk[k]
        tq = Slerp([0, 1], R.concatenate([tilt, R.identity()]))(ease[k])
        f[3:7] = (R.from_euler("Z", yawk[k]) * tq).as_quat()
        frames.append(f)
    frames = np.array(frames)
    hold = frames[-1].copy()
    hold[3:7] = R.from_euler("Z", yawk[-1]).as_quat()
    hold[7:] = home[7:]
    hold[2] = home[2]
    frames = frames[:-1] if end else frames[:-1][::-1]
    return frames, hold, dict(T=T, peak_acc=max(np.linalg.norm(acc(t)) for t in np.linspace(0, T, 60)), end_speed=float(np.linalg.norm(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CSV from gvhmr_to_nemov3.py (32 joints, NEMOV3_JOINT_ORDER)")
    ap.add_argument("--start_s", type=float, default=0.0)
    ap.add_argument("--end_s", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold_s", type=float, default=1.0)
    ap.add_argument("--blend_s", type=float, default=1.0, help="blend length (with --smooth: the minimum)")
    ap.add_argument("--smooth", action="store_true", help="root follows inertia across the junctions (see the docstring)")
    ap.add_argument("--a_max_out", type=float, default=None, help="--smooth: peak root acceleration after the clip, m/s^2 (default: p75 of the clip's own)")
    ap.add_argument("--a_max_in", type=float, default=None, help="--smooth: peak root acceleration before the clip, m/s^2 (default: p75 of the clip's own)")
    a = ap.parse_args()
    fps = 30
    c = np.loadtxt(a.src, delimiter=",")
    order = json.load(open(a.src.replace(".csv", ".json")))["joint_order"]
    sim_order = list(nemov3_joint_names())
    start, end = int(round(a.start_s * fps)), int(round(a.end_s * fps))
    clip = c[start : end + 1][:, list(range(7)) + [7 + order.index(n) for n in sim_order]]
    nh, nb = int(a.hold_s * fps), int(a.blend_s * fps)
    extra = {}
    if a.smooth:
        bo, bi = boundary_state(clip, fps, True), boundary_state(clip, fps, False)
        auto = own_accel_p75(clip, fps)
        a_out, a_in = a.a_max_out or auto, a.a_max_in or auto
        print(f"a_max: out {a_out:.2f}, in {a_in:.2f} m/s^2" + (f" (the clip's own p75 = {auto:.2f})" if auto else ""))
        nb_out = choose_blend_frames(bo[1], bo[2], nb, fps, a_out)
        nb_in = choose_blend_frames(bi[1], -bi[2], nb, fps, a_in)
        b_out, h1, so = smooth_blend(clip[-1], bo, fps, nb_out, sim_order, True)
        b_in, h0, si = smooth_blend(clip[0], bi, fps, nb_in, sim_order, False)
        extra = {"smooth": True, "blend_in_s": si["T"], "blend_out_s": so["T"], "peak_acc_in": si["peak_acc"], "peak_acc_out": so["peak_acc"]}
        print(f"smooth: start speed {np.linalg.norm(bi[1]):.2f} m/s, accel {np.linalg.norm(bi[2]):.2f}, yaw rate {np.degrees(bi[4]):+.0f} deg/s -> blend {si['T']:.2f} s, peak accel {si['peak_acc']:.2f} m/s^2 (limit {a_in:.2f})")
        print(f"        end speed {np.linalg.norm(bo[1]):.2f} m/s, accel {np.linalg.norm(bo[2]):.2f}, yaw rate {np.degrees(bo[4]):+.0f} deg/s -> blend {so['T']:.2f} s, peak accel {so['peak_acc']:.2f} m/s^2 (limit {a_out:.2f})")
    else:
        nb_in = nb_out = nb
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
    segs = {"home_hold": [0, nh - 1], "blend_in": [nh, nh + nb_in - 1], "clip": [nh + nb_in, nh + nb_in + len(clip) - 1],
            "blend_out": [nh + nb_in + len(clip), nh + nb_in + nb_out + len(clip) - 1], "home_hold_end": [nh + nb_in + nb_out + len(clip), len(out) - 1]}  # fmt: skip
    json.dump({"source": a.src, "start_s": a.start_s, "end_s": a.end_s, "fps": fps, "joint_order": sim_order, "segments": segs, **extra}, open(a.out.replace(".csv", ".json"), "w"), indent=1)
    nw = [7 + i for i, n in enumerate(sim_order) if "wheel" not in n]
    sp = np.abs(np.diff(out[:, nw], axis=0)) * fps
    print(f"wrote {a.out}: {len(out)} frames = {len(out) / fps:.1f} s; segments {segs}")
    print(f"max non-wheel joint speed: blend_in {sp[nh:nh + nb_in].max():.2f}, blend_out {sp[segs['blend_out'][0] - 1:segs['blend_out'][1] + 1].max():.2f} rad/s")


if __name__ == "__main__":
    main()
