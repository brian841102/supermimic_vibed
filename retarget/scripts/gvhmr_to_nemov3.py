"""GVHMR result (hmr4d_results.pt) -> nemo_v3 reference motion (plan sections 5.2-5.3).

Pipeline: GMR IK on nemov3_retarget_{full,hw}.xml (registered at runtime, GMR itself is not modified) with limb
scales measured from this subject, then wheel post-processing on the original nemov3.xml:
  ground alignment from the wheel bottoms, wheel angles integrated from the no-slip rolling constraint, lateral slip
  statistics, light smoothing, expansion to the 32 columns of NEMOV3_JOINT_ORDER.
Outputs (out_dir/<name>_<variant>.*):
  .csv  SuperMimic / LAFAN1-style: no header, 30 fps, root pos(3), root quat xyzw(4), 32 joint angles
  .pkl  booster_mjlab MotionFile (= GMR pkl): fps, root_pos, root_rot (xyzw), dof_pos (N, 32), local_body_pos, link_body_list
  .json stats (scales, slip, ground contact, limits)
Run in the `gmr` conda env: python retarget/scripts/gvhmr_to_nemov3.py --gvhmr_pred_file .../hmr4d_results.pt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.normpath(os.path.join(HERE, ".."))
GMR_ROOT = os.path.join(RT, "third_party", "GMR")
sys.path.insert(0, GMR_ROOT)
from general_motion_retargeting import params  # noqa: E402

ASSETS = os.path.join(RT, "assets", "nemo_v3")
REF_XML = os.path.join(ASSETS, "booster_mjlab_nemov3", "xmls", "nemov3.xml")
CONFIG = os.path.join(RT, "configs", "smplx_to_nemo_v3.json")
WHEEL_R = 0.075
GROUND_EPS = 0.02  # wheel bottom below this = in contact
LEG_KEYS = ["pelvis", "spine3", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"]
ARM_KEYS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]  # re-chained, GMR scale 1.0


def register(variant: str) -> str:
    name = f"nemo_v3_{variant}"
    params.ROBOT_XML_DICT[name] = os.path.join(ASSETS, "retarget_mjcf", f"nemov3_retarget_{variant}.xml")
    params.IK_CONFIG_DICT["smplx"][name] = CONFIG
    params.ROBOT_BASE_DICT[name] = "link_leg2pelvis"
    params.VIEWER_CAM_DISTANCE_DICT[name] = 1.5
    return name


def robot_arm_geometry():
    """Zero-pose shoulder offsets from the root (torso frame) and upper-arm / forearm(+hand point) lengths of nemo_v3."""
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "retarget_mjcf", "nemov3_retarget_hw.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    p = lambda b: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy()  # noqa: E731
    root = p("link_leg2pelvis")
    off = {s: p(f"link_{s[0]}ShoulderPitch") - root for s in ("left", "right")}
    return off, float(np.linalg.norm(p("link_lShoulderPitch") - p("link_lElbow"))), float(np.linalg.norm(p("link_lElbow") - p("left_hand_point")))


def rechain_arms(frames, shoulder_off, l_upper, l_fore, rot_offset=(0.5, -0.5, -0.5, -0.5)):
    """Arm targets with the human arm directions, hung from the robot's own shoulders.

    GMR scales each body about the pelvis, so with human torso proportions the shoulders land ~8 cm above the robot's
    and the IK folds the elbows. Here shoulder = pelvis' + R_chest @ shoulder_offset_robot (R_chest: human spine3
    orientation with the config's rotation offset, i.e. the torso orientation target), elbow = shoulder +
    l_upper * dir(upper arm), wrist = elbow + l_fore * dir(forearm). Returned positions reproduce these targets under
    GMR scale 1.0 (target = (p - pelvis) + scaled pelvis).
    """
    Roff = R.from_quat(rot_offset, scalar_first=True)
    out = []
    for f in frames:
        g = dict(f)
        pel = np.asarray(f["pelvis"][0])
        Rc = R.from_quat(f["spine3"][1], scalar_first=True) * Roff
        for s in ("left", "right"):
            sh, el, wr = (np.asarray(f[f"{s}_{k}"][0]) for k in ("shoulder", "elbow", "wrist"))
            u = (el - sh) / (np.linalg.norm(el - sh) + 1e-9)
            v = (wr - el) / (np.linalg.norm(wr - el) + 1e-9)
            sh_s = Rc.apply(shoulder_off[s])
            el_s = sh_s + l_upper * u
            wr_s = el_s + l_fore * v
            g[f"{s}_shoulder"] = (pel + sh_s, f[f"{s}_shoulder"][1])
            g[f"{s}_elbow"] = (pel + el_s, f[f"{s}_elbow"][1])
            g[f"{s}_wrist"] = (pel + wr_s, f[f"{s}_wrist"][1])
        out.append(g)
    return out


def robot_lengths():
    """Straight-leg hip-pitch -> wheel centre and shoulder -> hand-point lengths of nemo_v3 (zero pose)."""
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "retarget_mjcf", "nemov3_retarget_hw.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    p = lambda b: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)]  # noqa: E731
    return np.linalg.norm(p("left_hip_pitch_link") - p("left_wheel_center")), np.linalg.norm(p("link_lShoulderPitch") - p("left_hand_point"))


def human_lengths(frames):
    """95th percentile (≈ straight limb) of hip->ankle and shoulder->wrist distances over the clip."""
    leg = [np.linalg.norm(f[f"{s}_hip"][0] - f[f"{s}_ankle"][0]) for f in frames for s in ("left", "right")]
    arm = [np.linalg.norm(f[f"{s}_shoulder"][0] - f[f"{s}_wrist"][0]) for f in frames for s in ("left", "right")]
    return float(np.percentile(leg, 95)), float(np.percentile(arm, 95))


def velocity_heading(frames, fps, v0=0.3, v1=0.8, sigma=3.0):
    """Wheel heading target: blend the toe point toward the foot's direction of travel when the foot moves.

    A wheel can only roll along its heading, so for a moving foot the physically meaningful heading is its velocity
    direction; monocular foot yaw (feet are ~10 px at 360p) is too noisy for that. Blend weight ramps from 0 at v0 to
    1 at v1 (m/s, human scale); the toe stays at its original distance from the ankle, at ankle height.
    """
    from scipy.ndimage import gaussian_filter1d

    out = [dict(f) for f in frames]
    for s in ("left", "right"):
        ank = gaussian_filter1d(np.array([f[f"{s}_ankle"][0] for f in frames]), sigma, axis=0, mode="nearest")
        toe = np.array([f[f"{s}_foot"][0] for f in frames])
        v = np.gradient(ank, 1.0 / fps, axis=0)[:, :2]
        sp = np.linalg.norm(v, axis=1)
        fd = toe[:, :2] - ank[:, :2]
        L = np.linalg.norm(toe - ank, axis=1)
        fdu = fd / (np.linalg.norm(fd, axis=1, keepdims=True) + 1e-9)
        vu = v / (sp[:, None] + 1e-9)
        vu = np.where(((vu * fdu).sum(1) < 0)[:, None], -vu, vu)  # skating backwards: keep the toe side
        w = np.clip((sp - v0) / (v1 - v0), 0, 1)[:, None]
        h = (1 - w) * fdu + w * vu
        h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        for i, f in enumerate(out):
            a = np.asarray(frames[i][f"{s}_ankle"][0])
            new = np.array([a[0] + L[i] * h[i, 0], a[1] + L[i] * h[i, 1], a[2]])
            orig = np.asarray(toe[i])
            # low speed: original toe point; moving: horizontal heading point along the velocity
            f[f"{s}_foot"] = (np.where(w[i] > 0, new, orig) if w[i, 0] >= 1 else (1 - w[i, 0]) * orig + w[i, 0] * new, frames[i][f"{s}_foot"][1])
    return out


def lowpass(x, win=9, order=3):
    return savgol_filter(x, win, order, axis=0) if len(x) > win else x


def postprocess(qpos_ik, ik_names, fps, variant):
    """qpos (N, 7 + n_ik) on the retarget model -> root_pos, root_quat_xyzw, dof (N, 32) on nemov3.xml + stats."""
    m = mujoco.MjModel.from_xml_path(REF_XML)
    d = mujoco.MjData(m)
    order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    sys.path.insert(0, HERE)
    from make_retarget_mjcf import HOME_JOINT_POS, VARIANTS

    N = len(qpos_ik)
    dof = np.zeros((N, len(order)))
    for c, n in enumerate(order):
        if n in ik_names:
            dof[:, c] = qpos_ik[:, 7 + ik_names.index(n)]
        else:
            assert n in VARIANTS[variant], n
            dof[:, c] = HOME_JOINT_POS.get(n, 0.0)
    root_p = qpos_ik[:, :3].copy()
    root_q = qpos_ik[:, 3:7].copy()  # wxyz
    # smoothing: joints and root position (Savitzky-Golay ~0.3 s), root orientation via rotvec of relative rotations
    moving = [c for c, n in enumerate(order) if n in ik_names]
    dof[:, moving] = lowpass(dof[:, moving])
    root_p = lowpass(root_p)
    for i in range(1, N):  # sign-continuous quaternions, then smooth components and renormalise (safe through spins)
        if np.dot(root_q[i], root_q[i - 1]) < 0:
            root_q[i] = -root_q[i]
    root_q = lowpass(root_q)
    root_q /= np.linalg.norm(root_q, axis=1, keepdims=True)
    lim = m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in order]]
    limited = np.array([m.jnt_limited[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in order], bool)
    viol_frac = float(((dof[:, limited] < lim[limited, 0] - 1e-6) | (dof[:, limited] > lim[limited, 1] + 1e-6)).any(1).mean())
    dof[:, limited] = np.clip(dof[:, limited], lim[limited, 0], lim[limited, 1])

    qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in order}
    wheel_j = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_wheel_joint") for s in ("left", "right")}
    wheel_b = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wheel_link") for s in ("left", "right")}

    def fk(i):
        d.qpos[:3], d.qpos[3:7] = root_p[i], root_q[i]
        for c, n in enumerate(order):
            d.qpos[qadr[n]] = dof[i, c]
        mujoco.mj_kinematics(m, d)

    # 1) ground: per-frame lowest wheel bottom, smoothed, subtracted from the root height
    centres = {s: np.zeros((N, 3)) for s in wheel_j}
    axles = {s: np.zeros((N, 3)) for s in wheel_j}
    for i in range(N):
        fk(i)
        for s in wheel_j:
            centres[s][i] = d.xpos[wheel_b[s]]
            axles[s][i] = d.xaxis[wheel_j[s]]
    bottom = {s: centres[s][:, 2] - WHEEL_R * np.sqrt(np.clip(1 - axles[s][:, 2] ** 2, 0, 1)) for s in wheel_j}
    lowest = np.minimum(bottom["left"], bottom["right"])
    shift = savgol_filter(lowest, min(31, N - (1 - N % 2)), 2) if N > 31 else lowest
    root_p[:, 2] -= shift
    for s in wheel_j:
        centres[s][:, 2] -= shift
        bottom[s] -= shift
    pen = -np.minimum(np.minimum(bottom["left"], bottom["right"]), 0)
    root_p[:, 2] += pen  # nothing below the ground
    for s in wheel_j:
        bottom[s] += pen
        centres[s][:, 2] += pen
    # 2) wheels: no-slip rolling, omega = (v . d) / r while in contact
    dt = 1.0 / fps
    stats = {"joint_limit_violation_frac_before_clip": viol_frac}
    for s, c in wheel_j.items():
        v = np.gradient(centres[s], dt, axis=0)
        a = axles[s]
        roll = np.cross(a, [0.0, 0.0, 1.0])
        roll /= np.linalg.norm(roll, axis=1, keepdims=True) + 1e-9
        # no slip at the contact point (centre - r z): v + omega a x (-r z) = 0  ->  v = r omega (a x z)
        v_roll = (v * roll).sum(1)
        contact = bottom[s] < GROUND_EPS
        omega = np.where(contact, v_roll / WHEEL_R, 0.0)
        for i in range(1, N):  # decay while airborne
            if not contact[i]:
                omega[i] = 0.9 * omega[i - 1]
        dof[:, order.index(f"{s}_wheel_joint")] = np.concatenate(([0.0], np.cumsum(omega[1:] * dt)))
        speed = np.linalg.norm(v[:, :2], axis=1)
        slip = np.abs((v * a).sum(1)) / (speed + 1e-6)
        fast = contact & (speed > 0.2)
        stats[s] = {
            "contact_frac": float(contact.mean()),
            "wheel_speed_abs_p95_rad_s": float(np.percentile(np.abs(omega), 95)),
            "slip_mean_moving": float(slip[fast].mean()) if fast.any() else None,
            "slip_p95_moving": float(np.percentile(slip[fast], 95)) if fast.any() else None,
        }
    vxy = np.linalg.norm(np.gradient(root_p[:, :2], dt, axis=0), axis=1)
    stats["root_speed_mean_m_s"], stats["root_speed_p95_m_s"] = float(vxy.mean()), float(np.percentile(vxy, 95))
    stats["root_height_min_max"] = [float(root_p[:, 2].min()), float(root_p[:, 2].max())]
    legq = np.abs(np.gradient(dof[:, [order.index(n) for n in order if ("hip" in n or "calf" in n)]], dt, axis=0))
    stats["leg_joint_speed_p99_rad_s"] = float(np.percentile(legq, 99))
    # local body positions for booster_mjlab's mirror augmentation
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)]
    lbp = np.zeros((N, len(names), 3), np.float32)
    for i in range(N):
        fk(i)
        Rr = d.xmat[1].reshape(3, 3)
        lbp[i] = (d.xpos[1:] - d.xpos[1]) @ Rr
    return root_p, root_q, dof, order, names, lbp, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvhmr_pred_file", required=True)
    ap.add_argument("--variant", nargs="+", default=["full", "hw"], choices=["full", "hw"])
    ap.add_argument("--out_dir", default=os.path.join(RT, "data", "retarget_out"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--raw_heading", action="store_true", help="use the estimated toe point as wheel heading (no velocity blend)")
    a = ap.parse_args()
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.gvhmr_pred_file)))
    os.makedirs(a.out_dir, exist_ok=True)
    smplx_data, body_model, smplx_output, _ = load_gvhmr_pred_file(a.gvhmr_pred_file, os.path.join(GMR_ROOT, "assets", "body_models"))
    frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    if not a.raw_heading:
        frames = velocity_heading(frames, fps)
    h_leg, h_arm = human_lengths(frames)
    r_leg, r_arm = robot_lengths()
    s_leg, s_arm = r_leg / h_leg, r_arm / h_arm
    sh_off, l_upper, l_fore = robot_arm_geometry()
    frames = rechain_arms(frames, sh_off, l_upper, l_fore)
    print(f"[{name}] {len(frames)} frames @ {fps} fps; human leg {h_leg:.3f} arm {h_arm:.3f} -> scale leg/torso {s_leg:.3f}; "
          f"arms re-chained with robot segments {l_upper:.3f} / {l_fore:.3f} m")
    for variant in a.variant:
        robot = register(variant)
        rt = GMR(src_human="smplx", tgt_robot=robot, actual_human_height=None, verbose=False)
        for k in LEG_KEYS:
            rt.human_scale_table[k] = s_leg
        for k in ARM_KEYS:
            rt.human_scale_table[k] = 1.0
        ik_names = [mujoco.mj_id2name(rt.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(rt.model.njnt) if rt.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        qpos = np.stack([rt.retarget(f) for f in frames])
        root_p, root_q, dof, order, bodies, lbp, stats = postprocess(qpos, ik_names, fps, variant)
        stats.update(scale_leg=s_leg, scale_arm=s_arm, human_leg_m=h_leg, human_arm_m=h_arm, frames=len(frames), fps=fps, variant=variant, joint_order=order)
        quat_xyzw = root_q[:, [1, 2, 3, 0]]
        stem = os.path.join(a.out_dir, f"{name}_{variant}")
        np.savetxt(stem + ".csv", np.concatenate((root_p, quat_xyzw, dof), 1), delimiter=",", fmt="%.6f")
        with open(stem + ".pkl", "wb") as f:
            pickle.dump({"fps": float(fps), "root_pos": root_p.astype(np.float32), "root_rot": list(quat_xyzw.astype(np.float32)),
                         "dof_pos": dof.astype(np.float32), "local_body_pos": lbp, "link_body_list": bodies}, f)  # fmt: skip
        json.dump(stats, open(stem + ".json", "w"), indent=1)
        print(f"[{name}/{variant}] wrote {stem}.csv/.pkl/.json")
        print(json.dumps({k: v for k, v in stats.items() if k != "joint_order"}, indent=1))


if __name__ == "__main__":
    main()
