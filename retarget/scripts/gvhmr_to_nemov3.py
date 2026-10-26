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

sys.path.insert(0, HERE)
from make_retarget_mjcf import HEADING_AHEAD  # noqa: E402

ASSETS = os.path.join(RT, "assets", "nemo_v3")
REF_XML = os.path.join(ASSETS, "booster_mjlab_nemov3", "xmls", "nemov3.xml")
CONFIG = os.path.join(RT, "configs", "smplx_to_nemo_v3.json")
WHEEL_R = 0.075
GROUND_EPS = 0.02  # wheel bottom below this = in contact
ROBOT_HALF_TRACK = 0.086  # m, wheel centre from the mid-plane at the zero pose
KNEE_MIN_IK = 0.15  # rad, calf lower bound used inside the IK only
ELBOW_MAX_IK = 0.0  # rad, elbow upper bound inside the IK only (natural bend is negative; > 0 folds the arm backwards)
ROBOT_HIP_DROP = 0.097  # m, hip pitch joints below the root (link_leg2pelvis) at the zero pose
HIP_YAW_RATE_MAX = 5.0  # rad/s: hip yaw steers the wheel; keep steering gentle (URDF limit 10)
LEG_KEYS = ["pelvis", "spine3", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"]
ARM_KEYS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]  # re-chained, GMR scale 1.0


def register(variant: str, config: str = CONFIG) -> str:
    name = f"nemo_v3_{variant}"
    params.ROBOT_XML_DICT[name] = os.path.join(ASSETS, "retarget_mjcf", f"nemov3_retarget_{variant}.xml")
    params.IK_CONFIG_DICT["smplx"][name] = config
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


def rate_limit(x, max_step):
    """Symmetric slew-rate limit: forward and backward clamped passes averaged (no one-sided lag)."""
    f = x.copy()
    for i in range(1, len(f)):
        f[i] = f[i - 1] + np.clip(f[i] - f[i - 1], -max_step, max_step)
    b = x.copy()
    for i in range(len(b) - 2, -1, -1):
        b[i] = b[i + 1] + np.clip(b[i] - b[i + 1], -max_step, max_step)
    return 0.5 * (f + b)


def velocity_heading(frames, fps, ahead, v0=0.3, v1=0.8, sigma=3.0, yaw_sigma=5.0, yaw_rate_max=3.0, max_turnout=np.radians(75)):
    """Wheel heading target: a smooth, rate-limited yaw that follows the foot's travel direction when it moves.

    A wheel can only roll along its heading, so for a moving foot the physically meaningful heading is its velocity
    direction; monocular foot yaw (feet are ~10 px at 360p) is too noisy for that. Per frame the target yaw blends the
    toe yaw and the travel yaw (weight ramps 0 -> 1 between v0 and v1 m/s, human scale); the travel direction's sign
    (forward / backward rolling) is taken relative to the pelvis facing, and the heading stays within max_turnout of it
    (feet turn out <= ~60 deg when stroking; without this, noisy toe yaw sent the hip yaw to its +-135 deg limit). The yaw is
    then unwrapped, low-passed (yaw_sigma frames) and slew-limited to yaw_rate_max rad/s, because the real hip yaw
    cannot steer the wheel abruptly. The heading point sits at ankle height, `ahead` (human units) in front of the
    ankle: HEADING_AHEAD / s_leg, so after GMR's scaling it matches the robot's heading marker and only steers (a
    human toe length, ~6 cm scaled, pulled the wheel forward against the 12 cm marker).
    """
    from scipy.ndimage import gaussian_filter1d

    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    out = [dict(f) for f in frames]
    yaw_pel = pelvis_yaw(frames)
    for s in ("left", "right"):
        ank_raw = np.array([f[f"{s}_ankle"][0] for f in frames])
        ank = gaussian_filter1d(ank_raw, sigma, axis=0, mode="nearest")
        toe = np.array([f[f"{s}_foot"][0] for f in frames])
        v = np.gradient(ank, 1.0 / fps, axis=0)[:, :2]
        sp = np.linalg.norm(v, axis=1)
        w = gaussian_filter1d(np.clip((sp - v0) / (v1 - v0), 0, 1), sigma, mode="nearest")
        yaw_toe = np.arctan2(toe[:, 1] - ank_raw[:, 1], toe[:, 0] - ank_raw[:, 0])
        yaw_vel = np.arctan2(v[:, 1], v[:, 0])
        yaw = np.zeros(len(frames))
        prev = yaw_pel[0]
        for i in range(len(frames)):
            # signs relative to the body's facing: a wheel rolls forward or backward but never points behind the body
            toe_i = yaw_toe[i] if abs(wrap(yaw_toe[i] - yaw_pel[i])) <= np.pi / 2 else yaw_toe[i] + np.pi
            cand = yaw_vel[i] if abs(wrap(yaw_vel[i] - yaw_pel[i])) <= np.pi / 2 else yaw_vel[i] + np.pi
            target = toe_i + w[i] * wrap(cand - toe_i)
            target = yaw_pel[i] + np.clip(wrap(target - yaw_pel[i]), -max_turnout, max_turnout)
            yaw[i] = prev + wrap(target - prev)  # unwrapped, continuous
            prev = yaw[i]
        yaw = gaussian_filter1d(yaw, yaw_sigma, mode="nearest")
        yaw = rate_limit(yaw, yaw_rate_max / fps)
        for i, f in enumerate(out):
            a = ank_raw[i]
            f[f"{s}_foot"] = (np.array([a[0] + ahead * np.cos(yaw[i]), a[1] + ahead * np.sin(yaw[i]), a[2]]), frames[i][f"{s}_foot"][1])
    return out


def pelvis_yaw(frames, sigma=5.0):
    """Facing yaw of the human pelvis (SMPL body faces its local +z; GMR's world is z-up), unwrapped and smoothed."""
    from scipy.ndimage import gaussian_filter1d

    fwd = np.array([R.from_quat(f["pelvis"][1], scalar_first=True).apply([0.0, 0.0, 1.0]) for f in frames])
    return gaussian_filter1d(np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0])), sigma, mode="nearest")


def adapt_legs(frames, s_leg, robot_half_track, robot_hip_drop):
    """Move the leg targets (knee, ankle, toe) so the IK sees hip-to-foot geometry, not pelvis-to-foot geometry.

    GMR places the robot root at the scaled human pelvis. nemo_v3's hip pitch joints sit robot_hip_drop (9.7 cm) below
    its root, the scaled human hips only ~3-4 cm below the pelvis, so the robot lost ~6 cm of leg and folded its knees
    to the limit. The leg targets are shifted down (pelvis up axis) by the difference. Laterally, scaled human feet land
    almost under the body while the robot's wheels sit robot_half_track out and its hips adduct only 10 deg, so the
    targets are also shifted outward. Offsets are in human units (GMR then scales by s_leg); knees get half.
    """
    out = []
    for f in frames:
        g = dict(f)
        Rp = R.from_quat(f["pelvis"][1], scalar_first=True)
        up, lat = Rp.apply([0.0, 1.0, 0.0]), Rp.apply([1.0, 0.0, 0.0])  # SMPL: +y up, +x = body's left
        pel = np.asarray(f["pelvis"][0])
        hips = [np.asarray(f[f"{s}_hip"][0]) for s in ("left", "right")]
        human_drop = -np.mean([np.dot(h - pel, up) for h in hips])
        half_hip = 0.5 * np.linalg.norm(hips[0] - hips[1])
        dz = max(robot_hip_drop / s_leg - human_drop, 0.0)
        dy = max(robot_half_track / s_leg - half_hip, 0.0)
        lat_h = lat - np.dot(lat, [0.0, 0.0, 1.0]) * np.array([0.0, 0.0, 1.0])
        lat_h /= np.linalg.norm(lat_h) + 1e-9
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for k, frac in (("ankle", 1.0), ("foot", 1.0), ("knee", 0.5)):
                p, q = f[f"{side}_{k}"]
                g[f"{side}_{k}"] = (np.asarray(p) - dz * up + sign * frac * dy * lat_h, q)
        out.append(g)
    return out


def lowpass(x, win=9, order=3):
    return savgol_filter(x, win, order, axis=0) if len(x) > win else x


BAL_MIN_S = 0.5  # s: single-support phases shorter than this are not kept (GVHMR foot spikes last ~0.3-0.4 s)
BAL_HUMAN_ON, BAL_HUMAN_OFF = 0.05, 0.035  # m: human foot above the other foot -> lifted / back down (hysteresis; GVHMR feet wander 3-4 cm)
BAL_ROBOT_LIFT = 0.025  # m: a retargeted wheel bottom above this for >= BAL_MIN_S counts as lifted too (see robot_lift_phases)
BAL_RAMP_S = 0.4  # s: the CoM shift fades in before lift-off and out after touch-down
BAL_ITERS = 2  # plan -> IK passes (the second re-plans from the CoM the IK actually reached)
BAL_LIFT_MIN = 0.045  # m: lifted wheel bottom at least this high inside single-wheel phases (ground clearance)
BAL_LIFT_RAMP_S = 0.2  # s: the clearance fades in after lift-off and out before touch-down


def human_lift_phases(frames, fps):
    """Which human foot is off the ground, per frame: {"left": bool (N,), "right": bool (N,)}.

    A foot's height is its lower point (ankle or toe) above the other foot's lower point (GMR frames, z up). Lifted with
    hysteresis (above BAL_HUMAN_ON, down again below BAL_HUMAN_OFF) and only for at least BAL_MIN_S. Judging lifts from
    the retargeted wheels instead turned 2-4 cm of GVHMR foot noise and heel raises (toe still down) into lifts: the
    clearance pass then raised them to 4.5 cm, a support change the video does not show (roller40 18.7-21 s)."""
    from scipy.ndimage import median_filter

    low = np.array([[min(f[f"{s}_ankle"][0][2], f[f"{s}_foot"][0][2]) for s in ("left", "right")] for f in frames])
    clear = median_filter(low - low.min(1, keepdims=True), size=(5, 1), mode="nearest")
    out = {}
    for k, s in enumerate(("left", "right")):
        up, state = np.zeros(len(low), bool), False
        for i, h in enumerate(clear[:, k]):
            state = h > (BAL_HUMAN_OFF if state else BAL_HUMAN_ON)
            up[i] = state
        i = 0
        while i < len(up):
            if up[i]:
                j = i
                while j < len(up) and up[j]:
                    j += 1
                if j - i < BAL_MIN_S * fps:
                    up[i:j] = False
                i = j
            else:
                i += 1
        out[s] = up
    return out


def robot_lift_phases(bottom, human_lift, fps):
    """human_lift plus the wheels that the retargeted robot holds above BAL_ROBOT_LIFT for at least BAL_MIN_S.

    The human foot height (ankle / toe above the other foot) cannot tell a real low lift (three_turn right foot 7.9-10.1 s,
    2-5 cm, the video shows it lifted) from a heel raise (roller40 18-21 s, 2-5 cm, toe down): both read the same, so the
    thresholds of human_lift_phases stay as they are. After ground_unlifted_feet the robot wheel only stays up where the
    skater's foot really is, so such a wheel gets the same clearance and CoM balance as a clearly lifted one."""
    from scipy.ndimage import binary_closing

    out = {}
    for s in ("left", "right"):
        up = binary_closing(bottom[s] > BAL_ROBOT_LIFT, np.ones(2 * max(int(0.1 * fps), 1) + 1, bool))
        i = 0
        while i < len(up):
            if up[i]:
                j = i
                while j < len(up) and up[j]:
                    j += 1
                if j - i < BAL_MIN_S * fps:
                    up[i:j] = False
                i = j
            else:
                i += 1
        out[s] = human_lift[s] | up
    return out


def ground_unlifted_feet(frames, human_lift, fps):
    """Lower a foot that is not lifted (human_lift) to the other foot's height, on the IK targets (ankle, toe).

    GVHMR's feet wander 2-4 cm (and a heel raise lifts the ankle while the toe stays down); the wheel follows the ankle,
    so the robot showed lifts and support changes the video does not have. The offset is faded in / out over BAL_RAMP_S
    around the real lifts, so GMR's whole-body IK keeps the wheel on the ground (lowering a wheel afterwards can only
    straighten that knee, which sits near its limit, and made the leg jump)."""
    from scipy.ndimage import median_filter

    low = np.array([[min(f[f"{s}_ankle"][0][2], f[f"{s}_foot"][0][2]) for s in ("left", "right")] for f in frames])
    clear = median_filter(low - low.min(1, keepdims=True), size=(5, 1), mode="nearest")
    ramp = max(int(BAL_RAMP_S * fps), 1)
    kern = np.hanning(2 * ramp + 1)
    kern /= kern.sum()
    out = [dict(f) for f in frames]
    for k, s in enumerate(("left", "right")):
        keep = np.clip(np.convolve(human_lift[s].astype(float), kern, mode="same") * 2.0, 0.0, 1.0)  # 1 near real lifts
        keep = np.maximum(keep, human_lift[s])
        drop = (1.0 - keep) * clear[:, k]
        for i, g in enumerate(out):
            for j in ("ankle", "foot"):
                p, q = g[f"{s}_{j}"]
                g[f"{s}_{j}"] = (np.asarray(p) - np.array([0.0, 0.0, drop[i]]), q)
    return out


def support_phases(lifted, fps):
    """Per frame: supporting side (0 left, 1 right, -1 none) of the nearest single-support phase, and its blend weight.

    lifted: {"left"/"right": bool (N,)}. Inside a phase the weight is 1; within BAL_RAMP_S before / after it ramps
    linearly to 0, so the CoM moves over the support wheel before the other wheel leaves the ground and back after it
    lands."""
    N = len(lifted["left"])
    lifted = [lifted["left"], lifted["right"]]
    single = lifted[0] ^ lifted[1]
    side, w = np.full(N, -1), np.zeros(N)
    ramp = int(BAL_RAMP_S * fps)
    i = 0
    while i < N:
        if not single[i]:
            i += 1
            continue
        j = i
        while j < N and single[j] and lifted[0][j] == lifted[0][i]:
            j += 1
        if j - i >= BAL_MIN_S * fps:
            sup = 1 if lifted[0][i] else 0
            for k in range(max(i - ramp, 0), min(j + ramp, N)):
                wk = 1.0 if i <= k < j else 1.0 - (i - k if k < i else k - j + 1) / (ramp + 1)
                if wk > w[k]:
                    w[k], side[k] = wk, sup
        i = j
    return side, w


def zmp_com_plan(com, cp, side, w, lat, dt, g, w_zmp=10.0, w_keep=1.0, w_base=0.05, w_smooth=0.003):
    """Horizontal CoM path c (N, 2) whose ZMP c - (z/g) c'' lies on the support wheel during single support.

    Solved as one sparse least-squares problem over the whole clip (iterating "shift the CoM by the ZMP error" diverges:
    the c'' term amplifies anything faster than ~0.7 Hz). Residuals: lateral ZMP offset from the support contact
    (weight w_zmp * blend weight), departure from the original CoM (full weight outside the phases, longitudinal only
    inside them, plus a small w_base everywhere) and the acceleration of the change (w_smooth, m per m/s^2)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    N = len(com)
    I2 = sp.identity(2 * N, format="csr")
    D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(N - 2, N)) / dt**2
    D2 = sp.vstack((sp.csr_matrix((1, N)), D2, sp.csr_matrix((1, N))))
    D2xy = sp.kron(D2, sp.identity(2), format="csr")
    k = com[:, 2] / g
    zmp_op = I2 - sp.diags(np.repeat(k, 2)) @ D2xy  # (2N, 2N): c -> ZMP
    rows, rhs, wts = [], [], []
    # direction-projected rows: lat_i . (A c)_i = lat_i . b_i
    def proj(dirs, mask):
        P = sp.lil_matrix((N, 2 * N))
        for i in np.nonzero(mask)[0]:
            P[i, 2 * i : 2 * i + 2] = dirs[i, :2]
        return P.tocsr()

    act = (side >= 0) & (w > 0)
    lon = np.stack((-lat[:, 1], lat[:, 0], np.zeros(N)), 1)
    c0 = com[:, :2].reshape(-1)
    cp_sup = np.array([cp[i, side[i], :2] if side[i] >= 0 else com[i, :2] for i in range(N)])
    P_lat, P_lon = proj(lat, act), proj(lon, act)
    rows += [P_lat @ zmp_op, P_lon, I2, D2xy]
    rhs += [(lat[:, :2] * cp_sup).sum(1), P_lon @ c0, c0, D2xy @ c0]
    full = np.repeat(np.where(act, 0.0, w_keep) + w_base, 2)
    wts += [w_zmp * w, np.full(N, w_keep) * act, full, np.full(2 * N, w_smooth)]
    A = sp.vstack([sp.diags(wt) @ r for r, wt in zip(rows, wts)], format="csr")
    b = np.concatenate([wt * y for y, wt in zip(rhs, wts)])
    return spsolve((A.T @ A).tocsc(), A.T @ b).reshape(N, 2)


def lift_profile(side, w, fps):
    """0..1 per frame: inside single-wheel phases (w == 1), cosine ramps of BAL_LIFT_RAMP_S at both ends."""
    N, ramp = len(w), max(int(BAL_LIFT_RAMP_S * fps), 1)
    inside, prof, i = (side >= 0) & (w >= 1.0), np.zeros(len(w)), 0
    while i < N:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j < N and inside[j] and side[j] == side[i]:
            j += 1
        k = np.arange(i, j)
        edge = np.minimum(k - i, j - 1 - k) / ramp
        prof[i:j] = 0.5 - 0.5 * np.cos(np.pi * np.clip(edge, 0.0, 1.0))
        i = j
    return prof


def balance_single_support(m, d, root_p, root_q, dof, order, movable, fps, human_lift=None):
    """Make single-wheel phases dynamically feasible: move the CoM so the ZMP sits on the support wheel's contact.

    A crowned tyre gives a point contact, so with one wheel down the lateral ZMP must stay on that wheel (the robot
    has no ankle). Retargeted skating did not: the free leg was lifted by rolling the whole body ~20 deg, which put the
    ZMP 3-5 cm (p90 8-11 cm) beside the support wheel, and the trained policy kept both wheels down instead. Per frame,
    the lateral ZMP error (along the support wheel's axle) shifts the CoM target; IK (mink) then moves the root and the
    leg joints with the support wheel kept in place, the lifted wheel and the torso orientation held softly and the arms
    fixed. Repeated BAL_ITERS times since the shift changes the CoM acceleration. Edits root_p / root_q / dof in place.
    """
    import mink
    from scipy.ndimage import gaussian_filter1d

    N, dt, g = len(root_p), 1.0 / fps, 9.81
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in order]
    wb = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wheel_link") for s in ("left", "right")}
    wj = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_wheel_joint") for s in ("left", "right")}

    def set_q(i):
        d.qpos[:3], d.qpos[3:7] = root_p[i], root_q[i]
        d.qpos[qadr] = dof[i]

    def measure():
        com, cp, ax, bot = np.zeros((N, 3)), np.zeros((N, 2, 3)), np.zeros((N, 2, 3)), {"left": np.zeros(N), "right": np.zeros(N)}
        for i in range(N):
            set_q(i)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            com[i] = d.subtree_com[0]
            for k, s in enumerate(("left", "right")):
                a = d.xaxis[wj[s]]
                down = -(np.array([0.0, 0.0, 1.0]) - a[2] * a)
                down /= np.linalg.norm(down) + 1e-9
                cp[i, k] = d.xpos[wb[s]] + WHEEL_R * down
                ax[i, k] = a
                bot[s][i] = cp[i, k, 2]
        acc = gaussian_filter1d(com, 2.0, axis=0, order=2) / dt**2
        zmp = com[:, :2] - com[:, 2:3] * acc[:, :2] / g
        return com, cp, ax, bot, zmp

    def lateral_err(side, cp, ax, zmp):
        e, lat = np.zeros(N), np.zeros((N, 3))
        for i in np.nonzero(side >= 0)[0]:
            a = ax[i, side[i]].copy()
            a[2] = 0.0
            lat[i] = a / (np.linalg.norm(a) + 1e-9)
            e[i] = np.dot(zmp[i] - cp[i, side[i], :2], lat[i, :2])
        return e, lat

    com, cp, ax, bot, zmp = measure()
    if human_lift is None:  # fall back to the retargeted wheels themselves
        from scipy.ndimage import binary_closing

        human_lift = {s: binary_closing(bot[s] > GROUND_EPS, np.ones(2 * max(int(0.1 * fps), 1) + 1, bool)) for s in bot}
    side, w = support_phases(human_lift, fps)
    prof = lift_profile(side, w, fps)
    e0, _ = lateral_err(side, cp, ax, zmp)
    lifted = lambda b: np.array([b["right" if side[i] == 0 else "left"][i] if side[i] >= 0 else np.inf for i in range(N)])  # noqa: E731
    clear0 = lifted(bot)
    if not (side >= 0).any():
        return {"single_support_s": 0.0}

    cfg = mink.Configuration(m)
    com_task = mink.ComTask(cost=[100.0, 100.0, 1.0])
    wheel_tasks = {s: mink.FrameTask(f"{s}_wheel_link", "body", position_cost=1.0, orientation_cost=1.0) for s in wb}
    torso = mink.FrameTask("link_torso", "body", position_cost=0.0, orientation_cost=2.0)
    posture = mink.PostureTask(m, cost=0.05)
    frozen = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in order if n not in movable or "wheel" in n]
    constraints = [mink.DofFreezingTask(m, dof_indices=frozen)]
    limits = [mink.ConfigurationLimit(m)]
    # 1) clearance: inside a single-wheel phase the lifted wheel goes to at least BAL_LIFT_MIN (a 2-4 cm lift left no
    # margin: the policy tapped the wheel down to rebalance). IK on that leg's hip pitch / roll and calf only (3 joints
    # for a 3-D target: one smooth solution per frame); the whole-body IK below then holds it while it moves the CoM.
    # (Wheels that should stay down are kept down earlier, on the human targets: ground_unlifted_feet.)
    dz = {}
    for k, s in enumerate(("left", "right")):
        p_up = np.where((side >= 0) & (side != k), prof, 0.0)  # this wheel is the lifted one
        dz[s] = np.clip(BAL_LIFT_MIN - bot[s], 0.0, None) * p_up
    leg_tasks = {s: mink.FrameTask(f"{s}_wheel_link", "body", position_cost=[5.0, 5.0, 50.0], orientation_cost=0.0) for s in wb}  # height first
    leg_post = mink.PostureTask(m, cost=0.01)
    for s in ("left", "right"):
        free = {f"{s}_hip_pitch_joint", f"{s}_hip_roll_joint", f"{s}_calf_joint"}
        leg_frozen = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in order if n not in free] + list(range(6))
        for i in np.nonzero(dz[s] > 1e-4)[0]:
            set_q(i)
            cfg.update(d.qpos.copy())
            T = cfg.get_transform_frame_to_world(f"{s}_wheel_link", "body")
            leg_tasks[s].set_target(mink.SE3.from_rotation_and_translation(T.rotation(), T.translation() + np.array([0.0, 0.0, dz[s][i]])))
            leg_post.set_target(cfg.q.copy())
            for _ in range(30):
                v = mink.solve_ik(cfg, [leg_tasks[s], leg_post], 0.1, "daqp", 1e-3, limits=limits, constraints=[mink.DofFreezingTask(m, dof_indices=leg_frozen)])
                cfg.integrate_inplace(v, 0.1)
            dof[i] = cfg.q[qadr]
    com, cp, ax, bot, zmp = measure()
    # 2) CoM over the support wheel. The support wheel is held in place (position cost 50), the other follows the CoM
    # task softly (2). Switching between the two costs at the frame where the support changes let the wheel that had been
    # dragged a few cm snap back in one frame (roller_house: right wheel 2 m/s for one frame), so the hold fades over ~0.3 s.
    sup_hold = gaussian_filter1d(np.stack([(side == 0), (side == 1)], 1).astype(float), 0.1 * fps, axis=0)
    for _ in range(BAL_ITERS):
        _, lat = lateral_err(side, cp, ax, zmp)
        target = com.copy()
        target[:, :2] = zmp_com_plan(com, cp, side, w, lat, dt, g)
        for i in np.nonzero(np.linalg.norm(target[:, :2] - com[:, :2], axis=1) > 1e-4)[0]:
            set_q(i)
            cfg.update(d.qpos.copy())
            com_task.set_target(target[i])
            for k, s in enumerate(("left", "right")):
                T = cfg.get_transform_frame_to_world(f"{s}_wheel_link", "body")
                wheel_tasks[s].set_target(T)
                wheel_tasks[s].set_position_cost(2.0 + 48.0 * sup_hold[i, k])
            torso.set_target(cfg.get_transform_frame_to_world("link_torso", "body"))
            posture.set_target(cfg.q.copy())
            for _ in range(8):
                v = mink.solve_ik(cfg, [com_task, *wheel_tasks.values(), torso, posture], 0.1, "daqp", 1e-3, limits=limits, constraints=constraints)
                cfg.integrate_inplace(v, 0.1)
            q = cfg.q
            root_p[i], root_q[i], dof[i] = q[:3], q[3:7] / np.linalg.norm(q[3:7]), q[qadr]
        com, cp, ax, bot, zmp = measure()
        ik_err = np.linalg.norm(com[:, :2] - target[:, :2], axis=1)
    e1, _ = lateral_err(side, cp, ax, zmp)
    sel = (side >= 0) & (w >= 1.0)
    pct = lambda x: [round(float(np.median(np.abs(x[sel]))) * 100, 2), round(float(np.percentile(np.abs(x[sel]), 90)) * 100, 2)]  # noqa: E731
    return {"single_support_s": float(sel.sum() * dt), "zmp_lateral_cm_median_p90_before": pct(e0), "zmp_lateral_cm_median_p90_after": pct(e1),
            "com_ik_err_cm_p95": round(float(np.percentile(ik_err, 95)) * 100, 2),
            "lifted_wheel_cm_p10_median_before": [round(float(np.percentile(clear0[prof > 0.99], q)) * 100, 1) for q in (10, 50)],
            "lifted_wheel_cm_p10_median_after": [round(float(np.percentile(lifted(bot)[prof > 0.99], q)) * 100, 1) for q in (10, 50)]}


def postprocess(qpos_ik, ik_names, fps, variant, balance=True, human_lift=None):
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
    def limit_leg_rates():
        for c, n in enumerate(order):  # hardware joint speed limits (URDF): legs 10 rad/s, arms >= 23.9 rad/s
            if n in ik_names and ("hip" in n or "calf" in n):
                dof[:, c] = rate_limit(dof[:, c], (HIP_YAW_RATE_MAX if "hip_yaw" in n else 10.0) / fps)

    limit_leg_rates()
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

    def ground():
        """Per-frame lowest wheel bottom, smoothed, subtracted from the root height; nothing below the ground."""
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
        root_p[:, 2] += pen
        for s in wheel_j:
            bottom[s] += pen
            centres[s][:, 2] += pen
        return centres, axles, bottom

    # 1) ground alignment; 2) single-wheel balance (then the ground again: the legs changed)
    centres, axles, bottom = ground()
    if balance and human_lift is not None:
        merged = robot_lift_phases(bottom, human_lift, fps)
        added = {s: round(float((merged[s] & ~human_lift[s]).sum() / fps), 2) for s in merged}
        human_lift = merged
    stats = {"joint_limit_violation_frac_before_clip": viol_frac}
    if balance and human_lift is not None:
        stats["lift_added_by_robot_wheels_s"] = added
    if balance:
        stats["balance"] = balance_single_support(m, d, root_p, root_q, dof, order, {n for n in ik_names if "hip" in n or "calf" in n}, fps, human_lift)
        # the balance IK solves every frame on its own and can flip the leg between frames (three_turn: right hip yaw 4.9 ->
        # 8.0 rad/s, right calf 3.2 -> 9.1 rad/s), so the speed limits are applied again
        limit_leg_rates()
        dof[:, limited] = np.clip(dof[:, limited], lim[limited, 0], lim[limited, 1])
        centres, axles, bottom = ground()
    # 3) wheels: no-slip rolling, omega = (v . d) / r while in contact
    dt = 1.0 / fps
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


def retarget_frames(rt, frames, s_leg):
    """Run GMR frame by frame; the heading targets only steer.

    The heading marker is fixed in the calf frame, so it moves up and down when the shank pitches, while a wheel's
    heading does not depend on that pitch. With the target at ankle height the IK kept the shank vertical (and the
    knee straight). Each frame the heading target is therefore lifted / lowered by the marker's height relative to
    the wheel centre in the previous solution (human units = robot offset / s_leg)."""
    m, d = rt.model, rt.configuration.data
    bid = lambda b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731
    ids = {s: (bid(f"{s}_wheel_center"), bid(f"{s}_wheel_heading")) for s in ("left", "right")}
    dz = {"left": 0.0, "right": 0.0}
    out = []
    for f in frames:
        g = dict(f)
        for s in ("left", "right"):
            p, q = f[f"{s}_foot"]
            g[f"{s}_foot"] = (np.asarray(p) + np.array([0.0, 0.0, dz[s] / s_leg]), q)
        out.append(rt.retarget(g))
        for s, (c, h) in ids.items():
            dz[s] = d.xpos[h][2] - d.xpos[c][2]
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvhmr_pred_file", required=True)
    ap.add_argument("--variant", nargs="+", default=["full", "hw"], choices=["full", "hw"])
    ap.add_argument("--out_dir", default=os.path.join(RT, "data", "retarget_out"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--raw_heading", action="store_true", help="use the estimated toe point as wheel heading (no velocity blend)")
    ap.add_argument("--leg_reach", type=float, default=0.95, help="robot leg used per straight human leg (<1 keeps the knees off their straight limit)")
    ap.add_argument("--config", default=CONFIG, help="GMR IK config (task weights)")
    ap.add_argument("--no_balance", action="store_true", help="keep single-wheel phases as retargeted (ZMP may lie beside the support wheel)")
    a = ap.parse_args()
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.gvhmr_pred_file)))
    os.makedirs(a.out_dir, exist_ok=True)
    smplx_data, body_model, smplx_output, _ = load_gvhmr_pred_file(a.gvhmr_pred_file, os.path.join(GMR_ROOT, "assets", "body_models"))
    frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    h_leg, h_arm = human_lengths(frames)
    r_leg, r_arm = robot_lengths()
    s_leg, s_arm = a.leg_reach * r_leg / h_leg, r_arm / h_arm
    human_lift = human_lift_phases(frames, fps)  # from the raw human feet, before the leg targets are moved
    frames = ground_unlifted_feet(frames, human_lift, fps)
    frames = adapt_legs(frames, s_leg, ROBOT_HALF_TRACK, ROBOT_HIP_DROP)
    if not a.raw_heading:
        frames = velocity_heading(frames, fps, ahead=HEADING_AHEAD / s_leg)
    sh_off, l_upper, l_fore = robot_arm_geometry()
    frames = rechain_arms(frames, sh_off, l_upper, l_fore)
    print(f"[{name}] {len(frames)} frames @ {fps} fps; human leg {h_leg:.3f} arm {h_arm:.3f} -> scale leg/torso {s_leg:.3f}; "
          f"arms re-chained with robot segments {l_upper:.3f} / {l_fore:.3f} m")
    for variant in a.variant:
        robot = register(variant, a.config)
        rt = GMR(src_human="smplx", tgt_robot=robot, actual_human_height=None, verbose=False)
        for k in LEG_KEYS:
            rt.human_scale_table[k] = s_leg
        for k in ARM_KEYS:
            rt.human_scale_table[k] = 1.0
        ik_names = [mujoco.mj_id2name(rt.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(rt.model.njnt) if rt.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        # warm start from nemo_v3's HOME crouch: from GMR's default straight-leg zero pose the knee sits in the IK
        # singularity (no vertical gradient for the calf) and stayed straight, leaving the root ~7 cm above its target
        sys.path.insert(0, HERE)
        from make_retarget_mjcf import HOME_JOINT_POS

        q0 = rt.configuration.q.copy()
        for n_, v_ in HOME_JOINT_POS.items():
            j_ = mujoco.mj_name2id(rt.model, mujoco.mjtObj.mjOBJ_JOINT, n_)
            if j_ >= 0:
                q0[rt.model.jnt_qposadr[j_]] = v_
        rt.configuration.update(q0)
        # keep the knees off the straight-leg singularity during IK (the exported motion keeps the real 0 rad limit):
        # once a frame straightened a knee, the calf had no gradient left and never bent again
        import mink

        for side in ("left", "right"):
            j_ = mujoco.mj_name2id(rt.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_calf_joint")
            rt.model.jnt_range[j_, 0] = max(rt.model.jnt_range[j_, 0], KNEE_MIN_IK)
        # elbows: same branch on both arms. Both elbow axes are +y, and the natural bend is negative (right arm, and
        # the mirror sign of the elbow is +1). Unrestricted, the left arm slid into the equivalent "shoulder yaw +180 deg,
        # elbow bent backwards" solution and stayed there (shoulder yaw at its +2.79 limit, elbow at +1.57).
        for elbow in ("joint_lElbow", "joint_rElbow"):  # (not `name`: that is the clip name used for the outputs)
            j_ = mujoco.mj_name2id(rt.model, mujoco.mjtObj.mjOBJ_JOINT, elbow)
            rt.model.jnt_range[j_, 1] = min(rt.model.jnt_range[j_, 1], ELBOW_MAX_IK)
        rt.ik_limits = [mink.ConfigurationLimit(rt.model)]
        q0 = rt.configuration.q.copy()
        rt.configuration.update(q0)
        qpos = retarget_frames(rt, frames, s_leg)
        root_p, root_q, dof, order, bodies, lbp, stats = postprocess(qpos, ik_names, fps, variant, balance=not a.no_balance, human_lift=human_lift)
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
