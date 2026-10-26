"""Count foot taps (nemo_v3) in recorded trajectories (evaluate.py --record / ckpt_*_traj.npz), per clip.

A tap: during a reference single-wheel phase (reference wheel > 2 cm for >= 0.3 s, first/last 10 frames left out as
lift-off / touch-down), the policy's wheel drops below 1 cm, i.e. touches the ground again.
Usage: python tools/count_taps.py logs/<run>/ckpt_*_traj.npz     (needs mujoco + scipy, SUPERMIMIC_ROBOT=nemov3)
"""

import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
os.environ.setdefault("SUPERMIMIC_ROBOT", "nemov3")
from supermimic.robot_spec import get_spec  # noqa: E402

DT, REF_UP, TOUCH, MIN_S, EDGE, WHEEL_R = 0.02, 0.02, 0.01, 0.3, 10, 0.075
spec = get_spec()
model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
data = mujoco.MjData(model)
adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in spec.joint_names]
bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wheel_link") for s in ("left", "right")]


def wheel_heights(p, q, joints):
    out = np.empty((len(p), 2))
    for i in range(len(p)):
        data.qpos[:3], data.qpos[3:7], data.qpos[adr] = p[i], q[i], joints[i]
        mujoco.mj_kinematics(model, data)
        out[i] = data.xpos[bodies, 2] - WHEEL_R
    return out


def segments(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def count(d, c):
    """-> (reference single-wheel time [s], number of taps, tap time [s]) of clip c."""
    T = int(d["clip_len"][c]) - 1
    f = d["f"][:T, c].astype(int)
    hr = wheel_heights(d["ref_root_p"][f], R.from_matrix(d["ref_root_R"][f]).as_quat()[:, [3, 0, 1, 2]], d["ref_q"][f])
    hp = wheel_heights(d["root_p"][:T, c], d["root_quat"][:T, c], d["q"][:T, c])
    ref_t = taps = tap_t = 0
    for k in range(2):
        for a, b in segments(hr[:, k] > REF_UP):
            if (b - a) * DT < MIN_S:
                continue
            ref_t += (b - a) * DT
            for x, y in segments(hp[a + EDGE : b - EDGE, k] < TOUCH):
                taps += 1
                tap_t += (y - x) * DT
    return ref_t, taps, tap_t


if __name__ == "__main__":
    for path in sys.argv[1:]:
        d = dict(np.load(path))
        parts = []
        for c, name in enumerate(d["names"]):
            ref_t, taps, tap_t = count(d, c)
            parts.append(f"{name}: {taps} taps ({tap_t:.2f}s of {ref_t:.1f}s single-wheel)")
        print(os.path.basename(path), " | ".join(parts))
