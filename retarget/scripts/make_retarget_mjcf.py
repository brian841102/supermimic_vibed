"""Build the retargeting-only MJCFs of nemo_v3 from booster_mjlab's nemov3.xml (plan section 5.1).

Two variants share the same body names, so one GMR IK config serves both:
  full: welds head (3), waist push rods (4) and wheels (2)          -> 23 IK joints (waist + wrist pitch free)
  hw:   additionally welds waist (3) and wrist pitch (2), i.e. the  -> 18 IK joints (current hardware)
        robot built with booster_mjlab's NEMOV3_FIXED_HEAD_WAIST_JOINTS

Both add massless marker bodies used as IK targets: wheel centre and a heading point 0.12 m ahead of it along
the rolling direction (children of the calf links, which do not spin with the wheels), and a hand point 0.039 m
below each wrist-pitch link (robot_retarget.urdf's *_hand_base_link).

Usage: python retarget/scripts/make_retarget_mjcf.py [--src .../nemov3.xml] [--out_dir ...] [--variant full hw]
"""

from __future__ import annotations

import argparse
import os
import re

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets", "nemo_v3"))

# booster_mjlab robots/nemov3/nemov3_constants.py: HOME_KEYFRAME.joint_pos (joints not listed are 0) and
# NEMOV3_FIXED_HEAD_WAIST_JOINTS. Welded joints are baked in at these angles.
HOME_JOINT_POS = {
    "left_hip_pitch_joint": -0.26, "right_hip_pitch_joint": -0.26, "left_calf_joint": 0.43, "right_calf_joint": 0.43,
    "left_hip_roll_joint": 0.08, "right_hip_roll_joint": 0.08, "left_hip_yaw_joint": -0.09, "right_hip_yaw_joint": -0.06,
    "joint_pelvisPitch": 0.01, "joint_lShoulderRoll": 1.22, "joint_rShoulderRoll": 1.22,
}  # fmt: skip
HEAD = ("head_roll_joint", "head_pitch_joint", "head_yaw_joint")
PUSH_RODS = ("joint_rLinearOut", "joint_rLinear", "joint_lLinearOut", "joint_lLinear")
WAIST = ("joint_pelvisYaw", "joint_pelvisPitch", "joint_pelvisRoll")
WRIST_PITCH = ("joint_rWristPitch", "joint_lWristPitch")
WHEELS = ("left_wheel_joint", "right_wheel_joint")
VARIANTS = {
    "full": HEAD + PUSH_RODS + WHEELS,
    "hw": HEAD + WAIST + PUSH_RODS + WRIST_PITCH + WHEELS,
}
HEADING_AHEAD = 0.12  # m, heading point ahead of the wheel centre along the rolling direction
HAND_OFFSET = (0.0, 0.0, -0.039)  # robot_retarget.urdf: Lhand2arm / Rhand2arm origin below link_[lr]WristPitch
MARKER_RGBA = {"wheel_center": (0.9, 0.2, 0.2, 1), "wheel_heading": (0.2, 0.8, 0.2, 1), "hand_point": (0.2, 0.4, 1.0, 1)}


def weld_joint(spec: mujoco.MjSpec, name: str, q: float) -> None:
    """Remove joint ``name`` and its actuators, baking ``q`` into the child body's frame.

    Same as ``_weld_joint`` in booster_mjlab's nemov3_constants.py, so welded bodies sit exactly where the
    training model puts them.
    """
    joint = spec.joint(name)
    body = joint.parent
    if q != 0.0:
        axis = np.asarray(joint.axis, dtype=np.float64) / np.linalg.norm(joint.axis)
        body_quat = np.asarray(body.quat, dtype=np.float64)
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, q)
            anchor = np.asarray(joint.pos, dtype=np.float64)
            rotated = np.zeros(3)
            mujoco.mju_rotVecQuat(rotated, anchor, dq)
            local_shift = anchor - rotated
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, body_quat, dq)
            body.quat = new_quat
        elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            local_shift = axis * q
        else:
            raise ValueError(f"cannot weld joint '{name}' of type {joint.type}")
        shift = np.zeros(3)
        mujoco.mju_rotVecQuat(shift, local_shift, body_quat)
        body.pos = np.asarray(body.pos, dtype=np.float64) + shift
    for actuator in [a for a in spec.actuators if a.target == name]:
        spec.delete(actuator)
    spec.delete(joint)


def add_marker(parent: mujoco.MjsBody, name: str, pos, kind: str) -> None:
    """Massless IK target body with a small non-colliding sphere (geom group 2) so it shows in viewers."""
    body = parent.add_body(name=name, pos=list(map(float, pos)))
    body.add_geom(name=f"{name}_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], rgba=MARKER_RGBA[kind],
                  contype=0, conaffinity=0, group=2, density=0.0)  # fmt: skip


def build(src: str, variant: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(src)
    # wheel geometry in the (unwelded) calf frames, taken before any welding
    wheel = {}
    for side in ("left", "right"):
        wb = spec.body(f"{side}_wheel_link")
        wj = spec.joint(f"{side}_wheel_joint")
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(wb.quat, dtype=np.float64))
        axle = R.reshape(3, 3) @ np.asarray(wj.axis, dtype=np.float64)
        roll_dir = np.cross(axle, (0.0, 0.0, 1.0))
        roll_dir /= np.linalg.norm(roll_dir)
        if roll_dir[0] < 0:  # forward = +x of the robot (calf frames are axis-aligned at the zero pose)
            roll_dir = -roll_dir
        wheel[side] = (np.asarray(wb.pos, dtype=np.float64) + np.asarray(wj.pos, dtype=np.float64), roll_dir)
    for name in VARIANTS[variant]:
        weld_joint(spec, name, HOME_JOINT_POS.get(name, 0.0))
    for side in ("left", "right"):
        centre, roll_dir = wheel[side]
        calf = spec.body(f"{side}_calf_link")
        add_marker(calf, f"{side}_wheel_center", centre, "wheel_center")
        add_marker(calf, f"{side}_wheel_heading", centre + HEADING_AHEAD * roll_dir, "wheel_heading")
    for side, link in (("left", "link_lWristPitch"), ("right", "link_rWristPitch")):
        add_marker(spec.body(link), f"{side}_hand_point", HAND_OFFSET, "hand_point")
    spec.modelname = f"nemov3_retarget_{variant}"
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ASSETS, "booster_mjlab_nemov3", "xmls", "nemov3.xml"))
    ap.add_argument("--out_dir", default=os.path.join(ASSETS, "retarget_mjcf"))
    ap.add_argument("--variant", nargs="+", default=["full", "hw"], choices=sorted(VARIANTS))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    # to_xml() resolves meshes against the source xml's directory, so export with the original meshdir and only
    # then point meshdir (relative) back at the source directory from the output directory
    src_dir = os.path.dirname(os.path.abspath(a.src))
    for variant in a.variant:
        spec = build(a.src, variant)
        out = os.path.join(a.out_dir, f"nemov3_retarget_{variant}.xml")
        meshdir = os.path.normpath(os.path.join(os.path.relpath(src_dir, os.path.abspath(a.out_dir)), spec.meshdir or ""))
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshdir}/"', spec.to_xml(), count=1)
        with open(out, "w") as f:
            f.write(xml)
        m = mujoco.MjModel.from_xml_path(out)  # reload from disk: checks the mesh paths
        ik = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        print(f"{variant}: {out}  IK joints {len(ik)}  actuators {m.nu}  bodies {m.nbody}")


if __name__ == "__main__":
    main()
