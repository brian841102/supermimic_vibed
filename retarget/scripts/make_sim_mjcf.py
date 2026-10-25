"""Simulation MJCF of nemo_v3 for SuperMimic (Isaac Lab / Newton / MuJoCo-Warp), from booster_mjlab's nemov3.xml.

sim_hw: the current hardware build - head (3), waist (3), waist push rods (4) and wrist pitch (2) welded at their HOME
angles (same as booster_mjlab's NEMOV3_FIXED_HEAD_WAIST_JOINTS), wheels kept as joints: 20 joints (arms 10, legs 8,
wheels 2). The MJCF's own <actuator> block is removed: Isaac Lab drives the joints with its actuator models
(supermimic/robots/nemov3.py), and keeping the XML actuators would apply a second PD on top.
The wheels keep their mesh colliders (as booster_mjlab and the PhysX flamingo setup do); convert with a raised hull
vertex limit, or Newton reduces each wheel to a lopsided 64-vertex polygon that bumps as it rolls (MILESTONE 19.1):
Usage: python retarget/scripts/make_sim_mjcf.py   then (in the container)
       python tools/mjcf_to_usd.py <xml> <usd parent dir> --hull_vertices '.*_wheel_collision=1024'
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_retarget_mjcf import ASSETS, HEAD, HOME_JOINT_POS, PUSH_RODS, WAIST, WRIST_PITCH, weld_joint  # noqa: E402

SIM_VARIANTS = {"sim_hw": HEAD + WAIST + PUSH_RODS + WRIST_PITCH}
# Elbows bend with negative angles; the URDF's +1.57 lets them fold backwards, which a policy did in the HOME pose
# (+7-12 deg, tracking hardly sees it: the forearm is 13 cm). +0.1 keeps HOME's 0 inside the 0.9 soft-limit band.
ELBOW_RANGE = (-1.57, 0.1)


def build(src: str, variant: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(src)
    for name in SIM_VARIANTS[variant]:
        weld_joint(spec, name, HOME_JOINT_POS.get(name, 0.0))
    for a in list(spec.actuators):
        spec.delete(a)
    for name in ("joint_lElbow", "joint_rElbow"):
        spec.joint(name).range = list(ELBOW_RANGE)
    spec.modelname = f"nemov3_{variant}"
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ASSETS, "booster_mjlab_nemov3", "xmls", "nemov3.xml"))
    ap.add_argument("--out_dir", default=os.path.join(ASSETS, "sim"))
    ap.add_argument("--variant", nargs="+", default=["sim_hw"], choices=sorted(SIM_VARIANTS))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    src_dir = os.path.dirname(os.path.abspath(a.src))
    for variant in a.variant:
        spec = build(a.src, variant)
        out = os.path.join(a.out_dir, f"nemov3_{variant}.xml")
        meshdir = os.path.normpath(os.path.join(os.path.relpath(src_dir, os.path.abspath(a.out_dir)), spec.meshdir or ""))
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshdir}/"', spec.to_xml(), count=1)
        with open(out, "w") as f:
            f.write(xml)
        m = mujoco.MjModel.from_xml_path(out)
        joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        print(f"{variant}: {out}  joints {len(joints)}  actuators {m.nu}  mass {m.body_mass.sum():.2f} kg")
        print("  joint order:", joints)


if __name__ == "__main__":
    main()
