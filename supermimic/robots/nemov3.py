"""nemo_v3 wheel-legged humanoid, current hardware build (head, waist, push rods and wrist pitch welded): 20 joints.

Model: retarget/assets/nemo_v3/sim/nemov3_sim_hw.xml (retarget/scripts/make_sim_mjcf.py) and its USD
(tools/mjcf_to_usd.py). Gains follow the hardware (booster_mjlab nemov3_constants.py, NEMOV3_DEPLOY_PD_GAINS): legs
kp 50 / kd 0.5, shoulder pitch/roll 18 / 0.8, shoulder yaw / elbow / wrist roll 1.5 / 0.1; effort and velocity limits
from train_v2.urdf. Wheels are velocity-controlled (damping-only implicit drive, kv 0.5, 6 Nm).
"""

from __future__ import annotations

import os
import re

from ..robot_spec import REPO, RobotSpec

ASSETS = os.path.join(REPO, "retarget", "assets", "nemo_v3", "sim")
MJCF_PATH = os.path.join(ASSETS, "nemov3_sim_hw.xml")
USD_PATH = os.path.join(ASSETS, "usd", "nemov3_sim_hw", "nemov3_sim_hw.usda")

# Isaac Lab's articulation order for the USD (= motion CSV column order); differs from the MJCF declaration order.
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_calf_joint", "left_wheel_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_calf_joint", "right_wheel_joint",
    "joint_lShoulderPitch", "joint_lShoulderRoll", "joint_lShoulderYaw", "joint_lElbow", "joint_lWristRoll",
    "joint_rShoulderPitch", "joint_rShoulderRoll", "joint_rShoulderYaw", "joint_rElbow", "joint_rWristRoll",
)  # fmt: skip
WHEELS = ("right_wheel_joint", "left_wheel_joint")

# Wheel links spin, so their orientation is not tracked; the calf links carry the wheel's heading and camber.
TRACKED_BODIES = (
    "link_leg2pelvis", "left_hip_roll_link", "left_calf_link", "right_hip_roll_link", "right_calf_link",
    "link_lShoulderRoll", "link_lElbow", "link_lWristRoll", "link_rShoulderRoll", "link_rElbow", "link_rWristRoll",
)  # fmt: skip

# booster_mjlab HOME_KEYFRAME (crouched stance, arms down)
DEFAULT_Q = {
    "left_hip_pitch_joint": -0.26, "right_hip_pitch_joint": -0.26, "left_calf_joint": 0.43, "right_calf_joint": 0.43,
    "left_hip_roll_joint": 0.08, "right_hip_roll_joint": 0.08, "left_hip_yaw_joint": -0.09, "right_hip_yaw_joint": -0.06,
    "joint_lShoulderRoll": 1.22, "joint_rShoulderRoll": 1.22,
}  # fmt: skip
INIT_HEIGHT = 0.50
# wheel centres in their calf frames (wheel link pos in nemov3_sim_hw.xml)
WHEEL_CENTRES = (("left_calf_link", (0.0, -0.043, -0.175)), ("right_calf_link", (0.0, 0.043, -0.175)))

# (regex, kp, kd, effort, velocity limit)
ARMATURE = 0.01
GAINS = (
    (r".*_hip_(pitch|roll|yaw)_joint", 50.0, 0.5, 18.0, 10.0),
    (r".*_calf_joint", 50.0, 0.5, 20.0, 10.0),
    (r"joint_[rl]Shoulder(Pitch|Roll)", 18.0, 0.8, 5.6, 23.87),
    (r"joint_[rl](ShoulderYaw|Elbow|WristRoll)", 1.5, 0.1, 1.27, 40.2),
    (r".*_wheel_joint", 0.0, 0.5, 6.0, 25.0),
)
POS_ACTION_SCALE = 0.5  # rad per action unit (G1's BeyondMimic scales are ~0.3-0.55; 0.25*effort/kp would be 0.08-0.2)
WHEEL_ACTION_SCALE = 10.0  # rad/s per action unit (0.75 m/s at r = 0.075 m)


def _gains(name: str):
    for pat, kp, kd, eff, vel in GAINS:
        if re.fullmatch(pat, name):
            return kp, kd, eff, vel
    raise KeyError(name)


def articulation_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    acts = {}
    for n in JOINT_NAMES:
        kp, kd, eff, vel = _gains(n)
        acts[n] = ImplicitActuatorCfg(joint_names_expr=[n], effort_limit_sim=eff, velocity_limit_sim=vel, stiffness=kp, damping=kd, armature=ARMATURE)
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            variants={"Physics": "mujoco"},
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, INIT_HEIGHT), joint_pos=dict(DEFAULT_Q), joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        actuators=acts,
    )


def spec() -> RobotSpec:
    return RobotSpec(
        name="nemov3",
        joint_names=JOINT_NAMES,
        tracked_bodies=TRACKED_BODIES,
        mjcf_path=MJCF_PATH,
        default_q=tuple(DEFAULT_Q.get(n, 0.0) for n in JOINT_NAMES),
        action_scale=tuple(WHEEL_ACTION_SCALE if n in WHEELS else POS_ACTION_SCALE for n in JOINT_NAMES),
        articulation_cfg=articulation_cfg,
        motion_dir=os.path.join(REPO, "retarget", "data", "supermimic_motions", "nemov3_multi"),  # roller32 + roller_house
        com_body="link_torso",
        com_range=(0.0125, 0.025, 0.025),
        velocity_joints=WHEELS,
        term_height=0.15,
        length_scale=0.5,
        lift_points=WHEEL_CENTRES,
    )
