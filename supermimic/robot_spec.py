"""Robot selection: every robot-specific constant SuperMimic needs, behind one spec.

The robot is chosen with the environment variable ``SUPERMIMIC_ROBOT`` (default ``g1``, the LAFAN1 reproduction):
    SUPERMIMIC_ROBOT=nemov3 python supermimic/train.py ...
It must be set before ``supermimic.mdp`` is imported, because the state / observation sizes are module constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]  # simulator / MJCF order = motion CSV column order
    tracked_bodies: tuple[str, ...]  # reward / critic bodies; the first is the root
    mjcf_path: str  # for the torch FK and the viewers
    default_q: tuple[float, ...]
    action_scale: tuple[float, ...]  # position joints: rad per action unit; velocity joints: rad/s per unit
    articulation_cfg: Callable  # () -> isaaclab ArticulationCfg (imported lazily)
    motion_dir: str
    com_body: str  # body whose CoM is randomized (xi[-3:])
    com_range: tuple[float, float, float] = (0.025, 0.05, 0.05)
    velocity_joints: tuple[str, ...] = ()  # velocity-controlled (wheels): no position target, angle not observed
    term_height: float = 0.25
    length_scale: float = 1.0  # robot size relative to G1; scales the position reward widths
    extra: dict = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def vel_mask(self) -> tuple[bool, ...]:
        return tuple(n in self.velocity_joints for n in self.joint_names)


def _g1() -> RobotSpec:
    from . import robot as g1

    def cfg():
        from .robot import G1_CFG

        return G1_CFG

    return RobotSpec(
        name="g1",
        joint_names=tuple(g1.G1_JOINT_NAMES),
        tracked_bodies=tuple(g1.TRACKED_BODIES),
        mjcf_path=g1.MJCF_PATH,
        default_q=tuple(g1.default_joint_pos_list()),
        action_scale=tuple(g1.action_scale(n) for n in g1.G1_JOINT_NAMES),
        articulation_cfg=cfg,
        motion_dir=os.path.join(REPO, "data", "lafan1_g1"),
        com_body="torso_link",
    )


def _nemov3() -> RobotSpec:
    from .robots import nemov3

    return nemov3.spec()


_SPECS = {"g1": _g1, "nemov3": _nemov3}


@lru_cache(maxsize=None)
def get_spec(name: str | None = None) -> RobotSpec:
    name = name or os.environ.get("SUPERMIMIC_ROBOT", "g1")
    if name not in _SPECS:
        raise ValueError(f"unknown SUPERMIMIC_ROBOT={name!r}, choose from {sorted(_SPECS)}")
    return _SPECS[name]()
