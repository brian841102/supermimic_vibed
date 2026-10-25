"""Interactive browser playback of recorded evaluation trajectories (supermimic/evaluate.py --record) with viser.

The simulated robot is drawn solid; the reference motion is a translucent blue ghost (as in tools/render.py).
Open http://localhost:8080 in a browser after starting it; drag to orbit, scroll to zoom.
Usage: python tools/viser_play.py traj.npz [--port 8080] [--eval logs/<run>/ckpt_014000_eval.json]
"""

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np
import viser

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from supermimic.robot_spec import get_spec  # noqa: E402  (SUPERMIMIC_ROBOT=g1|nemov3, or --robot)
DT = 0.02  # 50 Hz control


def visual_geoms(model: mujoco.MjModel) -> list[int]:
    """Mesh geoms to draw: the URDF visuals (non-colliding) if present, else every mesh geom."""
    mesh = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    vis = [g for g in mesh if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return vis or mesh


def mesh_arrays(model: mujoco.MjModel, mesh_id: int):
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    return model.mesh_vert[va : va + vn].copy(), model.mesh_face[fa : fa + fn].copy()


class Robot:
    """One robot in the viser scene: a mesh handle per visual geom, posed from qpos via MuJoCo FK."""

    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel, name: str, color, opacity: float | None):
        self.model, self.data = model, mujoco.MjData(model)
        self.geoms = visual_geoms(model)
        self.handles = []
        for g in self.geoms:
            v, f = mesh_arrays(model, model.geom_dataid[g])
            self.handles.append(
                server.scene.add_mesh_simple(f"/{name}/geom_{g}", vertices=v, faces=f, color=color, opacity=opacity)
            )

    def set_qpos(self, qpos: np.ndarray):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        q = np.empty(4)
        for g, h in zip(self.geoms, self.handles):
            mujoco.mju_mat2Quat(q, self.data.geom_xmat[g])
            h.position = self.data.geom_xpos[g].copy()
            h.wxyz = q.copy()

    def set_visible(self, flag: bool):
        for h in self.handles:
            h.visible = flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--eval", type=str, default=None, help="optional ckpt_*_eval.json to show per-clip normalized return")
    ap.add_argument("--robot", type=str, default=None, help="g1 | nemov3 (default: $SUPERMIMIC_ROBOT or g1)")
    a = ap.parse_args()
    if a.robot:
        os.environ["SUPERMIMIC_ROBOT"] = a.robot

    d = dict(np.load(a.traj))
    names = [str(n) for n in d["names"]]
    per_clip = json.load(open(a.eval))["per_clip"] if a.eval else {}
    spec = get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    # q columns are in robot_spec order; the MJCF may declare its joints in another order
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in spec.joint_names])
    assert model.nq == 7 + spec.num_joints and d["q"].shape[-1] == spec.num_joints, (model.nq, d["q"].shape)

    server = viser.ViserServer(host=a.host, port=a.port)
    server.scene.set_up_direction("+z")
    world = server.scene.add_frame("/world", show_axes=False)
    grid = server.scene.add_grid("/world/ground", width=40.0, height=40.0, cell_size=0.5, section_size=2.0)
    robot = Robot(server, model, "world/robot", color=(110, 112, 122), opacity=None)
    ghost = Robot(server, model, "world/reference", color=(90, 165, 255), opacity=0.35)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        k = spec.length_scale  # nemo_v3 is about half the size of the G1
        client.camera.position = (3.0 * k, -3.0 * k, 1.6 * k)
        client.camera.look_at = (0.0, 0.0, 0.8 * k)

    with server.gui.add_folder("Playback"):
        clip_dd = server.gui.add_dropdown("Clip", options=tuple(names), initial_value=names[0])
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        t_slider = server.gui.add_slider("Time [s]", min=0.0, max=1.0, step=DT, initial_value=0.0)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        prev_btn = server.gui.add_button("Previous clip")
        next_btn = server.gui.add_button("Next clip")
    with server.gui.add_folder("Display"):
        show_ghost = server.gui.add_checkbox("Show reference (blue ghost)", initial_value=True)
        anchor = server.gui.add_checkbox("Anchor reference to robot xy", initial_value=True)
        follow = server.gui.add_checkbox("Keep robot centered", initial_value=True)
    info = server.gui.add_markdown("")

    T = d["root_p"].shape[0]
    state = {"c": 0, "steps": 1, "fell_at": None}

    def select(name: str):
        c = names.index(name)
        alive = d["alive"][:, c]
        n_alive = int(alive.sum())
        fell = n_alive < T and n_alive < int(d["clip_len"][c]) - 1
        state.update(c=c, steps=max(n_alive, 1), fell_at=n_alive if fell else None)
        t_slider.max = round((state["steps"] - 1) * DT, 2)
        t_slider.value = 0.0
        ret = f"{per_clip[name]:.3f}" if name in per_clip else "n/a"
        status = "**terminated (fell)**" if fell else "completed"
        info.content = (
            f"**{name}**  \nlength {state['steps'] * DT:.1f} s · {status} · normalized return {ret}  \n"
            f"grey: simulated {spec.name} policy (Isaac Lab + MuJoCo-Warp) · blue ghost: reference motion"
        )

    @clip_dd.on_update
    def _(_):
        select(clip_dd.value)

    @prev_btn.on_click
    def _(_):
        clip_dd.value = names[(names.index(clip_dd.value) - 1) % len(names)]

    @next_btn.on_click
    def _(_):
        clip_dd.value = names[(names.index(clip_dd.value) + 1) % len(names)]

    @show_ghost.on_update
    def _(_):
        ghost.set_visible(show_ghost.value)

    select(names[0])
    print(f"viser playback of {len(names)} clips: open http://localhost:{a.port}", flush=True)
    qpos_r, qpos_g = model.qpos0.copy(), model.qpos0.copy()
    rquat = np.empty(4)
    last = time.time()
    t_play, t_shown = 0.0, 0.0  # playback time (float): rounding it to the 50 Hz grid every loop made speeds < ~1.3 play alike
    while True:
        now = time.time()
        if abs(t_slider.value - t_shown) > 1e-6:  # the user dragged the slider, or a clip was selected
            t_play = t_slider.value
        if playing.value:
            t_play += (now - last) * speed.value
            if t_play > t_slider.max:
                t_play = 0.0 if loop.value else t_slider.max
            t_shown = round(min(t_play, t_slider.max) / DT) * DT
            t_slider.value = t_shown
        last = now
        c = state["c"]
        tt = min(int(round(t_slider.value / DT)), state["steps"] - 1)
        f = int(d["f"][tt, c])
        qpos_r[0:3], qpos_r[3:7] = d["root_p"][tt, c], d["root_quat"][tt, c]
        qpos_r[qadr] = d["q"][tt, c]
        qpos_g[0:3] = d["ref_root_p"][f]
        if anchor.value:  # as the tracking reward does: reference anchored at the robot's xy
            qpos_g[0:2] = qpos_r[0:2]
        mujoco.mju_mat2Quat(rquat, d["ref_root_R"][f].reshape(-1).astype(np.float64))
        qpos_g[3:7] = rquat
        qpos_g[qadr] = d["ref_q"][f]
        with server.atomic():
            robot.set_qpos(qpos_r)
            if show_ghost.value:
                ghost.set_qpos(qpos_g)
            # shift the world instead of the camera, so mouse orbit/zoom stays with the user;
            # the grid moves in whole sections under the robot, so it still shows the motion
            xy = qpos_r[0:2] if follow.value else np.zeros(2)
            world.position = (-xy[0], -xy[1], 0.0)
            grid.position = (round(xy[0] / 2.0) * 2.0, round(xy[1] / 2.0) * 2.0, 0.0)
        time.sleep(1.0 / 50)


if __name__ == "__main__":
    main()
