"""Browser playback of retargeted nemo_v3 motions (MotionFile pkl from gvhmr_to_nemov3.py) with viser.

Shows the robot (nemov3.xml visual meshes, posed by MuJoCo FK) and, optionally, the source video frame in the panel.
Usage (gmr env): python retarget/tools/viser_play_motion.py data/retarget_out/skate60_full.pkl data/retarget_out/skate60_hw.pkl \
    --video data/videos/skate60.mp4 [--port 8080]      then open http://<host>:8080
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import mujoco
import numpy as np
import viser

HERE = os.path.dirname(os.path.abspath(__file__))
REF_XML = os.path.join(HERE, "..", "assets", "nemo_v3", "booster_mjlab_nemov3", "xmls", "nemov3.xml")


def visual_geoms(m):
    mesh = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    vis = [g for g in mesh if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
    return vis or mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motions", nargs="+")
    ap.add_argument("--video", default=None, help="source video shown in the side panel")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    clips = {os.path.splitext(os.path.basename(p))[0]: pickle.load(open(p, "rb")) for p in a.motions}
    m = mujoco.MjModel.from_xml_path(REF_XML)
    d = mujoco.MjData(m)
    order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in order]
    frames = None
    if a.video:
        import cv2

        cap = cv2.VideoCapture(a.video)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2RGB))

    server = viser.ViserServer(host="0.0.0.0", port=a.port)
    server.scene.set_up_direction("+z")
    world = server.scene.add_frame("/world", show_axes=False)
    grid = server.scene.add_grid("/world/ground", width=30.0, height=30.0, cell_size=0.25, section_size=1.0)
    geoms = visual_geoms(m)
    handles = []
    for g in geoms:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        rgba = m.geom_rgba[g]
        handles.append(server.scene.add_mesh_simple(f"/world/robot/g{g}", vertices=v, faces=f, color=tuple(int(255 * c) for c in rgba[:3])))
    trail = None

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (1.6, -1.6, 0.9)
        client.camera.look_at = (0.0, 0.0, 0.35)

    names = list(clips)
    with server.gui.add_folder("播放 Playback"):
        clip_dd = server.gui.add_dropdown("Clip", options=tuple(names), initial_value=names[0])
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
        t_slider = server.gui.add_slider("Time [s]", min=0.0, max=1.0, step=1 / 30, initial_value=0.0)
        follow = server.gui.add_checkbox("Keep robot centred", initial_value=True)
        show_trail = server.gui.add_checkbox("Show root path", initial_value=True)
    info = server.gui.add_markdown("")
    img = server.gui.add_image(frames[0], label="原视频 source") if frames else None
    state = {}

    def select(name):
        nonlocal trail
        c = clips[name]
        state.update(c=c, fps=float(c["fps"]), n=len(c["root_pos"]))
        t_slider.max = round((state["n"] - 1) / state["fps"], 3)
        t_slider.value = 0.0
        p = np.asarray(c["root_pos"])
        sp = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * state["fps"]
        info.content = f"**{name}**  \n{state['n']} frames @ {state['fps']:.0f} fps · root speed mean {sp.mean():.2f} m/s · path {sp.sum() / state['fps']:.1f} m"
        pts = np.stack((p[:-1], p[1:]), 1).copy()
        pts[..., 2] = 0.005
        trail = server.scene.add_line_segments("/world/trail", points=pts.astype(np.float32), colors=(255, 120, 40), line_width=2.0)

    @clip_dd.on_update
    def _(_):
        select(clip_dd.value)

    select(names[0])
    print(f"viser: open http://<this-host>:{a.port}", flush=True)
    last = time.time()
    qd = np.empty(4)
    while True:
        now = time.time()
        if playing.value:
            t = t_slider.value + (now - last) * speed.value
            t_slider.value = 0.0 if t > t_slider.max else t
        last = now
        c, fps = state["c"], state["fps"]
        i = min(int(round(t_slider.value * fps)), state["n"] - 1)
        d.qpos[:3] = c["root_pos"][i]
        d.qpos[3:7] = np.asarray(c["root_rot"][i])[[3, 0, 1, 2]]
        d.qpos[qadr] = c["dof_pos"][i]
        mujoco.mj_kinematics(m, d)
        xy = d.qpos[:2].copy() if follow.value else np.zeros(2)
        with server.atomic():
            world.position = (-xy[0], -xy[1], 0.0)
            grid.position = (round(xy[0]), round(xy[1]), 0.0)
            for g, h in zip(geoms, handles):
                mujoco.mju_mat2Quat(qd, d.geom_xmat[g])
                h.position = d.geom_xpos[g].copy()
                h.wxyz = qd.copy()
            if trail is not None:
                trail.visible = show_trail.value
            if img is not None:
                img.image = frames[min(int(round(i / fps * 30)), len(frames) - 1)]
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
