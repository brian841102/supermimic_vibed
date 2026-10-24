"""Side-by-side 480p video: source video (left) vs the retargeted nemo_v3 motion (right, MuJoCo, EGL).

The robot camera tracks the root; with --gvhmr_pred its azimuth follows the source camera's viewing direction
(R_c2w from GVHMR), so both panels look from roughly the same side.
Usage (gmr env): MUJOCO_GL=egl python retarget/tools/render_compare.py --motion out/skate60_full.pkl \
    --video data/videos/skate60.mp4 --gvhmr_pred data/gvhmr_cam/skate60/hmr4d_results.pt --out compare.mp4
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
REF_XML = os.path.join(HERE, "..", "assets", "nemo_v3", "booster_mjlab_nemov3", "xmls", "nemov3.xml")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PW, PH = 640, 480  # each panel; output 1280x480


def camera_azimuth(pred_file, n):
    """Per-frame azimuth (deg, MuJoCo convention) of the source camera's optical axis in the z-up world."""
    import torch

    d = torch.load(pred_file, map_location="cpu")
    Rc2w = R.from_rotvec(d["smpl_params_global"]["global_orient"].numpy()) * R.from_rotvec(d["smpl_params_incam"]["global_orient"].numpy()).inv()
    fwd = Rc2w.apply([0.0, 0.0, 1.0])  # OpenCV camera looks along +z (y-up GVHMR world)
    M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # GMR: y-up -> z-up
    fwd = fwd @ M.T
    az = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    az = gaussian_filter1d(az, 15, mode="nearest")
    return np.degrees(az[:n])


def build_model():
    spec = mujoco.MjSpec.from_file(REF_XML)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="ice", contype=0, conaffinity=0)
    spec.add_texture(name="ice", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.86, 0.92, 0.96], rgb2=[0.74, 0.82, 0.88], width=512, height=512)  # fmt: skip
    mat = spec.add_material(name="ice", texrepeat=[20, 20], reflectance=0.15)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ice"
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = PW, PH
    m = spec.compile()
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.5
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="MotionFile pkl from gvhmr_to_nemov3.py")
    ap.add_argument("--video", required=True)
    ap.add_argument("--gvhmr_pred", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()

    mo = pickle.load(open(a.motion, "rb"))
    fps = float(mo["fps"])
    N = len(mo["root_pos"]) if a.seconds is None else min(len(mo["root_pos"]), int(a.seconds * fps))
    m = build_model()
    d = mujoco.MjData(m)
    order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    assert len(order) == mo["dof_pos"].shape[1]
    az = camera_azimuth(a.gvhmr_pred, N) if a.gvhmr_pred else np.full(N, 135.0)
    renderer = mujoco.Renderer(m, PH, PW)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 1.8, -12.0
    cap = cv2.VideoCapture(a.video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 16)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * PW}x{PH}", "-r", str(fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)  # fmt: skip
    look = gaussian_filter1d(mo["root_pos"][:N, :2], 6, axis=0, mode="nearest")
    for i in range(N):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(i / fps * vfps))) if abs(vfps - fps) > 1e-3 else None
        ok, frame = cap.read()
        left = np.zeros((PH, PW, 3), np.uint8)
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            s = min(PW / w, PH / h)
            fr = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            y0, x0 = (PH - fr.shape[0]) // 2, (PW - fr.shape[1]) // 2
            left[y0 : y0 + fr.shape[0], x0 : x0 + fr.shape[1]] = fr
        d.qpos[:3] = mo["root_pos"][i]
        q = np.asarray(mo["root_rot"][i])
        d.qpos[3:7] = q[[3, 0, 1, 2]]  # xyzw -> wxyz
        for c, n in enumerate(order):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = mo["dof_pos"][i, c]
        mujoco.mj_kinematics(m, d)
        cam.lookat[:] = [look[i, 0], look[i, 1], 0.35]
        cam.azimuth = az[i]  # MuJoCo azimuth = viewing direction: look the same way as the source camera
        renderer.update_scene(d, camera=cam)
        right = renderer.render()
        img = Image.fromarray(np.concatenate((left, right), 1))
        dr = ImageDraw.Draw(img)
        for xy, txt, f in (((10, 8), "原视频（GVHMR 输入）", font), ((PW + 10, 8), f"nemo_v3 重定向 {a.title}", font),
                           ((PW + 10, PH - 28), f"t = {i / fps:5.1f} s   root {mo['root_pos'][i][2]:.2f} m", small)):  # fmt: skip
            bb = dr.textbbox(xy, txt, font=f)
            dr.rectangle((bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3), fill=(0, 0, 0))
            dr.text(xy, txt, fill=(255, 255, 255), font=f)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out, f"({N} frames, {N / fps:.1f} s)")


if __name__ == "__main__":
    main()
