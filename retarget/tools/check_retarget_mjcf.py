"""Check the retargeting MJCFs from make_retarget_mjcf.py against booster_mjlab's nemov3.xml, and render them.

Checks: IK joint list; link poses equal the training model's (welded joints held at their weld angles) over random
poses; wheel-centre / heading / hand markers where they should be; wheel centre 0.075 m above ground when the
straight-legged robot stands on its wheels.
Usage (EGL needs a GPU): MUJOCO_GL=egl python retarget/tools/check_retarget_mjcf.py [--render out.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from make_retarget_mjcf import ASSETS, HEADING_AHEAD, HOME_JOINT_POS, VARIANTS  # noqa: E402

ROOT = "link_leg2pelvis"
WHEEL_R = 0.075


def joint_names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]


def set_q(m, d, q, root_z=0.0):
    d.qpos[:] = m.qpos0
    d.qpos[2] = root_z
    for n, v in q.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_kinematics(m, d)


def body(m, d, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return d.xpos[i].copy(), d.xmat[i].reshape(3, 3).copy()


def check(variant, ref_xml, xml):
    m, d = mujoco.MjModel.from_xml_path(xml), None
    d = mujoco.MjData(m)
    mr = mujoco.MjModel.from_xml_path(ref_xml)
    dr = mujoco.MjData(mr)
    ik = joint_names(m)
    welded = VARIANTS[variant]
    print(f"\n=== {variant}: {os.path.basename(xml)}")
    print(f"IK joints ({len(ik)}): {ik}")
    assert set(ik) == set(joint_names(mr)) - set(welded), "joint set mismatch"
    # welded joints at their weld angle in the reference model
    weld_q = {n: HOME_JOINT_POS.get(n, 0.0) for n in welded}
    common = [mujoco.mj_id2name(m, 1, b) for b in range(1, m.nbody) if mujoco.mj_name2id(mr, 1, mujoco.mj_id2name(m, 1, b)) >= 0]
    rng = np.random.default_rng(0)
    worst_p = worst_r = 0.0
    for trial in range(200):
        q = {}
        for n in ik:
            j = mujoco.mj_name2id(m, 3, n)
            q[n] = rng.uniform(*m.jnt_range[j]) if (trial and m.jnt_limited[j]) else 0.0
        set_q(m, d, q)
        set_q(mr, dr, {**q, **weld_q})
        p0, R0 = body(m, d, ROOT)
        pr0, Rr0 = body(mr, dr, ROOT)
        for b in common:
            p, R = body(m, d, b)
            pr, Rr = body(mr, dr, b)
            worst_p = max(worst_p, np.linalg.norm(R0.T @ (p - p0) - Rr0.T @ (pr - pr0)))
            worst_r = max(worst_r, np.arccos(np.clip((np.trace((R0.T @ R).T @ (Rr0.T @ Rr)) - 1) / 2, -1, 1)))
        # markers: wheel centre = reference wheel-link origin; heading point perpendicular to the axle
        for side in ("left", "right"):
            c = body(m, d, f"{side}_wheel_center")[0]
            h = body(m, d, f"{side}_wheel_heading")[0]
            cr = body(mr, dr, f"{side}_wheel_link")[0]
            ja = mujoco.mj_name2id(mr, 3, f"{side}_wheel_joint")
            assert np.linalg.norm(c - cr) < 1e-9, "wheel centre marker off"
            assert abs(np.linalg.norm(h - c) - HEADING_AHEAD) < 1e-9 and abs(np.dot(h - c, dr.xaxis[ja])) < 1e-9, "heading marker off"
    print(f"link poses vs nemov3.xml over 200 random poses ({len(common)} links): max pos err {worst_p:.1e} m, max rot err {worst_r:.1e} rad")
    stand_z = None
    set_q(m, d, {})
    cz = body(m, d, "left_wheel_center")[0][2]
    stand_z = WHEEL_R - cz  # root height that puts the straight-legged robot's wheels on the ground
    set_q(m, d, {}, stand_z)
    for side in ("left", "right"):
        c, h = body(m, d, f"{side}_wheel_center")[0], body(m, d, f"{side}_wheel_heading")[0]
        hp = body(m, d, f"{side}_hand_point")[0]
        print(f"{side}: wheel centre {np.round(c, 4)} (z = r = {WHEEL_R}), heading - centre {np.round(h - c, 4)}, hand point {np.round(hp, 4)}")
    print(f"straight-legged standing root height {stand_z:.4f} m")
    return stand_z


def render(xmls, out):
    from PIL import Image, ImageDraw, ImageFont

    W, H = 640, 520
    font = ImageFont.load_default(size=18) if hasattr(ImageFont, "load_default") else None
    home = {k: v for k, v in HOME_JOINT_POS.items()}
    panels = []
    for variant, xml in xmls:
        spec = mujoco.MjSpec.from_file(xml)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], rgba=[0.85, 0.87, 0.9, 1], contype=0, conaffinity=0)
        spec.worldbody.add_light(pos=[1.5, -1.5, 3], dir=[-0.4, 0.4, -1], diffuse=[0.7, 0.7, 0.7], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
        m = spec.compile()
        d = mujoco.MjData(m)
        set_q(m, d, {})
        stand_z = WHEEL_R - body(m, d, "left_wheel_center")[0][2]
        ik = len(joint_names(m))
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.55
        renderer = mujoco.Renderer(m, H, W)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        marker = np.array([m.geom_group[g] == 2 for g in range(m.ngeom)])
        base_rgba = m.geom_rgba.copy()
        # waist + wrist-pitch test pose on top of HOME: only the full variant can follow it (hw ignores those joints)
        bend = {**home, "joint_pelvisYaw": 0.6, "joint_pelvisPitch": 0.45, "joint_pelvisRoll": 0.15,
                "joint_lWristPitch": 1.0, "joint_rWristPitch": -1.0, "joint_lShoulderRoll": 0.6, "joint_rShoulderRoll": 0.6}  # fmt: skip
        views = [
            ("zero pose (T-pose), front", {}, stand_z, dict(lookat=[0, 0, 0.45], distance=1.6, azimuth=180, elevation=-8), 1.0),
            ("zero pose, from robot's left", {}, stand_z, dict(lookat=[0, 0, 0.45], distance=1.6, azimuth=-90, elevation=-8), 1.0),
            ("HOME + waist yaw .6 pitch .45 roll .15, wrist pitch +-1", bend, 0.50, dict(lookat=[0, 0, 0.42], distance=1.5, azimuth=150, elevation=-12), 1.0),
            ("left wheel from outside, HOME (robot 12% opaque)", home, 0.50, dict(lookat=[0.05, 0.13, 0.09], distance=0.45, azimuth=-105, elevation=-5), 0.12),
        ]
        for title, q, z, cam_kw, alpha in views:
            m.geom_rgba[:] = base_rgba
            m.geom_rgba[~marker, 3] = base_rgba[~marker, 3] * alpha
            set_q(m, d, q, z)
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = cam_kw["lookat"], cam_kw["distance"], cam_kw["azimuth"], cam_kw["elevation"]
            renderer.update_scene(d, camera=cam, scene_option=opt)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, W, 30), fill=(255, 255, 255))
            draw.text((10, 5), f"{variant} ({ik} IK joints): {title}", fill=(20, 20, 20), font=font)
            panels.append(img)
        renderer.close()
    legend_h = 36
    sheet = Image.new("RGB", (4 * W, len(xmls) * H + legend_h), (255, 255, 255))
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % 4) * W, (i // 4) * H))
    draw = ImageDraw.Draw(sheet)
    y = len(xmls) * H + 8
    for x, rgb, txt in ((10, (230, 50, 50), "wheel centre (IK target <- human ankle)"), (560, (50, 200, 50), "wheel heading point, 0.12 m ahead (<- human toe)"),
                        (1180, (50, 100, 255), "hand point, 0.039 m below wrist pitch (<- human wrist)")):  # fmt: skip
        draw.ellipse((x, y + 2, x + 16, y + 18), fill=rgb)
        draw.text((x + 24, y), txt, fill=(20, 20, 20), font=font)
    sheet.save(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(ASSETS, "booster_mjlab_nemov3", "xmls", "nemov3.xml"))
    ap.add_argument("--dir", default=os.path.join(ASSETS, "retarget_mjcf"))
    ap.add_argument("--render", default=None, help="write a screenshot sheet (png)")
    a = ap.parse_args()
    xmls = [(v, os.path.join(a.dir, f"nemov3_retarget_{v}.xml")) for v in ("full", "hw")]
    for v, x in xmls:
        check(v, a.ref, x)
    if a.render:
        render(xmls, a.render)


if __name__ == "__main__":
    main()
