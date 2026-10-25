"""Training clip CSV (make_supermimic_clip.py, SuperMimic's 20-joint order) -> pkl for viser_play_motion.py.

The viewer plays the 32-joint nemov3.xml model, so the 12 joints welded on the current hardware (head, waist, push
rods, wrist pitch) are filled in at their HOME angles. The clip's own HOME hold / transitions are included.
Usage: python retarget/tools/clip_to_pkl.py data/supermimic_motions/roller_house/roller_house.csv --out data/retarget_out/roller_house_clip.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from make_retarget_mjcf import HOME_JOINT_POS  # noqa: E402

REF_XML = os.path.join(HERE, "..", "assets", "nemo_v3", "booster_mjlab_nemov3", "xmls", "nemov3.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    x = np.loadtxt(a.csv, delimiter=",")
    meta = json.load(open(a.csv.replace(".csv", ".json")))
    m = mujoco.MjModel.from_xml_path(REF_XML)
    order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    dof = np.zeros((len(x), len(order)), np.float32)
    for c, n in enumerate(order):
        dof[:, c] = HOME_JOINT_POS.get(n, 0.0)
    for k, n in enumerate(meta["joint_order"]):
        dof[:, order.index(n)] = x[:, 7 + k]
    with open(a.out, "wb") as f:
        pickle.dump({"fps": float(meta["fps"]), "root_pos": x[:, :3].astype(np.float32), "root_rot": list(x[:, 3:7].astype(np.float32)),
                     "dof_pos": dof, "link_body_list": [], "local_body_pos": np.zeros((len(x), 0, 3), np.float32)}, f)
    print(f"wrote {a.out}: {len(x)} frames @ {meta['fps']} fps, segments {meta['segments']}")


if __name__ == "__main__":
    main()
