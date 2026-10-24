"""Convert an MJCF to USD with the standalone Isaac Sim MJCF importer (no Kit).

Usage: python tools/mjcf_to_usd.py [src.xml] [dst_dir] [--hull_vertices REGEX=N ...]   (defaults: the G1 MJCF -> data/g1_usd)

--hull_vertices raises the convex-hull vertex limit of the mesh colliders whose name matches REGEX. Newton reduces mesh
colliders to 64 hull vertices by default (qhull stopped after 64 points), which turns a wheel into a lopsided polygon;
MuJoCo itself (booster_mjlab) and PhysX setups keep a much finer hull. Written as newton:maxHullVertices (and the PhysX
hullVertexLimit, capped at 255) on the collider prims of the instance prototypes.
"""
import argparse
import glob
import os
import re

from pxr import Plug, Sdf

here = os.path.dirname(os.path.abspath(__file__))
import isaacsim.asset.importer.utils.impl as _iu

Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(_iu.__file__), "schemas"))
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig


def set_hull_vertices(usd_dir: str, rules: list[tuple[str, int]]) -> int:
    n = 0
    for path in glob.glob(os.path.join(usd_dir, "**", "*.usd*"), recursive=True):
        layer = Sdf.Layer.FindOrOpen(path)
        if layer is None or layer.GetFileFormat().formatId == "usdc":
            continue
        changed = []

        def visit(p):
            spec = layer.GetPrimAtPath(p)
            if spec is None or not p.IsPrimPath():
                return
            schemas = list(spec.GetInfo("apiSchemas").GetAddedOrExplicitItems()) if spec.HasInfo("apiSchemas") else []
            if "PhysicsMeshCollisionAPI" in schemas:
                for pat, nv in rules:
                    if re.fullmatch(pat, spec.name):
                        for attr, val in (("newton:maxHullVertices", nv), ("physxConvexHullCollision:hullVertexLimit", min(nv, 255))):
                            a = spec.attributes.get(attr) or Sdf.AttributeSpec(spec, attr, Sdf.ValueTypeNames.Int)
                            a.default = val
                        changed.append(spec.name)
                        break

        layer.Traverse(Sdf.Path.absoluteRootPath, visit)
        if changed:
            layer.Save()
            n += len(changed)
            print(f"hull vertex limit set on {len(changed)} collider(s) in {os.path.relpath(path, usd_dir)}: {sorted(set(changed))}")
    return n


ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="?", default=os.path.join(here, "..", "data", "g1_description", "g1_29dof.xml"))
ap.add_argument("dst", nargs="?", default=os.path.join(here, "..", "data", "g1_usd"))
ap.add_argument("--hull_vertices", nargs="*", default=[], metavar="REGEX=N", help="e.g. '.*_wheel_collision=1024'")
a = ap.parse_args()
cfg = MJCFImporterConfig(mjcf_path=os.path.normpath(a.src), usd_path=os.path.normpath(a.dst))
print(MJCFImporter(cfg).import_mjcf())
if a.hull_vertices:
    rules = [(r.rsplit("=", 1)[0], int(r.rsplit("=", 1)[1])) for r in a.hull_vertices]
    assert set_hull_vertices(os.path.normpath(a.dst), rules) > 0, f"no mesh collider matched {a.hull_vertices}"
