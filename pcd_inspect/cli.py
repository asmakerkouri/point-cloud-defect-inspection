"""Command line: `python -m pcd_inspect.cli demo` or `... scan.ply --reference ref.ply`."""
from __future__ import annotations

import argparse
import json

import open3d as o3d

from .pipeline import inspect, save
from .preprocess import to_pcd
from .synthetic import make_corner, rigid_transform


def main() -> None:
    p = argparse.ArgumentParser(description="Point-cloud surface defect inspection")
    p.add_argument("scan", help="scan .ply/.pcd, or 'demo' for a synthetic wall")
    p.add_argument("--reference", help="reference cloud to register against")
    p.add_argument("--threshold-mm", type=float, default=4.0)
    p.add_argument("--out", default="output")
    a = p.parse_args()

    if a.scan == "demo":
        pts, truth = make_corner()
        ref, _ = make_corner(defects=[], seed=1)
        scan = to_pcd(pts).transform(rigid_transform())  # scanner pose differs from reference
        reference = to_pcd(ref)
        print("ground truth:", [d.kind for d in truth])
    else:
        scan = o3d.io.read_point_cloud(a.scan)
        reference = o3d.io.read_point_cloud(a.reference) if a.reference else None

    report, colored = inspect(scan, reference, threshold_mm=a.threshold_mm)
    save(report, colored, a.out)
    print(json.dumps({k: report[k] for k in ("summary", "processing_ms")}, indent=2))
    print(f"report + deviation map written to {a.out}/")


if __name__ == "__main__":
    main()
