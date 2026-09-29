"""End-to-end inspection: preprocess -> (optional) register -> detect -> report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from .inspection import colorize, detect_defects, deviation_to_reference
from .preprocess import preprocess
from .registration import align


def inspect(scan: o3d.geometry.PointCloud, reference: o3d.geometry.PointCloud | None = None,
            voxel: float = 0.005, threshold_mm: float = 4.0) -> tuple[dict, o3d.geometry.PointCloud]:
    t0 = time.perf_counter()
    clean, stats = preprocess(scan, voxel=voxel)
    report: dict = {"preprocessing": stats}
    deviation = None
    if reference is not None:
        ref_clean, _ = preprocess(reference, voxel=voxel)
        a = align(clean, ref_clean)
        clean.transform(a.transformation)
        report["registration"] = {"fitness": round(a.fitness, 3), "rmse_mm": round(a.rmse_mm, 2),
                                  "transformation": np.round(a.transformation, 5).tolist()}
        deviation = deviation_to_reference(clean, ref_clean)  # expected geometry = reference
    defects, dev = detect_defects(clean, threshold_mm=threshold_mm, voxel=voxel, deviation=deviation)
    report["defects"] = [d.to_dict() for d in defects]
    report["summary"] = {k: sum(d.kind == k for d in defects) for k in ("bump", "hollow", "crack")}
    report["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return report, colorize(clean, dev)


def save(report: dict, colored: o3d.geometry.PointCloud, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    o3d.io.write_point_cloud(str(out / "deviation_map.ply"), colored)
