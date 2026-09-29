"""REST API: upload a scan (and optionally a reference), get the defect report."""
from __future__ import annotations

import tempfile
from pathlib import Path

import open3d as o3d
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .pipeline import inspect

app = FastAPI(title="Point-cloud defect inspection", version="1.0.0")


def _read(upload: UploadFile, data: bytes) -> o3d.geometry.PointCloud:
    suffix = Path(upload.filename or "scan.ply").suffix.lower()
    if suffix not in {".ply", ".pcd", ".xyz", ".pts"}:
        raise HTTPException(400, f"Unsupported format {suffix}: use .ply, .pcd, .xyz or .pts")
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        pcd = o3d.io.read_point_cloud(f.name)
    if len(pcd.points) == 0:
        raise HTTPException(400, "Empty or unreadable point cloud")
    return pcd


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "open3d": o3d.__version__}


@app.post("/inspect")
async def inspect_scan(scan: UploadFile = File(...), reference: UploadFile | None = File(None),
                       threshold_mm: float = Query(4.0, gt=0, le=100)) -> dict:
    scan_pcd = _read(scan, await scan.read())
    ref_pcd = _read(reference, await reference.read()) if reference is not None else None
    report, _ = inspect(scan_pcd, ref_pcd, threshold_mm=threshold_mm)
    return report
