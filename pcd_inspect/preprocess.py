"""Point-cloud cleaning: down-sampling, outlier removal, normals."""
from __future__ import annotations

import numpy as np
import open3d as o3d


def to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def preprocess(pcd: o3d.geometry.PointCloud, voxel: float = 0.005, nb_neighbors: int = 20,
               std_ratio: float = 3.0) -> tuple[o3d.geometry.PointCloud, dict]:
    n0 = len(pcd.points)
    down = pcd.voxel_down_sample(voxel) if voxel > 0 else pcd
    clean, _ = down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    stats = {"points_in": n0, "after_voxel": len(down.points), "after_outliers": len(clean.points)}
    return clean, stats
