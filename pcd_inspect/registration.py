"""Scan-to-reference alignment: FPFH + RANSAC global registration, refined by ICP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration


@dataclass
class Alignment:
    transformation: np.ndarray
    fitness: float
    rmse_mm: float


def _fpfh(pcd: o3d.geometry.PointCloud, voxel: float):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    return reg.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))


def align(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel: float = 0.02,
          icp_distance: float = 0.01, seed: int = 0) -> Alignment:
    """Estimate the transform that maps ``source`` (scan) onto ``target`` (reference)."""
    o3d.utility.random.seed(seed)
    src, tgt = source.voxel_down_sample(voxel), target.voxel_down_sample(voxel)
    f_src, f_tgt = _fpfh(src, voxel), _fpfh(tgt, voxel)
    coarse = reg.registration_ransac_based_on_feature_matching(
        src, tgt, f_src, f_tgt, True, voxel * 1.5,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        reg.RANSACConvergenceCriteria(100000, 0.999),
    )
    if not target.has_normals():
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=icp_distance * 4, max_nn=30))
    init = coarse.transformation
    for dist in (icp_distance * 4, icp_distance):  # coarse-to-fine ICP
        fine = reg.registration_icp(source, target, dist, init, reg.TransformationEstimationPointToPlane(),
                                    reg.ICPConvergenceCriteria(max_iteration=60))
        init = fine.transformation
    return Alignment(np.asarray(fine.transformation), float(fine.fitness), float(fine.inlier_rmse) * 1000)
