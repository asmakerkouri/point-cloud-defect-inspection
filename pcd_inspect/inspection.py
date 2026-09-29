"""Defect detection: deviation from the fitted reference surface + clustering."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d


@dataclass
class DetectedDefect:
    id: int
    kind: str              # bump | hollow | crack
    n_points: int
    area_cm2: float
    max_deviation_mm: float
    centroid: list[float]
    length_cm: float
    elongation: float      # principal-axis ratio, high for cracks

    def to_dict(self) -> dict:
        return asdict(self)


def fit_plane(pcd: o3d.geometry.PointCloud, distance: float = 0.003) -> np.ndarray:
    """RANSAC plane (a, b, c, d) with a unit normal oriented towards +z of the cloud."""
    model, _ = pcd.segment_plane(distance_threshold=distance, ransac_n=3, num_iterations=2000)
    n = np.asarray(model, dtype=float)
    n /= np.linalg.norm(n[:3])
    return n if n[2] >= 0 else -n


def signed_distances(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return points @ plane[:3] + plane[3]


def deviation_to_reference(pcd: o3d.geometry.PointCloud, reference: o3d.geometry.PointCloud,
                           viewpoint=(1.0, 1.0, 3.0)) -> np.ndarray:
    """Signed cloud-to-cloud distance: (p - q) . n_q with q the nearest reference point."""
    ref = o3d.geometry.PointCloud(reference)
    ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    ref.orient_normals_towards_camera_location(np.asarray(viewpoint, dtype=float))
    tree = o3d.geometry.KDTreeFlann(ref)
    rp, rn, pts = np.asarray(ref.points), np.asarray(ref.normals), np.asarray(pcd.points)
    idx = np.array([tree.search_knn_vector_3d(p, 1)[1][0] for p in pts])
    return np.einsum("ij,ij->i", pts - rp[idx], rn[idx])


def detect_defects(pcd: o3d.geometry.PointCloud, threshold_mm: float = 4.0, eps: float = 0.025,
                   min_points: int = 15, voxel: float = 0.005,
                   deviation: np.ndarray | None = None) -> tuple[list[DetectedDefect], np.ndarray]:
    """Cluster points deviating more than ``threshold_mm``.

    ``deviation`` comes from :func:`deviation_to_reference`; without it the
    surface is assumed planar and a RANSAC plane is used as reference.
    """
    pts = np.asarray(pcd.points)
    dev = signed_distances(pts, fit_plane(pcd)) if deviation is None else deviation
    mask = np.abs(dev) > threshold_mm / 1000
    defects: list[DetectedDefect] = []
    if mask.sum() >= min_points:
        sub = pts[mask]
        labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub))
                            .cluster_dbscan(eps=eps, min_points=min_points))
        for lab in sorted(set(labels) - {-1}):
            idx = labels == lab
            cpts, cdev = sub[idx], dev[mask][idx]
            # principal axes in the surface plane
            centred = cpts - cpts.mean(0)
            eig = np.sort(np.linalg.eigvalsh(np.cov(centred.T)))[::-1]
            elong = float(np.sqrt(eig[0] / max(eig[1], 1e-12)))
            length = float(np.sqrt(12 * eig[0]))  # extent of a uniformly sampled segment
            peak = float(cdev[np.argmax(np.abs(cdev))])
            kind = "crack" if elong > 4 else ("bump" if peak > 0 else "hollow")
            defects.append(DetectedDefect(
                id=len(defects), kind=kind, n_points=int(idx.sum()),
                area_cm2=round(float(idx.sum()) * voxel**2 * 1e4, 1),
                max_deviation_mm=round(peak * 1000, 2),
                centroid=[round(float(v), 3) for v in cpts.mean(0)],
                length_cm=round(length * 100, 1), elongation=round(elong, 1),
            ))
    return defects, dev


def colorize(pcd: o3d.geometry.PointCloud, dev: np.ndarray, scale_mm: float = 15.0) -> o3d.geometry.PointCloud:
    """Blue = hollow, white = on plane, red = bump."""
    t = np.clip(dev * 1000 / scale_mm, -1, 1)[:, None]
    white, red, blue = np.ones(3), np.array([0.85, 0.1, 0.1]), np.array([0.1, 0.3, 0.85])
    colors = np.where(t > 0, white + t * (red - white), white + (-t) * (blue - white))
    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(colors)
    return out
