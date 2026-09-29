import numpy as np
import pytest

from pcd_inspect.inspection import detect_defects
from pcd_inspect.pipeline import inspect
from pcd_inspect.preprocess import preprocess, to_pcd
from pcd_inspect.registration import align
from pcd_inspect.synthetic import make_corner, make_wall, rigid_transform


def _match(detected, truth, tol=0.03):
    """Every ground-truth defect is found once, with the right kind, near the right place."""
    for d in truth:
        near = [x for x in detected if np.hypot(x["centroid"][0] - d.center[0], x["centroid"][1] - d.center[1])
                < (d.size / 2 if d.kind == "crack" else tol)]
        assert len(near) == 1, (d, detected)
        assert near[0]["kind"] == d.kind


def test_preprocess_removes_outliers():
    pts, _ = make_wall(defects=[], outlier_ratio=0.01)
    clean, stats = preprocess(to_pcd(pts))
    z = np.asarray(clean.points)[:, 2]
    assert stats["after_outliers"] < stats["after_voxel"]
    assert np.abs(z).max() < 0.02  # spikes of 5-20 cm are gone
    assert clean.has_normals()


def test_detects_defects_on_planar_surface():
    pts, truth = make_wall()
    clean, _ = preprocess(to_pcd(pts))
    defects, dev = detect_defects(clean)
    assert len(defects) == 3
    assert len(dev) == len(clean.points)
    _match([d.to_dict() for d in defects], truth)
    hollow = next(d for d in defects if d.kind == "hollow")
    assert -17 < hollow.max_deviation_mm < -12  # true depth 15 mm


def test_clean_surface_has_no_defect():
    pts, _ = make_wall(defects=[])
    clean, _ = preprocess(to_pcd(pts))
    assert detect_defects(clean)[0] == []


def test_registration_recovers_known_pose():
    ref, _ = make_corner(defects=[], seed=1)
    scan, _ = make_corner(defects=[], seed=2)
    T = rigid_transform(angle_deg=6, translation=(0.04, -0.03, 0.05))
    a = align(to_pcd(scan).transform(T), to_pcd(ref))
    residual = a.transformation @ T  # should be identity
    angle = np.degrees(np.arccos(np.clip((np.trace(residual[:3, :3]) - 1) / 2, -1, 1)))
    assert angle < 0.5
    assert np.linalg.norm(residual[:3, 3]) < 0.005
    assert a.fitness > 0.95


@pytest.mark.slow
def test_end_to_end_with_reference():
    pts, truth = make_corner()
    ref, _ = make_corner(defects=[], seed=1)
    report, colored = inspect(to_pcd(pts).transform(rigid_transform()), to_pcd(ref))
    assert report["summary"] == {"bump": 1, "hollow": 1, "crack": 1}
    assert report["registration"]["fitness"] > 0.95
    _match(report["defects"], truth)
    assert colored.has_colors()
