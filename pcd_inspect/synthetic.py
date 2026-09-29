"""Synthetic wall scans with known defects, used for demos and tests.

Units are metres. A defect is a local deviation from the plane z = 0:
bumps are positive, hollows and cracks are negative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Defect:
    kind: str                 # "bump" | "hollow" | "crack"
    center: tuple[float, float]
    size: float               # radius (bump/hollow) or length (crack), m
    depth: float              # peak deviation, m (positive)
    angle: float = 0.0        # crack orientation, rad
    width: float = 0.01       # crack width, m

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_DEFECTS = [
    Defect("bump", (0.5, 0.4), 0.06, 0.012),
    Defect("hollow", (1.4, 1.0), 0.08, 0.015),
    Defect("crack", (1.0, 0.35), 0.40, 0.010, angle=0.3, width=0.02),
]


def _deviation(xy: np.ndarray, d: Defect) -> np.ndarray:
    dx, dy = xy[:, 0] - d.center[0], xy[:, 1] - d.center[1]
    if d.kind in ("bump", "hollow"):
        r2 = dx**2 + dy**2
        z = d.depth * np.exp(-r2 / (2 * (d.size / 2) ** 2))
        z[r2 > d.size**2] = 0
        return z if d.kind == "bump" else -z
    # crack: thin groove along a segment
    u = dx * np.cos(d.angle) + dy * np.sin(d.angle)
    v = -dx * np.sin(d.angle) + dy * np.cos(d.angle)
    inside = (np.abs(u) <= d.size / 2) & (np.abs(v) <= d.width / 2)
    return np.where(inside, -d.depth, 0.0)


def make_wall(width: float = 2.0, height: float = 1.5, spacing: float = 0.005,
              defects: list[Defect] | None = None, noise: float = 0.0008,
              outlier_ratio: float = 0.002, seed: int = 0) -> tuple[np.ndarray, list[Defect]]:
    """Return an (N, 3) array of points and the list of defects it contains."""
    rng = np.random.default_rng(seed)
    defects = DEFAULT_DEFECTS if defects is None else defects
    xs, ys = np.meshgrid(np.arange(0, width, spacing), np.arange(0, height, spacing))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    xy += rng.uniform(-spacing / 3, spacing / 3, xy.shape)  # scanner jitter
    z = sum((_deviation(xy, d) for d in defects), np.zeros(len(xy)))
    z += rng.normal(0, noise, len(xy))
    pts = np.column_stack([xy, z])
    n_out = int(outlier_ratio * len(pts))
    if n_out:
        idx = rng.choice(len(pts), n_out, replace=False)
        pts[idx, 2] += rng.uniform(0.05, 0.2, n_out) * rng.choice([-1, 1], n_out)
    return pts, defects


def _grid(u_len: float, v_len: float, spacing: float, rng) -> tuple[np.ndarray, np.ndarray]:
    us, vs = np.meshgrid(np.arange(0, u_len, spacing), np.arange(0, v_len, spacing))
    u, v = us.ravel(), vs.ravel()
    return u + rng.uniform(-spacing / 3, spacing / 3, u.shape), v + rng.uniform(-spacing / 3, spacing / 3, v.shape)


def make_corner(width: float = 2.0, height: float = 1.5, depth: float = 0.5, spacing: float = 0.005,
                defects: list[Defect] | None = None, noise: float = 0.0008, seed: int = 0
                ) -> tuple[np.ndarray, list[Defect]]:
    """Room corner: defective wall (z = 0) + floor (y = 0) + side wall (x = 0).

    The three planes make registration well-posed (a single flat wall can slide
    or flip under ICP, so real inspections always register on richer geometry).
    """
    rng = np.random.default_rng(seed + 100)
    wall, defects = make_wall(width, height, spacing, defects, noise, outlier_ratio=0.0, seed=seed)
    fx, fz = _grid(width, depth, spacing, rng)
    floor = np.column_stack([fx, rng.normal(0, noise, fx.shape), fz])
    sy, sz = _grid(height, depth, spacing, rng)
    side = np.column_stack([rng.normal(0, noise, sy.shape), sy, sz])
    return np.vstack([wall, floor, side]), defects


def rigid_transform(angle_deg: float = 5.0, axis=(0.2, 0.3, 1.0), translation=(0.03, -0.02, 0.05)) -> np.ndarray:
    """4x4 homogeneous transform (rotation about ``axis`` + translation)."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    a = np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, translation
    return T
