import io

import numpy as np
from fastapi.testclient import TestClient

from pcd_inspect.api import app
from pcd_inspect.synthetic import make_wall

client = TestClient(app)


def _xyz(points: np.ndarray) -> bytes:
    buf = io.StringIO()
    np.savetxt(buf, points, fmt="%.5f")
    return buf.getvalue().encode()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_inspect_upload():
    pts, _ = make_wall()
    r = client.post("/inspect", files={"scan": ("wall.xyz", _xyz(pts), "text/plain")})
    assert r.status_code == 200
    assert r.json()["summary"] == {"bump": 1, "hollow": 1, "crack": 1}


def test_rejects_bad_format():
    r = client.post("/inspect", files={"scan": ("x.txt", b"abc", "text/plain")})
    assert r.status_code == 400
