from fastapi.testclient import TestClient

from app.main import app


def test_health_and_infer_sample():
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        s = client.get("/v1/sample?seed=2")
        assert s.status_code == 200
        assert "image_png" in s.json()
        r = client.post("/v1/infer", data={"note": s.json()["note"], "seed": 2})
        assert r.status_code == 200
        body = r.json()
        assert "report" in body
        assert "fields" in body["report"]
        assert body["latency_ms"] >= 0
        m = client.get("/metrics")
        assert m.status_code == 200
        assert b"gliomagate_infer" in m.content
