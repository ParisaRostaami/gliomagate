"""Measure p95 latency and a Cloud Run cost estimate. Honest numbers only."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    k = int(round((p / 100) * (len(ys) - 1)))
    return ys[k]


def main() -> None:
    n = 80
    lat = []
    with TestClient(app) as client:
        # warmup
        for i in range(5):
            client.post("/v1/infer", data={"note": "left temporal seizure enhancement", "seed": i})
        t_all = time.perf_counter()
        for i in range(n):
            t0 = time.perf_counter()
            r = client.post("/v1/infer", data={"note": "left temporal seizure with ring enhancement grade 4", "seed": i})
            assert r.status_code == 200, r.text
            lat.append((time.perf_counter() - t0) * 1000)
        wall = time.perf_counter() - t_all

    rps = n / wall
    p50 = percentile(lat, 50)
    p95 = percentile(lat, 95)
    # Cloud Run us-central1, 1 vCPU $0.0000240 / vCPU-second (list price, 2025-2026 public)
    vcpu_s = (p95 / 1000.0) * 1.0
    cost = vcpu_s * 0.0000240
    out = {
        "n": n,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "mean_ms": round(statistics.mean(lat), 2),
        "throughput_rps_serial": round(rps, 2),
        "cloud_run_list_usd_per_infer_at_p95_1vcpu": round(cost, 6),
        "notes": "Measured in-process via TestClient on this machine. Not a 40 rps load-test on GPU. Cost is list-price Cloud Run CPU time at measured p95, no egress.",
    }
    path = ROOT / "models" / "bench.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
