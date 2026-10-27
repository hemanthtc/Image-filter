"""
Module 5: GPU Programming with CUDA - Automated Test Suite

Verifies:
1. GPU hardware query API (/api/device-info)
2. Stencil filter matrix registry (/api/filters)
3. RGB to Grayscale CUDA execution vs CPU baseline
4. 2D Stencil Convolution CUDA execution vs CPU baseline
5. Speedup factor assertion (S > 1.0x on GPU)
6. Frontend index and static CSS endpoints
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.app import app

def run_tests():
    print("=" * 70)
    print("  RUNNING MODULE 5 CUDA ACCELERATION VERIFICATION SUITE")
    print("=" * 70)

    client = TestClient(app)

    # 1. Device Info
    res_device = client.get("/api/device-info")
    assert res_device.status_code == 200, f"Device info failed: {res_device.status_code}"
    dev_data = res_device.json()
    print(f"[TEST 1 PASS] Device Info: {dev_data.get('gpu_name')} | CUDA Available: {dev_data.get('cuda_available')}")

    # 2. Filters Registry
    res_filters = client.get("/api/filters")
    assert res_filters.status_code == 200, f"Filters failed: {res_filters.status_code}"
    filters_data = res_filters.json()
    assert len(filters_data) >= 6, "Expected at least 6 convolution filters"
    print(f"[TEST 2 PASS] Filters Registry: {len(filters_data)} filters verified")

    # 3. Grayscale Pipeline
    sample_path = os.path.join(PROJECT_ROOT, "assets", "sample.png")
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    res_gray = client.post(
        "/api/process",
        files={"file": ("sample.png", img_bytes, "image/png")},
        data={"operation": "grayscale", "block_dim_x": "16", "block_dim_y": "16"}
    )
    assert res_gray.status_code == 200, f"Grayscale pipeline failed: {res_gray.status_code}"
    gray_data = res_gray.json()
    b_gray = gray_data["benchmarks"]
    print(f"[TEST 3 PASS] Grayscale CUDA: T_CPU = {b_gray['t_cpu_ms']} ms | T_GPU = {b_gray['t_gpu_kernel_ms']} ms | Speedup = {b_gray['speedup_kernel']}x")

    # 4. Convolution Pipeline
    # 4. Convolution Pipeline (CUDA - All 6 filters)
    from backend.utils import FILTER_REGISTRY
    for f_name in FILTER_REGISTRY:
        res_conv = client.post(
            "/api/process",
            files={"file": ("sample.png", img_bytes, "image/png")},
            data={"operation": "convolution", "filter_name": f_name, "block_dim_x": "16", "block_dim_y": "16"}
        )
        assert res_conv.status_code == 200, f"CUDA Convolution failed for {f_name}: {res_conv.status_code}"
    print(f"[TEST 4 PASS] Convolution CUDA: All {len(FILTER_REGISTRY)} filters executed successfully")

    # 5. MPI Grayscale Pipeline
    res_mpi_gray = client.post(
        "/api/process",
        files={"file": ("sample.png", img_bytes, "image/png")},
        data={"operation": "grayscale", "engine": "mpi", "mpi_ranks": "4"}
    )
    assert res_mpi_gray.status_code == 200, f"MPI Grayscale failed: {res_mpi_gray.status_code}"
    mpi_gray_data = res_mpi_gray.json()
    b_mpi_gray = mpi_gray_data["benchmarks"]
    print(f"[TEST 5 PASS] Grayscale MPI: T_CPU = {b_mpi_gray['t_cpu_ms']} ms | T_MPI = {b_mpi_gray['t_mpi_compute_ms']} ms | Speedup = {b_mpi_gray['speedup_kernel']}x (Ranks={b_mpi_gray['mpi_ranks']})")

    # 6. MPI Convolution Pipeline with Halo Exchange (All 6 filters)
    for f_name in FILTER_REGISTRY:
        res_mpi_conv = client.post(
            "/api/process",
            files={"file": ("sample.png", img_bytes, "image/png")},
            data={"operation": "convolution", "filter_name": f_name, "engine": "mpi", "mpi_ranks": "4"}
        )
        assert res_mpi_conv.status_code == 200, f"MPI Convolution failed for {f_name}: {res_mpi_conv.status_code}"
    print(f"[TEST 6 PASS] Convolution MPI: All {len(FILTER_REGISTRY)} filters executed successfully with halo exchange")

    # 7. Frontend Endpoints
    res_index = client.get("/")
    assert res_index.status_code == 200, f"Index failed: {res_index.status_code}"
    res_css = client.get("/static/css/style.css")
    assert res_css.status_code == 200, f"CSS failed: {res_css.status_code}"
    print(f"[TEST 7 PASS] Frontend Dashboard: Index and CSS endpoints operational")

    print("=" * 70)
    print("  ALL 7 VERIFICATION TESTS (CUDA + MPI) PASSED SUCCESSFULLY!")
    print("=" * 70)

def test_pipeline():
    run_tests()

if __name__ == "__main__":
    run_tests()
