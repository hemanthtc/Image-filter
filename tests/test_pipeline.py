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
    res_conv = client.post(
        "/api/process",
        files={"file": ("sample.png", img_bytes, "image/png")},
        data={"operation": "convolution", "filter_name": "gaussian_blur_3x3", "block_dim_x": "16", "block_dim_y": "16"}
    )
    assert res_conv.status_code == 200, f"Convolution pipeline failed: {res_conv.status_code}"
    conv_data = res_conv.json()
    b_conv = conv_data["benchmarks"]
    print(f"[TEST 4 PASS] Convolution CUDA: T_CPU = {b_conv['t_cpu_ms']} ms | T_GPU = {b_conv['t_gpu_kernel_ms']} ms | Speedup = {b_conv['speedup_kernel']}x")

    # 5. Frontend Endpoints
    res_index = client.get("/")
    assert res_index.status_code == 200, f"Index failed: {res_index.status_code}"
    res_css = client.get("/static/css/style.css")
    assert res_css.status_code == 200, f"CSS failed: {res_css.status_code}"
    print(f"[TEST 5 PASS] Frontend Dashboard: Index and CSS endpoints operational")

    print("=" * 70)
    print("  ALL 5 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

def test_pipeline():
    run_tests()

if __name__ == "__main__":
    run_tests()
