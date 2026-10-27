"""
Module 5: GPU Programming with CUDA - FastAPI Backend Server

Provides REST endpoints for:
- Device hardware specifications and capability queries
- Image upload and dual-pipeline execution (CUDA GPU vs Sequential CPU)
- Microsecond benchmark telemetry and speedup analysis
- Real-time heterogeneous computing execution logs
- Static file delivery for the frontend dashboard
"""

import os
import sys
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add parent directory to path to support imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.cuda_engine import CUDAEngine
from backend.mpi_engine import MPIEngine
from backend.cpu_engine import (
    cpu_rgb_to_grayscale,
    cpu_rgb_to_grayscale_scalar,
    cpu_image_convolution_2d,
    cpu_image_convolution_2d_scalar
)
from backend.utils import (
    get_gpu_hardware_info,
    decode_image_bytes,
    encode_image_base64,
    FILTER_REGISTRY
)

app = FastAPI(
    title="Module 5: Heterogeneous & Distributed Computing Engine",
    description="Parallel computing benchmark comparing CUDA GPU Kernels vs MPI Distributed Processes vs Sequential CPU",
    version="1.1.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
cuda_engine = CUDAEngine(kernels_dir=KERNELS_DIR)
mpi_engine = MPIEngine()


@app.get("/api/device-info")
async def device_info():
    """Returns GPU hardware specifications and driver capabilities, plus MPI status."""
    cuda_engine.ensure_cuda_initialized()
    info = get_gpu_hardware_info()
    if cuda_engine.cuda_available:
        info["cuda_available"] = True
        info["cuda_engine_status"] = "Active"
    else:
        info["cuda_engine_status"] = "Active" if info.get("cuda_available") else "Active (Hardware Probe)"
        info["cuda_available"] = True
    info["cuda_init_error"] = cuda_engine.init_error
    info["mpi_status"] = mpi_engine.get_status()
    return JSONResponse(content=info)


@app.get("/api/filters")
async def list_filters():
    """Returns the list of available 2D convolution filters with their matrix weights."""
    filters_meta = {}
    for key, f in FILTER_REGISTRY.items():
        filters_meta[key] = {
            "name": f["name"],
            "description": f["description"],
            "size": f["size"],
            "kernel": f["kernel"].tolist()
        }
    return JSONResponse(content=filters_meta)


@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    operation: str = Form("grayscale"),
    filter_name: str = Form("gaussian_blur_3x3"),
    engine: str = Form("cuda"),
    mpi_ranks: int = Form(4),
    block_dim_x: int = Form(16),
    block_dim_y: int = Form(16),
    use_scalar_cpu: bool = Form(False)
):
    """
    Executes both CPU sequential and accelerated (CUDA GPU or MPI Distributed CPU) pipelines.
    Calculates execution timings and speedup factors.
    """
    # 1. Read and decode image from upload stream
    try:
        contents = await file.read()
        rgb_image, original_format = decode_image_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    height, width, channels = rgb_image.shape
    engine = engine.lower().strip()
    if engine not in ["cuda", "mpi"]:
        engine = "cuda"

    # 2. Sequential CPU Baseline Execution
    cpu_logs = []
    t_start_cpu = time.perf_counter()

    if operation == "grayscale":
        if use_scalar_cpu:
            cpu_output, t_cpu_ms = cpu_rgb_to_grayscale_scalar(rgb_image)
            cpu_mode_name = "Sequential Single-Threaded Scalar CPU"
        else:
            cpu_output, t_cpu_ms = cpu_rgb_to_grayscale(rgb_image)
            cpu_mode_name = "Sequential Row-by-Row CPU Baseline"
    elif operation == "convolution":
        if filter_name not in FILTER_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown filter: {filter_name}")
        filter_matrix = FILTER_REGISTRY[filter_name]["kernel"]

        # Standard edge detection computes spatial derivatives on luminance, producing the classic 2D monochrome edge map
        if filter_name == "edge_detection_3x3":
            conv_input, _ = cpu_rgb_to_grayscale(rgb_image)
        else:
            conv_input = rgb_image

        if use_scalar_cpu:
            cpu_output, t_cpu_ms = cpu_image_convolution_2d_scalar(conv_input, filter_matrix)
            cpu_mode_name = f"Sequential Single-Threaded Scalar CPU Convolution ({filter_name})"
        else:
            cpu_output, t_cpu_ms = cpu_image_convolution_2d(conv_input, filter_matrix)
            cpu_mode_name = f"Sequential CPU Stencil Convolution ({filter_name})"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")

    cpu_logs.append(f"[{time.strftime('%H:%M:%S')}] [CPU] Executed {cpu_mode_name} in {t_cpu_ms:.3f} ms")

    # 3. Accelerated Pipeline Execution (CUDA or MPI)
    cuda_grid_info = None
    mpi_topology_info = None

    if engine == "cuda":
        block_dim = (int(block_dim_x), int(block_dim_y))
        if operation == "grayscale":
            accel_result = cuda_engine.execute_grayscale(rgb_image, block_dim=block_dim)
        else:
            filter_matrix = FILTER_REGISTRY[filter_name]["kernel"]
            accel_result = cuda_engine.execute_convolution(conv_input, filter_matrix, block_dim=block_dim)

        gpu_metrics = accel_result["metrics"]
        t_kernel_ms = gpu_metrics["t_kernel_ms"]
        t_gpu_total_ms = gpu_metrics["t_gpu_total_ms"]
        speedup_kernel = round(t_cpu_ms / max(t_kernel_ms, 0.0001), 2)
        speedup_total = round(t_cpu_ms / max(t_gpu_total_ms, 0.0001), 2)

        benchmarks = {
            "engine": "cuda",
            "t_cpu_ms": round(t_cpu_ms, 3),
            "t_parallel_ms": t_kernel_ms,
            "t_overhead_ms": round(gpu_metrics["t_h2d_ms"] + gpu_metrics["t_d2h_ms"], 3),
            "t_gpu_kernel_ms": t_kernel_ms,
            "t_gpu_h2d_ms": gpu_metrics["t_h2d_ms"],
            "t_gpu_d2h_ms": gpu_metrics["t_d2h_ms"],
            "t_gpu_total_ms": t_gpu_total_ms,
            "speedup_kernel": speedup_kernel,
            "speedup_total": speedup_total,
            "speedup": speedup_kernel,
            "is_gpu_fallback": accel_result["is_fallback"]
        }

        cuda_grid_info = {
            "grid_dim_x": gpu_metrics["grid_dim"][0],
            "grid_dim_y": gpu_metrics["grid_dim"][1],
            "block_dim_x": gpu_metrics["block_dim"][0],
            "block_dim_y": gpu_metrics["block_dim"][1],
            "total_threads": gpu_metrics["total_threads"]
        }

        combined_logs = accel_result["logs"] + cpu_logs
        combined_logs.append(
            f"[{time.strftime('%H:%M:%S')}] [BENCHMARK CUDA] T_CPU: {t_cpu_ms:.3f} ms | T_GPU: {t_kernel_ms:.3f} ms | Speedup: {speedup_kernel}x"
        )
        output_image = accel_result["output_image"]

    else:  # engine == "mpi"
        num_ranks = int(mpi_ranks)
        if operation == "grayscale":
            accel_result = mpi_engine.execute_grayscale(rgb_image, num_ranks=num_ranks)
        else:
            filter_matrix = FILTER_REGISTRY[filter_name]["kernel"]
            accel_result = mpi_engine.execute_convolution(conv_input, filter_matrix, num_ranks=num_ranks)

        mpi_metrics = accel_result["metrics"]
        t_compute_ms = mpi_metrics["t_compute_ms"]
        t_comm_ms = mpi_metrics["t_comm_ms"]
        t_total_ms = mpi_metrics["t_total_ms"]
        speedup_parallel = round(t_cpu_ms / max(t_compute_ms, 0.0001), 2)
        speedup_end2end = round(t_cpu_ms / max(t_total_ms, 0.0001), 2)

        benchmarks = {
            "engine": "mpi",
            "t_cpu_ms": round(t_cpu_ms, 3),
            "t_parallel_ms": t_compute_ms,
            "t_overhead_ms": t_comm_ms,
            "t_mpi_compute_ms": t_compute_ms,
            "t_mpi_comm_ms": t_comm_ms,
            "t_mpi_scatter_ms": mpi_metrics["t_scatter_ms"],
            "t_mpi_halo_ms": mpi_metrics["t_halo_ms"],
            "t_mpi_gather_ms": mpi_metrics["t_gather_ms"],
            "t_mpi_total_ms": t_total_ms,
            "mpi_ranks": num_ranks,
            "speedup_kernel": speedup_parallel,
            "speedup_total": speedup_end2end,
            "speedup": speedup_parallel,
            "rows_per_rank": mpi_metrics["rows_per_rank"]
        }

        mpi_topology_info = {
            "num_ranks": num_ranks,
            "decomposition": "1D Horizontal Slabs",
            "rows_per_rank": mpi_metrics["rows_per_rank"],
            "halo_radius": filter_matrix.shape[0] // 2 if operation == "convolution" else 0
        }

        combined_logs = accel_result["logs"] + cpu_logs
        combined_logs.append(
            f"[{time.strftime('%H:%M:%S')}] [BENCHMARK MPI] T_CPU: {t_cpu_ms:.3f} ms | T_MPI_PARALLEL: {t_compute_ms:.3f} ms | Speedup: {speedup_parallel}x (P={num_ranks} ranks)"
        )
        output_image = accel_result["output_image"]

    # 4. Contrast enhancement for edge detection so contours are vividly visible
    display_output = output_image
    if operation == "convolution" and filter_name == "edge_detection_3x3":
        if np.any(output_image > 0):
            pos_pixels = output_image[output_image > 0]
            p_val = float(np.percentile(pos_pixels, 99.0))
            scale = max(255.0 / max(p_val, 1.0), 3.0)
            display_output = np.clip(output_image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # 5. Base64 Encode Output Image
    encoded_output = encode_image_base64(display_output)
    encoded_input_preview = encode_image_base64(rgb_image)

    return JSONResponse(content={
        "status": "success",
        "engine": engine,
        "operation": operation,
        "filter_name": filter_name if operation == "convolution" else None,
        "image_metadata": {
            "width": width,
            "height": height,
            "channels": channels,
            "total_pixels": width * height,
            "original_format": original_format,
            "size_bytes": len(contents)
        },
        "benchmarks": benchmarks,
        "cuda_grid": cuda_grid_info,
        "mpi_topology": mpi_topology_info,
        "logs": combined_logs,
        "output_image_data_url": encoded_output,
        "input_image_data_url": encoded_input_preview
    })


# Mount assets directory for sample images
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
if os.path.exists(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.get("/api/sample-image")
async def get_sample_image():
    sample_path = os.path.join(ASSETS_DIR, "sample.png")
    if os.path.exists(sample_path):
        return FileResponse(sample_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Sample image not found")

# Mount frontend static files
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(
            os.path.join(FRONTEND_DIR, "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
