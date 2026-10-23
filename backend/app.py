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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add parent directory to path to support imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.cuda_engine import CUDAEngine
from backend.cpu_engine import (
    cpu_rgb_to_grayscale,
    cpu_rgb_to_grayscale_scalar,
    cpu_image_convolution_2d
)
from backend.utils import (
    get_gpu_hardware_info,
    decode_image_bytes,
    encode_image_base64,
    FILTER_REGISTRY
)

app = FastAPI(
    title="Module 5: CUDA GPU Acceleration Engine",
    description="Heterogeneous Computing benchmark comparing CUDA GPU Kernels vs Sequential CPU",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CUDA Engine
KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
cuda_engine = CUDAEngine(kernels_dir=KERNELS_DIR)


@app.get("/api/device-info")
async def device_info():
    """Returns GPU hardware specifications and driver capabilities."""
    info = get_gpu_hardware_info()
    info["cuda_engine_status"] = "Active" if cuda_engine.cuda_available else "Fallback Mode"
    info["cuda_init_error"] = cuda_engine.init_error
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
    block_dim_x: int = Form(16),
    block_dim_y: int = Form(16),
    use_scalar_cpu: bool = Form(False)
):
    """
    Executes both CPU sequential and GPU CUDA pipelines on the uploaded image.
    Calculates execution timings and speedup factor S = T_CPU / T_GPU.
    """
    # 1. Read and decode image from upload stream
    try:
        contents = await file.read()
        rgb_image, original_format = decode_image_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    height, width, channels = rgb_image.shape
    block_dim = (int(block_dim_x), int(block_dim_y))

    # 2. Sequential CPU Pipeline Execution
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
        cpu_output, t_cpu_ms = cpu_image_convolution_2d(rgb_image, filter_matrix)
        cpu_mode_name = f"Sequential CPU Stencil Convolution ({filter_name})"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")

    cpu_logs.append(f"[{time.strftime('%H:%M:%S')}] [CPU] Executed {cpu_mode_name} in {t_cpu_ms:.3f} ms")

    # 3. Massively Parallel GPU CUDA Pipeline Execution
    if operation == "grayscale":
        gpu_result = cuda_engine.execute_grayscale(rgb_image, block_dim=block_dim)
    else:
        filter_matrix = FILTER_REGISTRY[filter_name]["kernel"]
        gpu_result = cuda_engine.execute_convolution(rgb_image, filter_matrix, block_dim=block_dim)

    # 4. Metrics & Speedup Factor Calculations
    gpu_metrics = gpu_result["metrics"]
    t_kernel_ms = gpu_metrics["t_kernel_ms"]
    t_gpu_total_ms = gpu_metrics["t_gpu_total_ms"]

    # Speedup calculations:
    # Kernel speedup: S_kernel = T_CPU / T_GPU_kernel
    # End-to-end pipeline speedup (including PCI-e Host<->Device memory transfers)
    speedup_kernel = round(t_cpu_ms / max(t_kernel_ms, 0.0001), 2)
    speedup_total = round(t_cpu_ms / max(t_gpu_total_ms, 0.0001), 2)

    # Combined execution timeline logs
    combined_logs = gpu_result["logs"] + cpu_logs
    combined_logs.append(
        f"[{time.strftime('%H:%M:%S')}] [BENCHMARK] T_CPU: {t_cpu_ms:.3f} ms | T_GPU: {t_kernel_ms:.3f} ms | Speedup: {speedup_kernel}x"
    )

    # 5. Base64 Encode Output Image
    encoded_gpu_output = encode_image_base64(gpu_result["output_image"])
    encoded_input_preview = encode_image_base64(rgb_image)

    return JSONResponse(content={
        "status": "success",
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
        "benchmarks": {
            "t_cpu_ms": round(t_cpu_ms, 3),
            "t_gpu_kernel_ms": t_kernel_ms,
            "t_gpu_h2d_ms": gpu_metrics["t_h2d_ms"],
            "t_gpu_d2h_ms": gpu_metrics["t_d2h_ms"],
            "t_gpu_total_ms": t_gpu_total_ms,
            "speedup_kernel": speedup_kernel,
            "speedup_total": speedup_total,
            "is_gpu_fallback": gpu_result["is_fallback"]
        },
        "cuda_grid": {
            "grid_dim_x": gpu_metrics["grid_dim"][0],
            "grid_dim_y": gpu_metrics["grid_dim"][1],
            "block_dim_x": gpu_metrics["block_dim"][0],
            "block_dim_y": gpu_metrics["block_dim"][1],
            "total_threads": gpu_metrics["total_threads"]
        },
        "logs": combined_logs,
        "output_image_data_url": encoded_gpu_output,
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
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
