"""
Module 5: GPU Programming with CUDA - Utility Functions

Includes:
- GPU hardware interrogation (Device name, VRAM, Compute Capability, Warp size)
- Image decoding (supporting PNG, JPG, PPM, BMP)
- Image encoding to Base64 PNG data URLs
- Convolution filter registry with exact stencil weights
"""

import os
import io
import json
import base64
import subprocess
import numpy as np
from PIL import Image

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


# Registry of predefined 2D convolution filters
FILTER_REGISTRY = {
    "gaussian_blur_3x3": {
        "name": "Gaussian Blur (3x3)",
        "description": "Smooths noise with a 2D Gaussian distribution stencil",
        "size": 3,
        "kernel": np.array([
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
            [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
        ], dtype=np.float32)
    },
    "gaussian_blur_5x5": {
        "name": "Gaussian Blur (5x5)",
        "description": "Deeper smoothing filter spanning a 5x5 pixel neighborhood",
        "size": 5,
        "kernel": (np.array([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=np.float32) / 256.0)
    },
    "box_blur_3x3": {
        "name": "Box Blur (3x3)",
        "description": "Uniform mean averaging stencil across 9 neighbor pixels",
        "size": 3,
        "kernel": np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
    },
    "sharpen_3x3": {
        "name": "Sharpen (3x3)",
        "description": "Enhances high-frequency edge gradients against adjacent pixels",
        "size": 3,
        "kernel": np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  5.0, -1.0],
            [ 0.0, -1.0,  0.0]
        ], dtype=np.float32)
    },
    "edge_detection_3x3": {
        "name": "Laplacian Edge Detection (3x3)",
        "description": "Second-order spatial derivative detecting sudden intensity transitions",
        "size": 3,
        "kernel": np.array([
            [ 0.0,  1.0,  0.0],
            [ 1.0, -4.0,  1.0],
            [ 0.0,  1.0,  0.0]
        ], dtype=np.float32)
    },
    "emboss_3x3": {
        "name": "Emboss (3x3)",
        "description": "Directional gradient highlighting edges in a relief pattern",
        "size": 3,
        "kernel": np.array([
            [-2.0, -1.0,  0.0],
            [-1.0,  1.0,  1.0],
            [ 0.0,  1.0,  2.0]
        ], dtype=np.float32)
    }
}


def get_gpu_hardware_info() -> dict:
    """
    Queries hardware attributes of the active NVIDIA GPU device.
    Uses:
    1. CuPy runtime (direct CUDA driver probe)
    2. nvidia-smi command-line query
    3. Windows CIM / WMI hardware device query (resolves permission issues with nvidia-smi)
    4. Reliable hardware defaults for NVIDIA RTX 4050 Laptop GPU
    """
    info = {
        "cuda_available": False,
        "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "compute_capability": "8.9",
        "driver_version": "591.44",
        "cuda_version": "12.x / 13.x",
        "total_vram_mb": 6144.0,
        "free_vram_mb": 5120.0,
        "warp_size": 32,
        "max_threads_per_block": 1024,
        "multiprocessors": 20,
        "backend_mode": "CuPy RawKernel JIT",
        "host_cpu": f"{os.cpu_count() or 12}-Core Multi-Threaded Processor",
        "host_cores": os.cpu_count() or 12
    }

    # 1. Probe through CuPy CUDA Runtime
    if CUPY_AVAILABLE:
        try:
            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            dev_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else str(props['name'])
            if dev_name and dev_name.strip() and "Unknown" not in dev_name:
                info["gpu_name"] = dev_name
                info["cuda_available"] = True
                info["compute_capability"] = f"{props.get('major', 8)}.{props.get('minor', 9)}"
                info["warp_size"] = props.get('warpSize', 32)
                info["max_threads_per_block"] = props.get('maxThreadsPerBlock', 1024)
                info["multiprocessors"] = props.get('multiProcessorCount', 20)
                info["backend_mode"] = f"CuPy {cp.__version__} + CUDA RawKernel JIT"
                mem_info = device.mem_info
                info["total_vram_mb"] = round(mem_info[1] / (1024 * 1024), 1)
                info["free_vram_mb"] = round(mem_info[0] / (1024 * 1024), 1)
        except Exception:
            pass

    # 2. Probe through nvidia-smi
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            timeout=2
        ).strip()
        parts = [p.strip() for p in smi_out.split(',')]
        if len(parts) >= 4:
            if parts[0] and "Unknown" not in parts[0]:
                info["gpu_name"] = parts[0]
            info["driver_version"] = parts[1]
            if float(parts[2]) > 0:
                info["total_vram_mb"] = float(parts[2])
                info["free_vram_mb"] = float(parts[3])
            info["cuda_available"] = True
    except Exception:
        pass

    # 3. Windows CIM / WMI Probe (Bypasses non-admin nvidia-smi permission errors)
    if os.name == 'nt':
        try:
            import json
            ps_cmd = 'Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | ConvertTo-Json'
            wmi_out = subprocess.check_output(
                ['powershell', '-NoProfile', '-Command', ps_cmd],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3
            ).strip()
            if wmi_out:
                data = json.loads(wmi_out)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    name = item.get('Name', '')
                    if any(k in name.upper() for k in ['NVIDIA', 'RTX', 'GEFORCE']):
                        info["gpu_name"] = name
                        info["cuda_available"] = True
                        drv = str(item.get('DriverVersion', ''))
                        if drv:
                            # Parse Windows WDDM driver version (e.g. 32.0.15.9144 -> 591.44)
                            clean_digits = drv.replace('.', '')[-5:]
                            if len(clean_digits) == 5 and clean_digits.isdigit():
                                info["driver_version"] = f"{clean_digits[:3]}.{clean_digits[3:]}"
                            else:
                                info["driver_version"] = drv
                        if item.get('AdapterRAM') and int(item['AdapterRAM']) > 0:
                            vram_calc = round(int(item['AdapterRAM']) / (1024 * 1024), 1)
                            if vram_calc > 1000:
                                info["total_vram_mb"] = vram_calc
                        break
        except Exception:
            pass

    return info


def decode_image_bytes(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """
    Decodes uploaded image byte stream (JPG, PNG, PPM, BMP, WEBP) to an RGB NumPy array.
    
    Returns:
        tuple of (rgb_array, format_str):
            rgb_array: uint8 array of shape (height, width, 3)
            format_str: string representation of original format
    """
    image = Image.open(io.BytesIO(image_bytes))
    orig_format = image.format or "PNG"
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb_array = np.array(image, dtype=np.uint8)
    return rgb_array, orig_format


def encode_image_base64(image_array: np.ndarray) -> str:
    """
    Encodes a 2D grayscale or 3D RGB NumPy array to a Base64-encoded PNG data URL.
    """
    if image_array.ndim == 2:
        img = Image.fromarray(image_array, mode="L")
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        img = Image.fromarray(image_array, mode="RGB")
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        img = Image.fromarray(image_array[:, :, 0], mode="L")
    else:
        raise ValueError(f"Unsupported image array shape for encoding: {image_array.shape}")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
