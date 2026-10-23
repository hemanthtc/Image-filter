"""
Module 5: GPU Programming with CUDA - Sequential CPU Baseline Engine

Implements the CPU baseline for:
1. RGB to Grayscale image conversion
2. 2D Image Convolution (Arbitrary odd filter size, e.g., 3x3, 5x5)

Provides millisecond-precision timing (T_CPU) to enable accurate speedup
factor calculation S = T_CPU / T_GPU.
"""

import time
import numpy as np


def cpu_rgb_to_grayscale(rgb_image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Sequential CPU implementation of RGB to Grayscale conversion.
    Identical mathematical formulation:
        Gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    Args:
        rgb_image: 3D NumPy array of shape (height, width, 3) and dtype uint8.
        
    Returns:
        tuple of (gray_image, execution_time_ms):
            gray_image: 2D NumPy array of shape (height, width) and dtype uint8.
            execution_time_ms: Elapsed execution time in milliseconds.
    """
    height, width, _ = rgb_image.shape
    gray_image = np.empty((height, width), dtype=np.uint8)

    start_time = time.perf_counter()

    # Sequential row-by-row iteration matching sequential C/C++ CPU baseline
    r = rgb_image[:, :, 0].astype(np.float32)
    g = rgb_image[:, :, 1].astype(np.float32)
    b = rgb_image[:, :, 2].astype(np.float32)
    
    # Compute luminance and round/clamp to uint8
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    np.clip(gray, 0, 255, out=gray)
    gray_image[:, :] = gray.astype(np.uint8)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return gray_image, elapsed_ms


def cpu_rgb_to_grayscale_scalar(rgb_image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Pure scalar nested-loop CPU implementation (strict single-threaded baseline).
    Used for academic comparison when demonstrating sequential vs massively parallel execution.
    """
    height, width, _ = rgb_image.shape
    gray_image = np.empty((height, width), dtype=np.uint8)

    start_time = time.perf_counter()

    for row in range(height):
        for col in range(width):
            r = float(rgb_image[row, col, 0])
            g = float(rgb_image[row, col, 1])
            b = float(rgb_image[row, col, 2])
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            if lum > 255.0:
                lum = 255.0
            elif lum < 0.0:
                lum = 0.0
            gray_image[row, col] = int(lum)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return gray_image, elapsed_ms


def cpu_image_convolution_2d(image: np.ndarray, filter_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Sequential CPU implementation of 2D Image Convolution.
    Applies an (N x N) stencil across height and width with border clamping.
    
    Args:
        image: 3D NumPy array of shape (height, width, channels) or 2D (height, width).
        filter_matrix: 2D square NumPy array of float32 weights (odd dimensions).
        
    Returns:
        tuple of (output_image, execution_time_ms):
            output_image: NumPy array of identical shape and dtype uint8.
            execution_time_ms: Elapsed execution time in milliseconds.
    """
    is_2d = image.ndim == 2
    if is_2d:
        img_3d = image[:, :, np.newaxis]
    else:
        img_3d = image

    height, width, channels = img_3d.shape
    k_size = filter_matrix.shape[0]
    half_k = k_size // 2

    output = np.empty((height, width, channels), dtype=np.uint8)

    start_time = time.perf_counter()

    # Pad image with replicated edge pixels for clamping behavior
    padded = np.pad(
        img_3d.astype(np.float32),
        ((half_k, half_k), (half_k, half_k), (0, 0)),
        mode='edge'
    )

    # Perform 2D convolution stencil accumulation
    acc = np.zeros((height, width, channels), dtype=np.float32)
    for f_r in range(k_size):
        for f_c in range(k_size):
            weight = filter_matrix[f_r, f_c]
            if weight != 0.0:
                acc += padded[f_r : f_r + height, f_c : f_c + width, :] * weight

    np.clip(acc, 0, 255, out=acc)
    output[:, :, :] = acc.astype(np.uint8)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if is_2d:
        return output[:, :, 0], elapsed_ms
    return output, elapsed_ms
