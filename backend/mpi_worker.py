"""
Standalone MPI Worker Script for Native mpiexec Launches

Usage:
    mpiexec -n 4 python -m backend.mpi_worker --operation grayscale --input assets/sample.png
    mpiexec -n 4 python -m backend.mpi_worker --operation convolution --filter gaussian_blur_3x3 --input assets/sample.png

This script runs genuine MPI via mpi4py (MPI_COMM_WORLD) with:
- MPI_Scatterv for 1D slab decomposition
- MPI_Sendrecv for halo / ghost cell exchange across neighbor ranks
- MPI_Gatherv for master rank image reassembly
"""

import os
import sys
import argparse
import time
import numpy as np
from PIL import Image

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.utils import FILTER_REGISTRY


def run_mpi_native():
    try:
        from mpi4py import MPI
    except ImportError:
        print("[ERROR] mpi4py is required to run this script. Install with: pip install mpi4py")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Native MPI Image Filter")
    parser.add_argument("--operation", choices=["grayscale", "convolution"], default="grayscale")
    parser.add_argument("--filter", default="gaussian_blur_3x3")
    parser.add_argument("--input", default="assets/sample.png")
    parser.add_argument("--output", default="mpi_output.png")
    args = parser.parse_args()

    # Rank 0 loads the input image
    full_image = None
    h, w = 0, 0
    if rank == 0:
        pil_img = Image.open(args.input).convert("RGB")
        full_image = np.array(pil_img, dtype=np.uint8)
        h, w, _ = full_image.shape
        print(f"[MPI MASTER] Loaded '{args.input}' ({w}x{h} px) across {size} MPI ranks.")

    # Broadcast image dimensions to all ranks
    h = comm.bcast(h, root=0)
    w = comm.bcast(w, root=0)

    # Compute slab partition
    base_rows = h // size
    rem = h % size
    row_counts = [base_rows + (1 if r < rem else 0) for r in range(size)]
    row_starts = [sum(row_counts[:r]) for r in range(size)]
    local_rows = row_counts[rank]

    t_start = MPI.Wtime()

    if args.operation == "grayscale":
        # Scatter RGB slices
        local_rgb = np.empty((local_rows, w, 3), dtype=np.uint8)
        # Compute scatter displacements in bytes
        sendcounts = [cnt * w * 3 for cnt in row_counts]
        displs = [st * w * 3 for st in row_starts]
        comm.Scatterv([full_image, sendcounts, displs, MPI.UINT8_T], local_rgb, root=0)

        # Local compute
        t_comp_start = MPI.Wtime()
        r = local_rgb[:, :, 0].astype(np.float32)
        g = local_rgb[:, :, 1].astype(np.float32)
        b = local_rgb[:, :, 2].astype(np.float32)
        local_gray = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)
        t_comp = MPI.Wtime() - t_comp_start

        # Gather results
        output_image = np.empty((h, w), dtype=np.uint8) if rank == 0 else None
        recvcounts = [cnt * w for cnt in row_counts]
        r_displs = [st * w for st in row_starts]
        comm.Gatherv(local_gray, [output_image, recvcounts, r_displs, MPI.UINT8_T], root=0)

    else:
        # 2D Convolution with Halo Exchange
        filter_matrix = FILTER_REGISTRY.get(args.filter, {}).get("kernel", np.ones((3, 3), dtype=np.float32) / 9.0)
        radius = filter_matrix.shape[0] // 2

        # Convert full image to grayscale on rank 0 for convolution
        gray_source = None
        if rank == 0:
            r = full_image[:, :, 0].astype(np.float32)
            g = full_image[:, :, 1].astype(np.float32)
            b = full_image[:, :, 2].astype(np.float32)
            gray_source = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)

        local_interior = np.empty((local_rows, w), dtype=np.uint8)
        sendcounts = [cnt * w for cnt in row_counts]
        displs = [st * w for st in row_starts]
        comm.Scatterv([gray_source, sendcounts, displs, MPI.UINT8_T], local_interior, root=0)

        # Halo Exchange (MPI_Sendrecv)
        top_ghost = np.empty((radius, w), dtype=np.uint8)
        bot_ghost = np.empty((radius, w), dtype=np.uint8)

        # Exchange top halo with rank-1
        if rank > 0:
            comm.Sendrecv(
                sendbuf=local_interior[:radius, :],
                dest=rank - 1,
                sendtag=10,
                recvbuf=top_ghost,
                source=rank - 1,
                recvtag=20
            )
        else:
            top_ghost = np.repeat(local_interior[0:1, :], radius, axis=0)

        # Exchange bottom halo with rank+1
        if rank < size - 1:
            comm.Sendrecv(
                sendbuf=local_interior[-radius:, :],
                dest=rank + 1,
                sendtag=20,
                recvbuf=bot_ghost,
                source=rank + 1,
                recvtag=10
            )
        else:
            bot_ghost = np.repeat(local_interior[-1:, :], radius, axis=0)

        # Build augmented local slab
        aug_slab = np.vstack([top_ghost, local_interior, bot_ghost])

        # Local convolution
        t_comp_start = MPI.Wtime()
        local_out = np.empty((local_rows, w), dtype=np.uint8)
        k_h, k_w = filter_matrix.shape
        f_mat = filter_matrix.astype(np.float32)

        for i in range(local_rows):
            row_center = i + radius
            sum_val = np.zeros(w, dtype=np.float32)
            for ki in range(k_h):
                ky = ki - radius
                slab_y = row_center + ky
                row_data = aug_slab[slab_y, :]
                for kj in range(k_w):
                    kx = kj - radius
                    weight = f_mat[ki, kj]
                    if weight == 0:
                        continue
                    if kx < 0:
                        shifted = np.pad(row_data[:-abs(kx)], (abs(kx), 0), mode='edge')
                    elif kx > 0:
                        shifted = np.pad(row_data[kx:], (0, kx), mode='edge')
                    else:
                        shifted = row_data
                    sum_val += shifted.astype(np.float32) * weight
            np.clip(sum_val, 0, 255, out=sum_val)
            local_out[i, :] = sum_val.astype(np.uint8)

        t_comp = MPI.Wtime() - t_comp_start

        # Gather
        output_image = np.empty((h, w), dtype=np.uint8) if rank == 0 else None
        recvcounts = [cnt * w for cnt in row_counts]
        r_displs = [st * w for st in row_starts]
        comm.Gatherv(local_out, [output_image, recvcounts, r_displs, MPI.UINT8_T], root=0)

    t_total = MPI.Wtime() - t_start

    # Gather timing metrics to Rank 0
    all_comp_times = comm.gather(t_comp, root=0)

    if rank == 0:
        out_pil = Image.fromarray(output_image)
        out_pil.save(args.output)
        print(f"[MPI COMPLETE] Total Execution Time: {t_total * 1000.0:.2f} ms")
        print(f"[MPI COMPLETE] Max Rank Compute Time: {max(all_comp_times) * 1000.0:.2f} ms")
        print(f"[MPI COMPLETE] Saved output to {args.output}")


if __name__ == "__main__":
    run_mpi_native()
