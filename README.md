# CUDA Image Accelerator: Module 5 Heterogeneous Computing Demo

A full-stack, high-performance college project demonstrating **Module 5: GPU Programming with CUDA (Heterogeneous Computing)**.

The application accelerates 2D image processing pipelines (RGB to Grayscale conversion and 2D Stencil Convolution) using genuine CUDA C++ kernels running on an NVIDIA GPU (tested on **NVIDIA GeForce RTX 4050 Laptop GPU**), with sequential CPU baselines, microsecond-precision benchmarks, and an interactive dark-mode dashboard.

---

## ⚡ Quickstart (One-Click Launch)

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10+ (tested on Python 3.13)
- NVIDIA GPU with modern drivers (RTX / GTX series)

### 1. Launch Options
- **In PowerShell:** Run:
  ```powershell
  .\run.ps1
  # or
  .\run.bat
  ```
- **In Command Prompt (cmd.exe):** Run:
  ```cmd
  run.bat
  ```
- **In Windows File Explorer:** Simply double-click `run.bat`.

The launcher starts the FastAPI ASGI server and automatically opens `http://127.0.0.1:8000` in your web browser.

### 2. Or Launch Manually via Terminal
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start FastAPI ASGI server
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Then navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🎓 Syllabus Alignment (Module 5 Concepts)

| Module 5 & HPC Concept | Implementation in this Project | Source Code Reference |
|---|---|---|
| **Heterogeneous Architecture** | Separation of Host RAM (CPU) and Device VRAM (NVIDIA GPU) | [`backend/cuda_engine.py`](file:///backend/cuda_engine.py) |
| **CUDA C++ Kernel Definition** | Real `__global__` functions written in standard CUDA C++ | [`backend/kernels/grayscale.cu`](file:///backend/kernels/grayscale.cu)<br>[`backend/kernels/convolution.cu`](file:///backend/kernels/convolution.cu) |
| **2D Grid & Block Layout** | `dim3 grid(ceil(W/Bx), ceil(H/By))`, `dim3 block(Bx, By)` | [`backend/cuda_engine.py`](file:///backend/cuda_engine.py) |
| **Global Thread Mapping** | `col = blockIdx.x * blockDim.x + threadIdx.x`<br>`row = blockIdx.y * blockDim.y + threadIdx.y` | [`backend/kernels/grayscale.cu`](file:///backend/kernels/grayscale.cu) |
| **Boundary Condition Checking** | `if (col < width && row < height)` prevents illegal memory access | [`backend/kernels/grayscale.cu`](file:///backend/kernels/grayscale.cu) |
| **Host $\leftrightarrow$ Device Memory Transfers** | `cudaMemcpyHostToDevice` ($H \to D$) and `cudaMemcpyDeviceToHost` ($D \to H$) | [`backend/cuda_engine.py`](file:///backend/cuda_engine.py) |
| **Distributed Memory (MPI)** | 1D Horizontal Slab Domain Decomposition ($H/P$ rows per rank) | [`backend/mpi_engine.py`](file:///backend/mpi_engine.py) |
| **Halo / Ghost Cell Exchange** | Neighboring rank boundary row exchange (`MPI_Sendrecv`) for stencils | [`backend/mpi_engine.py`](file:///backend/mpi_engine.py) |
| **Scatter & Gather Protocol** | Master rank distribution (`MPI_Scatterv`) and assembly (`MPI_Gatherv`) | [`backend/mpi_engine.py`](file:///backend/mpi_engine.py)<br>[`backend/mpi_worker.py`](file:///backend/mpi_worker.py) |
| **High-Precision Timing** | GPU hardware timing via `cudaEventRecord` / `cudaEventElapsedTime` | [`backend/cuda_engine.py`](file:///backend/cuda_engine.py) |
| **Speedup Factor Formulation** | 3-way empirical speedup $S_{CUDA} = T_{CPU}/T_{GPU}$ vs $S_{MPI} = T_{CPU}/T_{MPI}$ | [`backend/app.py`](file:///backend/app.py) |

---

## 🖥️ Project Architecture

```
c:\Users\cheta\Downloads\hemanth\pc\
├── backend/
│   ├── app.py               # FastAPI server (API endpoints, static files, device telemetry)
│   ├── cuda_engine.py       # CuPy RawKernel runtime, Host/Device lifecycle, CUDA Event timing
│   ├── mpi_engine.py        # MPI 1D Domain Decomposition, Ghost cell halo exchange, rank pool
│   ├── mpi_worker.py        # Standalone MPI worker script for mpiexec launches
│   ├── cpu_engine.py        # Sequential CPU baseline algorithms (identical math formulations)
│   ├── kernels/
│   │   ├── grayscale.cu     # Standalone CUDA C++ source for RGB to Grayscale
│   │   └── convolution.cu   # Standalone CUDA C++ source for 2D Stencil Convolution
│   └── utils.py             # Multi-layer GPU/MPI hardware probe, Base64 codecs, filter registry
├── frontend/
│   ├── index.html           # Dark-mode dashboard with CUDA/MPI switcher, telemetry, visual cards
│   ├── css/
│   │   └── style.css        # Glassmorphism aesthetic, responsive design, glowing metrics
│   ├── js/
│   │   └── app.js           # Engine switcher, drag & drop upload, real-time log streaming
│   └── images/
│       └── placeholder.svg  # Visual placeholder SVG
├── assets/
│   └── sample.png           # Built-in 800x600 benchmark pattern for 1-click execution
├── tests/
│   └── test_pipeline.py     # Automated verification suite (7/7 tests passing)
├── requirements.txt         # Pinned runtime dependencies
├── run.bat                  # Windows batch launcher (CMD / File Explorer)
├── run.ps1                  # Windows PowerShell launcher
├── README.md                # Quickstart and overview documentation
└── report.md                # Detailed academic technical report
```

---

## 📊 Dashboard Features

1. **Hardware Telemetry Banner:** Automatically queries and displays active GPU name (e.g. `NVIDIA GeForce RTX 4050 Laptop GPU`), VRAM, and driver version.
2. **Interactive Input Section:** Drag-and-drop or file picker supporting PNG, JPG, PPM, BMP, WEBP, with a "Load Built-in Sample Image" button for instant testing.
3. **Algorithm & Stencil Controls:**
   - **RGB to Grayscale:** Standard NTSC luminance conversion.
   - **2D Image Convolution:** Selectable filters (Gaussian Blur 3×3/5×5, Box Blur, Edge Detection, Sharpen, Emboss) with live stencil matrix weight preview.
4. **CUDA Block Configuration:** Switch between $16 \times 16$, $32 \times 32$, and $8 \times 8$ blocks with dynamic real-time grid dimension calculator.
5. **Side-by-Side Visual Verification:** Host Input Image vs. Device GPU Output Image with one-click download.
6. **Benchmark Metrics Cards:**
   - Sequential CPU Execution Time ($T_{CPU}$ in ms)
   - Massively Parallel GPU Execution Time ($T_{GPU}$ in ms)
   - Speedup Factor ($S = T_{CPU} / T_{GPU}$ highlighted in neon green)
   - PCIe Memory Transfer Overhead ($T_{H2D} + T_{D2H}$ in ms)
7. **Live Execution Terminal Console:** Embedded UI terminal logging every step of the heterogeneous lifecycle, mirrored directly to the browser DevTools console with styled logs.

---

## 📄 Complete Project Report
For the complete technical report covering hardware specifications, mathematical derivations, CUDA memory hierarchies, and benchmark tables, read **[`report.md`](file:///report.md)**.
