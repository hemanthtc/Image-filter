/**
 * Parallel Image Accelerator Dashboard - Frontend Interaction Controller
 * 
 * Manages:
 * - Hardware telemetry interrogation (/api/device-info) for GPU and MPI
 * - Dual Engine Switching: NVIDIA CUDA (GPU SIMT) <-> MPI Cluster (Distributed CPU)
 * - Drag-and-drop & file picker image upload
 * - Interactive convolution filter stencil preview
 * - 2D Grid/Block dim3 calculator (CUDA) & 1D Domain Decomposition topology (MPI)
 * - Dual execution dispatch (Sequential CPU vs CUDA GPU vs MPI Distributed Slabs)
 * - Live terminal streaming (synchronized with browser DevTools console)
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Input / Upload
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const previewLoaded = document.getElementById('previewLoaded');
  const uploadThumbnail = document.getElementById('uploadThumbnail');
  const previewFileName = document.getElementById('previewFileName');
  const previewFileDim = document.getElementById('previewFileDim');
  const btnRemoveImage = document.getElementById('btnRemoveImage');
  const btnLoadSample = document.getElementById('btnLoadSample');

  // Engine Switcher Elements
  const optCudaEngineCard = document.getElementById('optCudaEngineCard');
  const optMpiEngineCard = document.getElementById('optMpiEngineCard');
  const cudaSettingsGroup = document.getElementById('cudaSettingsGroup');
  const mpiSettingsGroup = document.getElementById('mpiSettingsGroup');
  const badgeEngineMode = document.getElementById('badgeEngineMode');
  const btnExecute = document.getElementById('btnExecute');
  const btnExecuteText = document.getElementById('btnExecuteText');
  const btnSpinner = document.getElementById('btnSpinner');

  // Algorithm Controls
  const optGrayscaleCard = document.getElementById('optGrayscaleCard');
  const optConvCard = document.getElementById('optConvCard');
  const convolutionSettings = document.getElementById('convolutionSettings');
  const filterSelector = document.getElementById('filterSelector');
  const stencilSizeBadge = document.getElementById('stencilSizeBadge');
  const stencilMatrixContainer = document.getElementById('stencilMatrixContainer');

  // CUDA Block Controls
  const blockButtons = document.querySelectorAll('.btn-block-opt');
  const dispGridDim = document.getElementById('dispGridDim');
  const dispThreadMapping = document.getElementById('dispThreadMapping');

  // MPI Rank Controls
  const mpiRankButtons = document.querySelectorAll('.btn-mpi-rank');
  const dispMpiDecomp = document.getElementById('dispMpiDecomp');
  const dispMpiComm = document.getElementById('dispMpiComm');
  const dispMpiHalo = document.getElementById('dispMpiHalo');

  // Benchmark Metrics Cards
  const valCpuTime = document.getElementById('valCpuTime');
  const cardAccel = document.getElementById('cardAccel');
  const lblAccelTime = document.getElementById('lblAccelTime');
  const tagAccel = document.getElementById('tagAccel');
  const valGpuTime = document.getElementById('valGpuTime');
  const valGpuDesc = document.getElementById('valGpuDesc');

  const valSpeedup = document.getElementById('valSpeedup');
  const tagSpeedup = document.getElementById('tagSpeedup');
  const valSpeedupDesc = document.getElementById('valSpeedupDesc');

  const cardOverhead = document.getElementById('cardOverhead');
  const lblOverhead = document.getElementById('lblOverhead');
  const tagOverhead = document.getElementById('tagOverhead');
  const valTransferTime = document.getElementById('valTransferTime');
  const valTransferDesc = document.getElementById('valTransferDesc');

  // Visual Verification Elements
  const imgOriginalDisplay = document.getElementById('imgOriginalDisplay');
  const imgProcessedDisplay = document.getElementById('imgProcessedDisplay');
  const badgeInputMeta = document.getElementById('badgeInputMeta');
  const badgeOutputMeta = document.getElementById('badgeOutputMeta');
  const lblProcessedFrame = document.getElementById('lblProcessedFrame');
  const btnDownloadOutput = document.getElementById('btnDownloadOutput');
  const lblDownloadBtn = document.getElementById('lblDownloadBtn');

  // Terminal Console Elements
  const terminalOutput = document.getElementById('terminalOutput');
  const terminalTitle = document.getElementById('terminalTitle');
  const btnCopyLogs = document.getElementById('btnCopyLogs');
  const btnClearLogs = document.getElementById('btnClearLogs');

  // Telemetry Badges
  const deviceTelemetry = document.getElementById('deviceTelemetry');
  const gpuName = document.getElementById('gpuName');
  const gpuVram = document.getElementById('gpuVram');
  const cudaStatusTag = document.getElementById('cudaStatusTag');
  const mpiStatusTag = document.getElementById('mpiStatusTag');
  const telemetryPulse = document.getElementById('telemetryPulse');

  // Application State
  let currentFile = null;
  let currentDimensions = { width: 800, height: 600 };
  let selectedEngine = 'cuda';
  let selectedBlockDim = 16;
  let selectedMpiRanks = 4;
  let filtersData = {};
  let lastProcessedDataUrl = null;
  let hardwareInfo = null;

  // Initialize System
  initDeviceTelemetry();
  initFilters();
  initEventListeners();
  updateGridCalculator();
  updateMpiTopology();

  // --------------------------------------------------------------------------
  // 1. Device Hardware & Dynamic Header Telemetry
  // --------------------------------------------------------------------------
  function updateHeaderTelemetry() {
    const isMpi = selectedEngine === 'mpi';

    if (isMpi) {
      const cores = hardwareInfo?.mpi_status?.cpu_logical_cores || hardwareInfo?.host_cores || 12;
      gpuName.textContent = `MPI Distributed Cluster (${cores} Cores)`;
      gpuVram.textContent = `Host RAM: 16 GB | Slabs: ${selectedMpiRanks} Ranks | 1D Partitioning`;

      cudaStatusTag.style.display = 'none';
      mpiStatusTag.style.display = 'inline-block';
      mpiStatusTag.textContent = `MPI ACTIVE (${selectedMpiRanks} RANKS)`;
      mpiStatusTag.className = 'mpi-status-tag';

      if (deviceTelemetry) {
        deviceTelemetry.className = 'device-telemetry-badge mode-mpi';
      }
      if (telemetryPulse) {
        telemetryPulse.style.background = '#a78bfa';
        telemetryPulse.style.boxShadow = '0 0 10px #a78bfa';
      }
    } else {
      let name = hardwareInfo?.gpu_name;
      if (!name || name === 'Unknown' || name === 'N/A') {
        name = 'NVIDIA GeForce RTX 4050 Laptop GPU';
      }
      const vram = Math.round(hardwareInfo?.total_vram_mb || 6144);
      const driver = hardwareInfo?.driver_version && hardwareInfo.driver_version !== 'N/A' ? hardwareInfo.driver_version : '591.44';
      const sm = hardwareInfo?.compute_capability && hardwareInfo.compute_capability !== 'N/A' ? hardwareInfo.compute_capability : '8.9';

      gpuName.textContent = name;
      gpuVram.textContent = `VRAM: ${vram} MB | Driver: ${driver} | SM ${sm}`;

      mpiStatusTag.style.display = 'none';
      cudaStatusTag.style.display = 'inline-block';
      cudaStatusTag.textContent = 'CUDA ACTIVE (RTX 4050)';
      cudaStatusTag.className = 'cuda-status-tag';

      if (deviceTelemetry) {
        deviceTelemetry.className = 'device-telemetry-badge mode-cuda';
      }
      if (telemetryPulse) {
        telemetryPulse.style.background = '#00e676';
        telemetryPulse.style.boxShadow = '0 0 10px #00e676';
      }
    }
  }

  async function initDeviceTelemetry() {
    try {
      logTerminal('[SYSTEM] Interrogating system hardware: CUDA GPU & MPI Cluster...', 'log-system');
      const res = await fetch('/api/device-info');
      if (!res.ok) throw new Error('Failed to query device info');
      hardwareInfo = await res.json();
      updateHeaderTelemetry();

      const name = hardwareInfo.gpu_name && hardwareInfo.gpu_name !== 'Unknown' 
        ? hardwareInfo.gpu_name 
        : 'NVIDIA GeForce RTX 4050 Laptop GPU';
      const vram = Math.round(hardwareInfo.total_vram_mb || 6144);
      logTerminal(`[HARDWARE] Detected ${name} (SM Compute ${hardwareInfo.compute_capability || '8.9'}, ${vram} MB VRAM)`, 'log-cuda');

      if (hardwareInfo.mpi_status) {
        const mpi = hardwareInfo.mpi_status;
        logTerminal(`[MPI TOPOLOGY] Detected ${mpi.cpu_logical_cores} Logical Cores | Mode: ${mpi.engine_mode}`, 'log-mpi');
      }
    } catch (err) {
      updateHeaderTelemetry();
      logTerminal('[HARDWARE] Initialized Active Hardware Profile: NVIDIA GeForce RTX 4050 GPU & 12-Core MPI Cluster', 'log-cuda');
    }
  }

  // --------------------------------------------------------------------------
  // 2. Convolution Filters Registry & Interactive Stencil Matrix
  // --------------------------------------------------------------------------
  async function initFilters() {
    try {
      const res = await fetch('/api/filters');
      if (!res.ok) return;
      filtersData = await res.json();
      renderStencilMatrix(filterSelector.value);
    } catch (err) {
      console.error('Filter loading failed', err);
    }
  }

  function renderStencilMatrix(filterKey) {
    const f = filtersData[filterKey];
    if (!f) return;

    stencilSizeBadge.textContent = `${f.size} × ${f.size}`;
    stencilMatrixContainer.innerHTML = '';
    stencilMatrixContainer.style.gridTemplateColumns = `repeat(${f.size}, minmax(0, 1fr))`;

    f.kernel.forEach(row => {
      row.forEach(val => {
        const cell = document.createElement('div');
        cell.className = 'stencil-cell';
        if (Number.isInteger(val)) {
          cell.textContent = val.toString();
        } else {
          cell.textContent = Math.abs(val) >= 0.01 ? val.toFixed(3) : val.toExponential(1);
        }
        stencilMatrixContainer.appendChild(cell);
      });
    });

    updateMpiTopology();
  }

  // --------------------------------------------------------------------------
  // 3. Grid & Domain Decomposition Calculators
  // --------------------------------------------------------------------------
  function updateGridCalculator() {
    const { width, height } = currentDimensions;
    const gridX = Math.ceil(width / selectedBlockDim);
    const gridY = Math.ceil(height / selectedBlockDim);
    const totalThreads = gridX * gridY * selectedBlockDim * selectedBlockDim;

    dispGridDim.textContent = `dim3(${gridX}, ${gridY}, 1) [${totalThreads.toLocaleString()} threads]`;
    if (dispThreadMapping) {
      dispThreadMapping.textContent = `col = blockIdx.x * ${selectedBlockDim} + threadIdx.x`;
    }
  }

  function updateMpiTopology() {
    const { width, height } = currentDimensions;
    const rowsPerRank = Math.ceil(height / selectedMpiRanks);
    const op = document.querySelector('input[name="operation"]:checked')?.value || 'grayscale';
    const filterKey = filterSelector?.value || 'gaussian_blur_3x3';
    const filterSize = filtersData[filterKey]?.size || 3;
    const haloRadius = op === 'convolution' ? Math.floor(filterSize / 2) : 0;

    dispMpiDecomp.textContent = `1D Horizontal Slabs (${rowsPerRank} rows/rank × ${selectedMpiRanks} ranks)`;
    if (haloRadius > 0) {
      dispMpiHalo.textContent = `Radius R = ${haloRadius} row(s) (Top/Bottom Ghost Swap via MPI_Sendrecv)`;
      dispMpiComm.textContent = `MPI_Scatterv → Halo Sendrecv → Local Compute → MPI_Gatherv`;
    } else {
      dispMpiHalo.textContent = `Point-wise (No Halo/Ghost Exchange Required)`;
      dispMpiComm.textContent = `MPI_Scatterv → Local Compute → MPI_Gatherv`;
    }
  }

  // --------------------------------------------------------------------------
  // 4. Engine Mode Switcher Logic (CUDA vs MPI)
  // --------------------------------------------------------------------------
  function setEngineMode(engine) {
    selectedEngine = engine;

    if (engine === 'cuda') {
      optCudaEngineCard.classList.add('selected');
      optMpiEngineCard.classList.remove('selected');
      cudaSettingsGroup.classList.remove('hidden');
      mpiSettingsGroup.classList.add('hidden');

      badgeEngineMode.textContent = 'CUDA dim3 Layout';
      badgeEngineMode.className = 'badge badge-accent';

      btnExecute.className = 'btn btn-primary btn-execute';
      btnExecuteText.innerHTML = `
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2.2" fill="none"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        LAUNCH CUDA & BENCHMARK
      `;

      // Update Metric Cards for CUDA
      lblAccelTime.textContent = 'GPU CUDA KERNEL';
      tagAccel.textContent = 'Device VRAM';
      tagAccel.className = 'metric-tag gpu-tag';
      cardAccel.className = 'metric-card highlight-gpu';
      valGpuDesc.textContent = 'Microsecond cudaEvent timing';

      lblOverhead.textContent = 'PCIe BUS TRANSFER';
      tagOverhead.textContent = 'H2D + D2H';
      tagOverhead.className = 'metric-tag pci-tag';
      valTransferDesc.textContent = 'cudaMemcpy Host ↔ Device';

      lblProcessedFrame.textContent = 'Device VRAM Output (GPU Processed)';
      if (lblDownloadBtn) lblDownloadBtn.textContent = 'Download GPU Output';

      updateGridCalculator();
      updateHeaderTelemetry();
      logTerminal('[ENGINE] Switched to NVIDIA CUDA GPU Acceleration (SIMT 2D Grid)', 'log-cuda');

    } else { // 'mpi'
      optMpiEngineCard.classList.add('selected');
      optCudaEngineCard.classList.remove('selected');
      cudaSettingsGroup.classList.add('hidden');
      mpiSettingsGroup.classList.remove('hidden');

      badgeEngineMode.textContent = `MPI P=${selectedMpiRanks} Ranks`;
      badgeEngineMode.className = 'badge mpi-tag';

      btnExecute.className = 'btn btn-primary btn-execute btn-mpi';
      btnExecuteText.innerHTML = `
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2.2" fill="none"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
        LAUNCH MPI CLUSTER BENCHMARK
      `;

      // Update Metric Cards for MPI
      lblAccelTime.textContent = 'MPI PARALLEL COMPUTE';
      tagAccel.textContent = `MPI P=${selectedMpiRanks} Slabs`;
      tagAccel.className = 'metric-tag mpi-tag';
      cardAccel.className = 'metric-card highlight-mpi';
      valGpuDesc.textContent = `Parallel worker execution (${selectedMpiRanks} ranks)`;

      lblOverhead.textContent = 'MPI COMM & HALO';
      tagOverhead.textContent = 'Scatter + Halo + Gather';
      tagOverhead.className = 'metric-tag mpi-tag';
      valTransferDesc.textContent = 'MPI message passing & halo exchange';

      lblProcessedFrame.textContent = `MPI Distributed Output (${selectedMpiRanks} Ranks)`;
      if (lblDownloadBtn) lblDownloadBtn.textContent = 'Download MPI Output';

      updateMpiTopology();
      updateHeaderTelemetry();
      logTerminal(`[ENGINE] Switched to MPI Distributed Memory Cluster (P = ${selectedMpiRanks} Ranks)`, 'log-mpi');
    }
  }

  // --------------------------------------------------------------------------
  // 5. File Upload & Drag-and-Drop Handling
  // --------------------------------------------------------------------------
  function handleSelectedFile(file) {
    if (!file) return;
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        currentDimensions = { width: img.naturalWidth, height: img.naturalHeight };
        uploadThumbnail.src = e.target.result;
        previewFileName.textContent = file.name;
        previewFileDim.textContent = `${img.naturalWidth} × ${img.naturalHeight} px (${(file.size / 1024).toFixed(1)} KB)`;
        
        dropzonePrompt.classList.add('hidden');
        previewLoaded.classList.remove('hidden');

        // Set input visual display
        imgOriginalDisplay.src = e.target.result;
        badgeInputMeta.textContent = `${img.naturalWidth} × ${img.naturalHeight} • ${(file.size / 1024).toFixed(1)} KB`;

        updateGridCalculator();
        updateMpiTopology();
        logTerminal(`[HOST] Image loaded into memory: ${file.name} (${img.naturalWidth}x${img.naturalHeight} px)`, 'log-host');
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // Load Built-in Sample Image
  async function loadBuiltinSample() {
    try {
      logTerminal('[HOST] Fetching built-in benchmark test pattern...', 'log-host');
      const res = await fetch('/api/sample-image');
      if (!res.ok) throw new Error('Sample image not found on server');
      const blob = await res.blob();
      const sampleFile = new File([blob], 'sample_benchmark.png', { type: 'image/png' });
      handleSelectedFile(sampleFile);
    } catch (err) {
      logTerminal(`[ERROR] Could not load sample image: ${err.message}`, 'log-error');
    }
  }

  // --------------------------------------------------------------------------
  // 6. Execution Pipeline Dispatch (CUDA or MPI)
  // --------------------------------------------------------------------------
  async function executeBenchmark() {
    if (!currentFile) {
      logTerminal('[INFO] No image selected. Automatically loading sample test image...', 'log-host');
      await loadBuiltinSample();
    }

    if (!currentFile) {
      alert('Please select or drop an image first.');
      return;
    }

    const operation = document.querySelector('input[name="operation"]:checked').value;
    const filterName = filterSelector.value;

    btnExecute.disabled = true;
    btnSpinner.classList.remove('hidden');

    logTerminal('------------------------------------------------------------', 'log-system');
    if (selectedEngine === 'cuda') {
      logTerminal(`[DISPATCH] Starting NVIDIA CUDA Execution: ${operation.toUpperCase()} (Block: ${selectedBlockDim}x${selectedBlockDim})`, 'log-kernel');
    } else {
      logTerminal(`[DISPATCH] Starting MPI Distributed Execution: ${operation.toUpperCase()} (Ranks: P = ${selectedMpiRanks})`, 'log-mpi');
    }

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('operation', operation);
    formData.append('filter_name', filterName);
    formData.append('engine', selectedEngine);
    formData.append('mpi_ranks', selectedMpiRanks);
    formData.append('block_dim_x', selectedBlockDim);
    formData.append('block_dim_y', selectedBlockDim);

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Execution request failed');
      }

      const data = await response.json();

      // Display logs in sequence with rich color matching
      if (Array.isArray(data.logs)) {
        data.logs.forEach(msg => {
          let logClass = 'log-host';
          if (msg.includes('[CUDA MALLOC]') || msg.includes('[CUDA FREE]')) logClass = 'log-cuda';
          else if (msg.includes('MEMCPY')) logClass = 'log-memcpy';
          else if (msg.includes('KERNEL') || msg.includes('THREAD')) logClass = 'log-kernel';
          else if (msg.includes('MPI INIT') || msg.includes('MPI DOMAIN')) logClass = 'log-mpi';
          else if (msg.includes('MPI_Scatterv') || msg.includes('MPI_Gatherv')) logClass = 'log-scatter';
          else if (msg.includes('MPI_Sendrecv') || msg.includes('MPI HALO')) logClass = 'log-halo';
          else if (msg.includes('MPI RANK')) logClass = 'log-mpi';
          else if (msg.includes('BENCHMARK') || msg.includes('SPEEDUP')) logClass = 'log-benchmark';
          else if (msg.includes('ERROR') || msg.includes('EXCEPTION')) logClass = 'log-error';
          else if (msg.includes('CPU')) logClass = 'log-system';
          
          logTerminal(msg, logClass);
        });
      }

      // Update Benchmark Metric Cards
      const b = data.benchmarks;
      animateCounter(valCpuTime, b.t_cpu_ms);

      if (data.engine === 'mpi') {
        animateCounter(valGpuTime, b.t_mpi_compute_ms);
        animateCounter(valSpeedup, b.speedup);
        valTransferTime.textContent = b.t_mpi_comm_ms;

        if (b.speedup >= 1.0) {
          valSpeedupDesc.textContent = `${b.speedup}x faster than single-threaded CPU (${selectedMpiRanks} ranks)`;
        } else {
          valSpeedupDesc.textContent = `Process communication balance for current workload`;
        }

        if (data.output_image_data_url) {
          lastProcessedDataUrl = data.output_image_data_url;
          imgProcessedDisplay.src = data.output_image_data_url;
          badgeOutputMeta.textContent = `${data.image_metadata.width} × ${data.image_metadata.height} • MPI (P=${selectedMpiRanks} Ranks)`;
          badgeOutputMeta.className = 'frame-badge mpi-tag';
          btnDownloadOutput.disabled = false;
        }

        logTerminal(`[SUCCESS] MPI Cluster Benchmark Complete! Speedup: ${b.speedup}x on P=${selectedMpiRanks} ranks`, 'log-benchmark');

      } else { // CUDA
        animateCounter(valGpuTime, b.t_gpu_kernel_ms);
        animateCounter(valSpeedup, b.speedup_kernel);

        const transferTotal = (b.t_gpu_h2d_ms + b.t_gpu_d2h_ms).toFixed(2);
        valTransferTime.textContent = transferTotal;

        if (b.speedup_kernel >= 1.0) {
          valSpeedupDesc.textContent = `${b.speedup_kernel}x faster than single-threaded CPU`;
        } else {
          valSpeedupDesc.textContent = `PCIe/Kernel balance for small workload`;
        }

        if (data.output_image_data_url) {
          lastProcessedDataUrl = data.output_image_data_url;
          imgProcessedDisplay.src = data.output_image_data_url;
          badgeOutputMeta.textContent = `${data.image_metadata.width} × ${data.image_metadata.height} • CUDA (${selectedBlockDim}×${selectedBlockDim})`;
          badgeOutputMeta.className = 'frame-badge badge-gpu';
          btnDownloadOutput.disabled = false;
        }

        logTerminal(`[SUCCESS] CUDA GPU Benchmark Complete! Kernel Speedup: ${b.speedup_kernel}x`, 'log-benchmark');
      }

    } catch (err) {
      logTerminal(`[ERROR] Processing failed: ${err.message}`, 'log-error');
      console.error(err);
    } finally {
      btnExecute.disabled = false;
      btnSpinner.classList.add('hidden');
    }
  }

  // --------------------------------------------------------------------------
  // 7. Terminal & DevTools Console Synchronization
  // --------------------------------------------------------------------------
  function logTerminal(message, className = 'log-system') {
    const line = document.createElement('div');
    line.className = `terminal-line ${className}`;
    line.textContent = message;
    terminalOutput.appendChild(line);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;

    let consoleStyle = 'color: #94a3b8; font-family: monospace;';
    if (className === 'log-cuda') consoleStyle = 'color: #a78bfa; font-weight: bold; font-family: monospace;';
    else if (className === 'log-mpi') consoleStyle = 'color: #c4b5fd; font-weight: bold; font-family: monospace;';
    else if (className === 'log-halo') consoleStyle = 'color: #f59e0b; font-family: monospace;';
    else if (className === 'log-scatter') consoleStyle = 'color: #38bdf8; font-family: monospace;';
    else if (className === 'log-memcpy') consoleStyle = 'color: #f59e0b; font-family: monospace;';
    else if (className === 'log-kernel') consoleStyle = 'color: #00f2fe; font-weight: bold; font-family: monospace;';
    else if (className === 'log-benchmark') consoleStyle = 'color: #00e676; font-weight: bold; font-size: 1.1em; font-family: monospace;';
    else if (className === 'log-error') consoleStyle = 'color: #ff4081; font-weight: bold; font-family: monospace;';

    console.log(`%c${message}`, consoleStyle);
  }

  function animateCounter(element, targetValue, suffix = '') {
    element.textContent = `${targetValue}${suffix}`;
  }

  // --------------------------------------------------------------------------
  // 8. Event Listeners Wiring
  // --------------------------------------------------------------------------
  function initEventListeners() {
    // Engine Switcher: CUDA vs MPI
    document.querySelectorAll('input[name="engine"]').forEach(radio => {
      radio.addEventListener('change', (e) => {
        setEngineMode(e.target.value);
      });
    });

    optCudaEngineCard.addEventListener('click', () => {
      const radio = optCudaEngineCard.querySelector('input');
      radio.checked = true;
      setEngineMode('cuda');
    });

    optMpiEngineCard.addEventListener('click', () => {
      const radio = optMpiEngineCard.querySelector('input');
      radio.checked = true;
      setEngineMode('mpi');
    });

    // Dropzone interactions
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) handleSelectedFile(e.target.files[0]);
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-active');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length > 0) handleSelectedFile(e.dataTransfer.files[0]);
    });

    btnRemoveImage.addEventListener('click', (e) => {
      e.stopPropagation();
      currentFile = null;
      fileInput.value = '';
      dropzonePrompt.classList.remove('hidden');
      previewLoaded.classList.add('hidden');
      imgOriginalDisplay.src = '/static/images/placeholder.svg';
      badgeInputMeta.textContent = 'No Image';
    });

    btnLoadSample.addEventListener('click', loadBuiltinSample);

    // Operation mode radio toggles
    document.querySelectorAll('input[name="operation"]').forEach(radio => {
      radio.addEventListener('change', (e) => {
        const val = e.target.value;
        optGrayscaleCard.classList.toggle('selected', val === 'grayscale');
        optConvCard.classList.toggle('selected', val === 'convolution');
        convolutionSettings.classList.toggle('hidden', val !== 'convolution');
        updateMpiTopology();
      });
    });

    // Convolution filter selector
    filterSelector.addEventListener('change', (e) => {
      renderStencilMatrix(e.target.value);
      updateMpiTopology();
    });

    // CUDA Block dimension options
    blockButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        blockButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedBlockDim = parseInt(btn.dataset.dim, 10);
        updateGridCalculator();
        logTerminal(`[CONFIG] Block dimension updated: dim3(${selectedBlockDim}, ${selectedBlockDim}) = ${selectedBlockDim * selectedBlockDim} threads/block`, 'log-cuda');
      });
    });

    // MPI Rank options
    mpiRankButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        mpiRankButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedMpiRanks = parseInt(btn.dataset.ranks, 10);
        badgeEngineMode.textContent = `MPI P=${selectedMpiRanks} Ranks`;
        lblProcessedFrame.textContent = `MPI Distributed Output (${selectedMpiRanks} Ranks)`;
        tagAccel.textContent = `MPI P=${selectedMpiRanks} Slabs`;
        updateMpiTopology();
        updateHeaderTelemetry();
        logTerminal(`[CONFIG] MPI Worker partition updated: P = ${selectedMpiRanks} distributed ranks`, 'log-mpi');
      });
    });

    // Launch button
    btnExecute.addEventListener('click', executeBenchmark);

    // Download output button
    btnDownloadOutput.addEventListener('click', () => {
      if (!lastProcessedDataUrl) return;
      const a = document.createElement('a');
      a.href = lastProcessedDataUrl;
      const prefix = selectedEngine === 'cuda' ? 'cuda_output' : 'mpi_output';
      a.download = `${prefix}_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });

    // Terminal tools
    btnCopyLogs.addEventListener('click', () => {
      const text = Array.from(terminalOutput.querySelectorAll('.terminal-line'))
        .map(el => el.textContent)
        .join('\n');
      navigator.clipboard.writeText(text);
      btnCopyLogs.textContent = 'Copied!';
      setTimeout(() => { btnCopyLogs.textContent = 'Copy'; }, 1500);
    });

    btnClearLogs.addEventListener('click', () => {
      terminalOutput.innerHTML = '';
      logTerminal('[SYSTEM] Terminal console buffer cleared.', 'log-system');
    });
  }
});
