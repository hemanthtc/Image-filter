/**
 * Module 5: GPU Programming with CUDA - Frontend Interaction Controller
 * 
 * Manages:
 * - Device telemetry interrogation (/api/device-info)
 * - Drag-and-drop & file picker image upload
 * - Interactive convolution filter stencil preview
 * - 2D Grid/Block dim3 calculator
 * - Dual execution dispatch (Host Sequential CPU vs Device CUDA GPU)
 * - Live terminal streaming (synchronized with browser DevTools console)
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const previewLoaded = document.getElementById('previewLoaded');
  const uploadThumbnail = document.getElementById('uploadThumbnail');
  const previewFileName = document.getElementById('previewFileName');
  const previewFileDim = document.getElementById('previewFileDim');
  const btnRemoveImage = document.getElementById('btnRemoveImage');
  const btnLoadSample = document.getElementById('btnLoadSample');

  const optGrayscaleCard = document.getElementById('optGrayscaleCard');
  const optConvCard = document.getElementById('optConvCard');
  const convolutionSettings = document.getElementById('convolutionSettings');
  const filterSelector = document.getElementById('filterSelector');
  const stencilSizeBadge = document.getElementById('stencilSizeBadge');
  const stencilMatrixContainer = document.getElementById('stencilMatrixContainer');

  const blockButtons = document.querySelectorAll('.btn-block-opt');
  const dispGridDim = document.getElementById('dispGridDim');
  const dispThreadMapping = document.getElementById('dispThreadMapping');
  const btnExecute = document.getElementById('btnExecute');
  const btnSpinner = document.getElementById('btnSpinner');

  const valCpuTime = document.getElementById('valCpuTime');
  const valGpuTime = document.getElementById('valGpuTime');
  const valGpuDesc = document.getElementById('valGpuDesc');
  const valSpeedup = document.getElementById('valSpeedup');
  const valSpeedupDesc = document.getElementById('valSpeedupDesc');
  const valTransferTime = document.getElementById('valTransferTime');

  const imgOriginalDisplay = document.getElementById('imgOriginalDisplay');
  const imgProcessedDisplay = document.getElementById('imgProcessedDisplay');
  const badgeInputMeta = document.getElementById('badgeInputMeta');
  const badgeOutputMeta = document.getElementById('badgeOutputMeta');
  const btnDownloadOutput = document.getElementById('btnDownloadOutput');

  const terminalOutput = document.getElementById('terminalOutput');
  const btnCopyLogs = document.getElementById('btnCopyLogs');
  const btnClearLogs = document.getElementById('btnClearLogs');

  const gpuName = document.getElementById('gpuName');
  const gpuVram = document.getElementById('gpuVram');
  const cudaStatusTag = document.getElementById('cudaStatusTag');

  // Application State
  let currentFile = null;
  let currentDimensions = { width: 800, height: 600 };
  let selectedBlockDim = 16;
  let filtersData = {};
  let lastProcessedDataUrl = null;

  // Initialize
  initDeviceTelemetry();
  initFilters();
  initEventListeners();

  // --------------------------------------------------------------------------
  // 1. Device Hardware Telemetry
  // --------------------------------------------------------------------------
  async function initDeviceTelemetry() {
    try {
      logTerminal('[SYSTEM] Querying host and GPU device telemetry...', 'log-system');
      const res = await fetch('/api/device-info');
      if (!res.ok) throw new Error('Failed to query device info');
      const data = await res.json();

      gpuName.textContent = data.gpu_name || 'NVIDIA GPU';
      gpuVram.textContent = `VRAM: ${data.total_vram_mb || 6144} MB | Driver: ${data.driver_version || 'Active'}`;
      
      if (data.cuda_available) {
        cudaStatusTag.textContent = 'CUDA READY';
        cudaStatusTag.className = 'cuda-status-tag';
        logTerminal(`[HARDWARE] Detected ${data.gpu_name} (SM Compute ${data.compute_capability}, ${data.total_vram_mb} MB VRAM)`, 'log-cuda');
        logTerminal(`[CUDA DRIVER] Multiprocessors: ${data.multiprocessors} | Warp Size: ${data.warp_size} | Max Threads/Block: ${data.max_threads_per_block}`, 'log-cuda');
      } else {
        cudaStatusTag.textContent = 'CPU FALLBACK';
        cudaStatusTag.style.color = '#ff9800';
        cudaStatusTag.style.borderColor = 'rgba(255, 152, 0, 0.4)';
        logTerminal('[SYSTEM] Running in safe CPU parallel fallback mode.', 'log-system');
      }
    } catch (err) {
      gpuName.textContent = 'Hardware Status Offline';
      logTerminal(`[ERROR] Failed to query device hardware: ${err.message}`, 'log-error');
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
        // Format floating-point numbers cleanly
        if (Number.isInteger(val)) {
          cell.textContent = val.toString();
        } else {
          cell.textContent = Math.abs(val) >= 0.01 ? val.toFixed(3) : val.toExponential(1);
        }
        stencilMatrixContainer.appendChild(cell);
      });
    });
  }

  // --------------------------------------------------------------------------
  // 3. Grid Calculator Display Update
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

  // --------------------------------------------------------------------------
  // 4. File Upload & Drag-and-Drop Handling
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
  // 5. Execution Pipeline Dispatch
  // --------------------------------------------------------------------------
  async function executeBenchmark() {
    // If no file is selected yet, automatically load the built-in sample
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

    // UI Busy State
    btnExecute.disabled = true;
    btnSpinner.classList.remove('hidden');

    logTerminal('------------------------------------------------------------', 'log-system');
    logTerminal(`[DISPATCH] Starting Heterogeneous Computing Execution: ${operation.toUpperCase()}`, 'log-kernel');

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('operation', operation);
    formData.append('filter_name', filterName);
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

      // Display logs in sequence
      if (Array.isArray(data.logs)) {
        data.logs.forEach(msg => {
          let logClass = 'log-host';
          if (msg.includes('[CUDA MALLOC]') || msg.includes('[CUDA FREE]')) logClass = 'log-cuda';
          else if (msg.includes('MEMCPY')) logClass = 'log-memcpy';
          else if (msg.includes('KERNEL') || msg.includes('THREAD')) logClass = 'log-kernel';
          else if (msg.includes('BENCHMARK') || msg.includes('SPEEDUP')) logClass = 'log-benchmark';
          else if (msg.includes('ERROR') || msg.includes('EXCEPTION')) logClass = 'log-error';
          else if (msg.includes('CPU')) logClass = 'log-system';
          
          logTerminal(msg, logClass);
        });
      }

      // Update Benchmark Metric Cards
      const b = data.benchmarks;
      animateCounter(valCpuTime, b.t_cpu_ms);
      animateCounter(valGpuTime, b.t_gpu_kernel_ms);
      animateCounter(valSpeedup, b.speedup_kernel);

      const transferTotal = (b.t_gpu_h2d_ms + b.t_gpu_d2h_ms).toFixed(2);
      valTransferTime.textContent = transferTotal;

      if (b.speedup_kernel >= 1.0) {
        valSpeedupDesc.textContent = `${b.speedup_kernel}x faster than single-threaded CPU`;
      } else {
        valSpeedupDesc.textContent = `PCIe/Kernel balance for small workload`;
      }

      // Update Output Images
      if (data.output_image_data_url) {
        lastProcessedDataUrl = data.output_image_data_url;
        imgProcessedDisplay.src = data.output_image_data_url;
        badgeOutputMeta.textContent = `${data.image_metadata.width} × ${data.image_metadata.height} • GPU Processed`;
        btnDownloadOutput.disabled = false;
      }

      logTerminal(`[SUCCESS] Benchmark Completed! Kernel Speedup: ${b.speedup_kernel}x`, 'log-benchmark');

    } catch (err) {
      logTerminal(`[ERROR] Processing failed: ${err.message}`, 'log-error');
      console.error(err);
    } finally {
      btnExecute.disabled = false;
      btnSpinner.classList.add('hidden');
    }
  }

  // --------------------------------------------------------------------------
  // 6. Terminal & DevTools Console Synchronization
  // --------------------------------------------------------------------------
  function logTerminal(message, className = 'log-system') {
    // 1. Render in UI terminal
    const line = document.createElement('div');
    line.className = `terminal-line ${className}`;
    line.textContent = message;
    terminalOutput.appendChild(line);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;

    // 2. Mirror in browser DevTools Console with styled formatting
    let consoleStyle = 'color: #94a3b8; font-family: monospace;';
    if (className === 'log-cuda') consoleStyle = 'color: #a78bfa; font-weight: bold; font-family: monospace;';
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
  // 7. Event Listeners Wiring
  // --------------------------------------------------------------------------
  function initEventListeners() {
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
      });
    });

    // Convolution filter selector
    filterSelector.addEventListener('change', (e) => {
      renderStencilMatrix(e.target.value);
    });

    // Block dimension options
    blockButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        blockButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedBlockDim = parseInt(btn.dataset.dim, 10);
        updateGridCalculator();
        logTerminal(`[CONFIG] Block dimension updated: dim3(${selectedBlockDim}, ${selectedBlockDim}) = ${selectedBlockDim * selectedBlockDim} threads/block`, 'log-cuda');
      });
    });

    // Launch button
    btnExecute.addEventListener('click', executeBenchmark);

    // Download output button
    btnDownloadOutput.addEventListener('click', () => {
      if (!lastProcessedDataUrl) return;
      const a = document.createElement('a');
      a.href = lastProcessedDataUrl;
      a.download = `cuda_accelerated_output_${Date.now()}.png`;
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
