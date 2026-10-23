/**
 * Module 5: GPU Programming with CUDA - Heterogeneous Computing
 * Kernel: imageConvolution2DKernel
 * 
 * Concept Demonstrated:
 * - 2D Stencil computation mapped across CUDA thread blocks
 * - Thread coordinate mapping in 2D:
 *     col = blockIdx.x * blockDim.x + threadIdx.x
 *     row = blockIdx.y * blockDim.y + threadIdx.y
 * - Arbitrary odd filter sizes (e.g. 3x3, 5x5)
 * - Edge boundary clamping (mirror / clamp to edge) to avoid illegal memory access
 * - Parallel execution across all color channels (RGB)
 * - Output normalization and clamping to [0, 255]
 */

extern "C" __global__
void imageConvolution2DKernel(const unsigned char* __restrict__ d_input,
                              unsigned char* __restrict__ d_output,
                              int width,
                              int height,
                              int channels,
                              const float* __restrict__ d_filter,
                              int filterSize)
{
    // Compute global 2D column and row index for current thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int halfFilter = filterSize / 2;

    // Check if the current thread falls within image dimensions
    if (col < width && row < height) {
        // Iterate through each channel (e.g., Red, Green, Blue)
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;

            // Apply 2D convolution stencil centered at (row, col)
            for (int fRow = -halfFilter; fRow <= halfFilter; ++fRow) {
                for (int fCol = -halfFilter; fCol <= halfFilter; ++fCol) {
                    // Clamp neighbor coordinates to valid image boundaries
                    int inRow = row + fRow;
                    int inCol = col + fCol;

                    if (inRow < 0) inRow = 0;
                    else if (inRow >= height) inRow = height - 1;

                    if (inCol < 0) inCol = 0;
                    else if (inCol >= width) inCol = width - 1;

                    int pixelIndex = (inRow * width + inCol) * channels + c;
                    int filterIndex = (fRow + halfFilter) * filterSize + (fCol + halfFilter);

                    sum += (float)d_input[pixelIndex] * d_filter[filterIndex];
                }
            }

            // Linear index for the output pixel
            int outIndex = (row * width + col) * channels + c;

            // Clamp result to valid [0, 255] range
            if (sum < 0.0f) sum = 0.0f;
            if (sum > 255.0f) sum = 255.0f;

            d_output[outIndex] = (unsigned char)sum;
        }
    }
}
