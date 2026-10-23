/**
 * Module 5: GPU Programming with CUDA - Heterogeneous Computing
 * Kernel: rgbToGrayscaleKernel
 * 
 * Concept Demonstrated:
 * - 2D Grid and 2D Block dim3 thread hierarchy
 * - Global thread coordinate mapping:
 *     col = blockIdx.x * blockDim.x + threadIdx.x
 *     row = blockIdx.y * blockDim.y + threadIdx.y
 * - Boundary condition validation to prevent out-of-bounds memory accesses
 * - Linear memory addressing:
 *     grayOffset = row * width + col
 *     rgbOffset  = (row * width + col) * 3
 * - Luminance conversion via NTSC / ITU-R BT.601 standard weights:
 *     Gray = 0.299 * R + 0.587 * G + 0.114 * B
 */

extern "C" __global__
void rgbToGrayscaleKernel(const unsigned char* __restrict__ d_rgb,
                         unsigned char* __restrict__ d_gray,
                         int width,
                         int height)
{
    // Compute 2D coordinates of the thread in the entire image grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if within image boundaries (handles non-multiples of block dimensions)
    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset  = grayOffset * 3;

        // Fetch RGB color channels from global device memory
        unsigned char r = d_rgb[rgbOffset];
        unsigned char g = d_rgb[rgbOffset + 1];
        unsigned char b = d_rgb[rgbOffset + 2];

        // Compute weighted luminance value
        float gray = 0.299f * (float)r + 0.587f * (float)g + 0.114f * (float)b;

        // Clamp to valid 8-bit unsigned integer range [0, 255] and store
        d_gray[grayOffset] = (unsigned char)(gray > 255.0f ? 255.0f : (gray < 0.0f ? 0.0f : gray));
    }
}
