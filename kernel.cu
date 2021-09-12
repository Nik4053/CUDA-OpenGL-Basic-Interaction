#include "kernel.h"
#include "stdio.h"
#include "math.h"
#include "device_launch_parameters.h"

#define TX 32
#define TY 32


__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__global__
void distanceKernel(GpuData* gpudata, int2 pos) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gpudata->w) || (y >= gpudata->h)) return; // Check if within image bounds
    const int i = x + y * gpudata->w; // 1D indexing
    const int dist = sqrtf((x - pos.x) * (x - pos.x) +
                           (y - pos.y) * (y - pos.y));
    gpudata->distanceData[i] = dist;
}

__global__
void imageKernel(uchar4 *d_out, GpuData* gpudata) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gpudata->w) || (y >= gpudata->h)) return; // Check if within image bounds
    const int i = x + y * gpudata->w; // 1D indexing
    const int dist = gpudata->distanceData[i];
    const unsigned char intensity = clip(255 - dist);
    d_out[i].x = intensity;
    d_out[i].y = intensity;
    d_out[i].z = 0;
    d_out[i].w = 255;
}

void kernelLauncher(uchar4 *d_out, GpuData *gpudata, int2 pos, int w, int h) {
    // 3d version
    const dim3 blockSize(TX, TY, 1);
    const dim3 gridSize = dim3((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y,
                               1); // + TX - 1 for w size that is not divisible by TX

    distanceKernel<<<gridSize, blockSize>>>(gpudata, pos);
    gpuErrchk(cudaPeekAtLastError());
    imageKernel<<<gridSize, blockSize>>>(d_out, gpudata);
    gpuErrchk(cudaPeekAtLastError());
}

void init(GpuData *gpudata) {
    cudaMalloc((void **) &gpudata->distanceData, gpudata->w * gpudata->h * sizeof(*gpudata->distanceData));
}

void destroy(GpuData *gpudata) {
    cudaFree(gpudata->distanceData);
}