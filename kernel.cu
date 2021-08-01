#include "kernel.h"
#include "stdio.h"
#define TX 32
#define TY 32



__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

template<typename T>
__global__
void distanceKernel(uchar4 *d_out, int w, int h, int2 pos) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= w) || (y >= h)) return; // Check if within image bounds
    const int i = x + y * w; // 1D indexing
    const int dist = sqrtf((x - pos.x) * (x - pos.x) +
                           (y - pos.y) * (y - pos.y));
    const unsigned char intensity = clip(255 - dist);
    d_out[i].x = intensity;
    d_out[i].y = intensity;
    d_out[i].z = 0;
    d_out[i].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {
    // 3d version
    const dim3 blockSize(TX, TY, 1);
    const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY, 1); // + TX - 1 for w size that is not divisible by TX

    distanceKernel<float><<<gridSize, blockSize>>>(d_out, w, h, pos);
    gpuErrchk( cudaPeekAtLastError() );
}