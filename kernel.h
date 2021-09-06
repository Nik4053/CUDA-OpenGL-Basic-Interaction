#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

struct uchar4;
struct int2;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct GpuData {
    int w;
    int h;
    int2 pos;
    int *CUDAdistanceData;
};

void kernelLauncher(uchar4 *d_out, GpuData *gpudata);

void init(GpuData *gpudata);

void destroy(GpuData *gpudata);

#endif