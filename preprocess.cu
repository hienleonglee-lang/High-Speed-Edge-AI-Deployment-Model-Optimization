#include <cuda_runtime.h>
#include <stdint.h>

// src: uint8 BGR, HWC (sh x sw x 3).  dst: float RGB, CHW (3 x H x W), values 0..1
__global__ void preprocessKernel(const uint8_t* src, int sw, int sh,
                                 float* dst, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  // bilinear sampling, same pixel-center convention as cv::resize
  float fx = (x + 0.5f) * ((float)sw / W) - 0.5f;
  float fy = (y + 0.5f) * ((float)sh / H) - 0.5f;
  fx = fminf(fmaxf(fx, 0.f), sw - 1.f);
  fy = fminf(fmaxf(fy, 0.f), sh - 1.f);
  int x0 = (int)fx, y0 = (int)fy;
  int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
  float ax = fx - x0, ay = fy - y0;

  float rgb[3];
  for (int c = 0; c < 3; ++c) {            // c = channel index in the source (B,G,R)
    float p00 = src[(y0 * sw + x0) * 3 + c];
    float p01 = src[(y0 * sw + x1) * 3 + c];
    float p10 = src[(y1 * sw + x0) * 3 + c];
    float p11 = src[(y1 * sw + x1) * 3 + c];
    float top = p00 + (p01 - p00) * ax;
    float bot = p10 + (p11 - p10) * ax;
    rgb[2 - c] = (top + (bot - top) * ay) / 255.f;   // 2 - c flips BGR -> RGB
  }
  int plane = W * H;
  dst[0 * plane + y * W + x] = rgb[0];
  dst[1 * plane + y * W + x] = rgb[1];
  dst[2 * plane + y * W + x] = rgb[2];
}

void gpuPreprocess(const uint8_t* dSrc, int sw, int sh,
                   float* dDst, int W, int H, cudaStream_t s) {
  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16);
  preprocessKernel<<<grid, block, 0, s>>>(dSrc, sw, sh, dDst, W, H);
}
