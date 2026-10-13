#include <cuda_runtime.h>
struct Det { float x1, y1, x2, y2, score; int cls; };

__global__ void decodeKernel(const float* out, int A, int C, float thr,
                             Det* dets, int* count, int maxDets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= A) return;

  // output layout: [4+C, A]  (row-major, each row has A entries)
  float best = 0.f; int cls = -1;
  for (int c = 0; c < C; ++c) {
    float s = out[(4 + c) * A + i];
    if (s > best) { best = s; cls = c; }
  }
  if (best < thr) return;

  int idx = atomicAdd(count, 1);
  if (idx >= maxDets) return;

  float cx = out[0 * A + i], cy = out[1 * A + i];
  float w  = out[2 * A + i], h  = out[3 * A + i];
  dets[idx] = { cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, cls };
}

int gpuDecode(const float* dOut, int A, int C, float thr,
              Det* dDets, int* dCount, int maxDets, cudaStream_t s) {
  int threads = 256, blocks = (A + threads - 1) / threads;
  decodeKernel<<<blocks, threads, 0, s>>>(dOut, A, C, thr, dDets, dCount, maxDets);
  return 0;
}
