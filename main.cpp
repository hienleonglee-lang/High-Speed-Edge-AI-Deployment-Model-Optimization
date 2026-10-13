#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

struct Det { float x1, y1, x2, y2, score; int cls; };
int gpuDecode(const float* dOut, int numAnchors, int numClasses, float thr,
              Det* dDets, int* dCount, int maxDets, cudaStream_t s);
void gpuPreprocess(const uint8_t* dSrc, int sw, int sh,
                   float* dDst, int W, int H, cudaStream_t s);

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << m << "\n";
  }
} gLogger;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::cerr << cudaGetErrorString(e) << " @" << __LINE__ << "\n"; std::exit(1);} } while (0)

int main(int argc, char** argv) {
  const char* enginePath = argc > 1 ? argv[1] : "models/yolov8m_fp16.engine";
  std::ifstream f(enginePath, std::ios::binary);
  if (!f) { std::cerr << "Cannot open engine file\n"; return 1; }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  auto runtime = nvinfer1::createInferRuntime(gLogger);
  auto engine  = runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!engine) { std::cerr << "Engine deserialization failed\n"; return 1; }
  auto ctx = engine->createExecutionContext();

  cv::Mat img = cv::imread("test.jpg");
  if (img.empty()) { std::cerr << "Cannot read test.jpg\n"; return 1; }
  const int sw = img.cols, sh = img.rows;
  const size_t imgBytes = (size_t)sw * sh * 3;
  float sx = (float)sw / 640, sy = (float)sh / 640;

  const int W = 640, H = 640, A = 8400, C = 80, MAXD = 1000;
  uint8_t* dSrc; float *dIn, *dOut; Det* dDets; int* dCount;
  CUDA_CHECK(cudaMalloc(&dSrc, imgBytes));
  CUDA_CHECK(cudaMalloc(&dIn,  3 * H * W * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOut, (4 + C) * A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dDets, MAXD * sizeof(Det)));
  CUDA_CHECK(cudaMalloc(&dCount, sizeof(int)));
  cudaStream_t stream; cudaStreamCreate(&stream);
  ctx->setTensorAddress("images",  dIn);
  ctx->setTensorAddress("output0", dOut);

  // pin the image memory so the async upload is a true DMA copy
  CUDA_CHECK(cudaHostRegister(img.data, imgBytes, cudaHostRegisterDefault));

  std::vector<Det> dets(MAXD);
  std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> keep;

  // Full pass: upload raw image -> GPU preprocess -> infer -> GPU decode -> NMS
  auto runOnce = [&]() {
    CUDA_CHECK(cudaMemcpyAsync(dSrc, img.data, imgBytes, cudaMemcpyHostToDevice, stream));
    gpuPreprocess(dSrc, sw, sh, dIn, W, H, stream);
    ctx->enqueueV3(stream);
    cudaMemsetAsync(dCount, 0, sizeof(int), stream);
    gpuDecode(dOut, A, C, 0.25f, dDets, dCount, MAXD, stream);
    int n = 0;
    cudaMemcpyAsync(&n, dCount, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    n = std::min(n, MAXD);
    cudaMemcpy(dets.data(), dDets, n * sizeof(Det), cudaMemcpyDeviceToHost);

    boxes.clear(); scores.clear(); keep.clear();
    for (int i = 0; i < n; ++i) {
      boxes.emplace_back((int)dets[i].x1, (int)dets[i].y1,
                         (int)(dets[i].x2 - dets[i].x1), (int)(dets[i].y2 - dets[i].y1));
      scores.push_back(dets[i].score);
    }
    cv::dnn::NMSBoxes(boxes, scores, 0.25f, 0.45f, keep);
  };

  for (int i = 0; i < 20; ++i) runOnce();            // warmup
  std::vector<double> times;
  for (int i = 0; i < 200; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    runOnce();
    auto t1 = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(times.begin(), times.end());
  double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  std::cout << "Detections after NMS: " << keep.size() << "\n";
  std::cout << "End-to-end  mean: " << mean << " ms, median: " << times[100]
            << " ms, p95: " << times[190] << " ms\n";

  cv::Mat vis = img.clone();
  for (int k : keep) {
    cv::Rect r((int)(boxes[k].x * sx), (int)(boxes[k].y * sy),
               (int)(boxes[k].width * sx), (int)(boxes[k].height * sy));
    cv::rectangle(vis, r, {0, 255, 0}, 2);
  }
  cv::imwrite("result.jpg", vis);
  cudaHostUnregister(img.data);
}
