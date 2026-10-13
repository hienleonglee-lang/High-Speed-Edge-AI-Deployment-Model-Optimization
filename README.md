[README YOLOv8m TensorRT + CUDA Pipeline.md](https://github.com/user-attachments/files/33404398/README.YOLOv8m.TensorRT.%2B.CUDA.Pipeline.md)
# YOLOv8m Edge AI: TensorRT + CUDA Inference Pipeline

A C++ object-detection inference pipeline for YOLOv8m using TensorRT (FP32, FP16, INT8) with custom CUDA pre- and post-processing kernels. Benchmarked on an NVIDIA T4 (Google Colab).

## Results

All latency numbers are medians over 200 timed runs after 20 warmup runs, on one test image, on a shared Colab T4.

| Configuration | End-to-end latency | Speedup vs PyTorch | mAP50-95 |
| --- | --- | --- | --- |
| PyTorch FP32 (Ultralytics) | 40.7 ms | 1.0x | 0.608 |
| C++ TensorRT FP32 (CPU preprocessing, CUDA decode) | 23.5 ms | 1.7x | not measured |
| C++ TensorRT FP16 (CPU preprocessing, CUDA decode) | 10.0 ms | 4.1x | 0.609 (ratio 1.00) |
| **C++ TensorRT FP16 (CUDA preprocessing and decode)** | **6.77 ms** | **6.0x** | 0.609 (ratio 1.00) |
| C++ TensorRT INT8 Q/DQ (CUDA preprocessing and decode) | about 8.6 ms | about 4.7x | 0.599 (ratio 0.985) |

Model-only latency from `trtexec` (GPU compute, median): FP32 21.1 ms, FP16 6.83 ms, INT8 about 8.5 ms.

FP16 is the best configuration: it is the fastest and shows no measurable accuracy loss. INT8 retained 98.5% of FP32 mAP but was slower than FP16 on this GPU (see Notes).

## How it works

1. Export YOLOv8m from PyTorch to ONNX (static 640x640 input).
2. Convert the ONNX to FP16, or quantize it to INT8 with Q/DQ nodes.
3. Build a TensorRT engine with `trtexec`.
4. The C++ program uploads the raw image and runs a CUDA preprocessing kernel (bilinear resize, BGR to RGB, scale to 0-1, HWC to CHW).
5. TensorRT runs the model.
6. A CUDA decode kernel scans the 8400 candidate boxes in parallel and keeps those above the confidence threshold.
7. The few surviving boxes are copied back, and NMS runs on the CPU.

## Repository layout

```
python/   export, FP16 conversion, INT8 quantization, validation
cpp/      main.cpp, preprocess.cu, postprocess.cu, CMakeLists.txt
models/   (not committed) .pt, .onnx, .engine files
```

Engines and ONNX files are large and tied to a specific GPU and TensorRT version, so they are not included. Rebuild them with the steps below.

## Requirements

- NVIDIA GPU (tested on T4, compute capability 7.5)
- CUDA toolkit with `nvcc`, TensorRT with `trtexec` (tested with TensorRT 11.4 and CUDA 13)
- OpenCV (C++), CMake 3.18 or newer, a C++17 compiler
- Python with `ultralytics`, `onnx`, `onnxsim`, `onnxconverter-common`, `onnxruntime`

## Build and run

### 1. Export to ONNX

```python
from ultralytics import YOLO
YOLO("models/yolov8m.pt").export(format="onnx", imgsz=640, opset=13, simplify=True, dynamic=False)
```

Opset 13 is needed for per-channel INT8 quantization.

### 2. FP16 engine

TensorRT 11 uses strongly typed networks, so precision comes from the ONNX file instead of a `--fp16` flag. Convert the ONNX first:

```python
import onnx
from onnxconverter_common import float16
m = onnx.load("models/yolov8m.onnx")
onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), "models/yolov8m_fp16.onnx")
```

```bash
trtexec --onnx=models/yolov8m_fp16.onnx --saveEngine=models/yolov8m_fp16.engine
```

### 3. INT8 engine (optional)

Quantize with ONNX Runtime using symmetric settings and no bias quantization, which TensorRT requires:

```python
quantize_static("models/yolov8m.onnx", "models/yolov8m_int8_qdq.onnx", Reader(calib_files),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, op_types_to_quantize=["Conv"],
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False})
```

```bash
trtexec --onnx=models/yolov8m_int8_qdq.onnx --saveEngine=models/yolov8m_int8.engine
```

`Reader` is a calibration data reader that yields preprocessed 640x640 RGB images (100 were used).

### 4. Build the C++ program

```bash
cd cpp && mkdir build && cd build
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make -j2
```

`CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES 75` for the T4. Change it for other GPUs.

### 5. Run

From the project root, with `test.jpg` present:

```bash
./cpp/build/edge_ai models/yolov8m_fp16.engine
```

It prints the detection count and the end-to-end mean, median and p95, and saves `result.jpg` with the boxes drawn.

## Accuracy evaluation

mAP was measured with the Ultralytics validator on `coco128`:

| Model | mAP50-95 |
| --- | --- |
| PyTorch FP32 | 0.6079 |
| FP16 ONNX | 0.6093 |
| INT8 Q/DQ ONNX | 0.5989 |

## Notes

- **Why INT8 is slower than FP16:** under TensorRT 11's strongly typed mode only the layers wrapped in Q/DQ nodes run in INT8. Only `Conv` layers were quantized, so the rest of the network runs in FP32, and the Q/DQ boundaries add conversion work. This is a likely explanation and has not been profiled.
- **Why the PyTorch comparison is not like for like:** the PyTorch baseline includes Ultralytics' own preprocessing and NMS, so the 6x figure compares whole pipelines.

## Limitations

- Latency was measured on one image on a shared Colab GPU, so absolute numbers vary between sessions.
- The mAP values come from validating ONNX models with ONNX Runtime. They do not test the TensorRT engines, the custom CUDA kernels, or the C++ NMS. Only detection count and visual output were checked for the C++ pipeline.
- `coco128` is small and drawn from COCO training images that the pretrained model has already seen, so absolute mAP is inflated; only the ratios between models are meaningful.
- INT8 calibration used images from `coco128`, the same set used for validation, so the INT8 accuracy is probably optimistic.
- The TensorRT FP32 row was measured with CPU preprocessing, so it is not directly comparable to the GPU-preprocessing rows.
- Tested only on a T4 on Google Colab, not on edge hardware such as a Jetson.

## Possible future work

- Score the C++ pipeline on the full validation set.
- GPU NMS kernel, batching and CUDA graphs.
- Calibrate INT8 on a separate image set and try quantizing more layer types.
- Run on a Jetson device.
