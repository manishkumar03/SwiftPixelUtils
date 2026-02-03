# SwiftPixelUtils Documentation

Comprehensive guides and references for machine learning on iOS and macOS.

## 📚 Documentation Index

### Core Guides

| Guide | Description | Topics |
|-------|-------------|--------|
| [01 - Image Preprocessing](01-image-preprocessing-fundamentals.md) | Complete guide to preparing images for ML models | Pixel extraction, resizing, normalization, data layouts, batch processing |
| [02 - Quantization](02-quantization-guide.md) | Understanding quantized models | INT8, UINT8, FP16, INT4, scale/zero-point, per-channel |
| [03 - Classification](03-classification-output.md) | Image classification reference | Softmax, top-K, architectures, ImageNet labels |
| [04 - Detection](04-detection-output.md) | Object detection with YOLO | Bounding boxes, NMS, IoU, COCO labels |
| [05 - Segmentation](05-segmentation-output.md) | Semantic segmentation guide | DeepLab, masks, color maps, VOC labels |
| [06 - Augmentation](06-image-augmentation.md) | Data augmentation techniques | Geometric, photometric, MixUp, AutoAugment |
| [07 - Visualization](07-visualization-guide.md) | Drawing ML results | Boxes, masks, heatmaps, debugging |
| [08 - Label Database](08-label-database.md) | Complete class label reference | ImageNet, COCO, VOC, Open Images, LVIS, Kinetics |
| [09 - Depth Estimation](09-depth-estimation.md) | Monocular depth estimation | MiDaS, DPT, ZoeDepth, Depth Anything, colormaps, Float16 utilities |
| [10 - ONNX Runtime](10-onnx-runtime-integration.md) | ONNX Runtime integration | Tensor creation, output parsing, YOLOv8, RT-DETR, batch inference |
| [11 - Image Processing Theory](11-image-processing-theory.md) | Core image processing theory | Sampling, filtering, color, noise, frequency domain |

## 🚀 Quick Start

### Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/manishkumar03/SwiftPixelUtils.git", from: "1.0.0")
]
```

### Basic Usage

> **Note:** SwiftPixelUtils functions are **synchronous** (`throws` not `async throws`). Remote URLs are not supported - download images first.

```swift
import SwiftPixelUtils

// Classification preprocessing
let pixelData = try PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: ModelPresets.mobilenet
)

// Detection postprocessing
let detections = try DetectionOutput.process(
    outputData: detectionOutputData,
    format: .yolov8(numClasses: 80),
    confidenceThreshold: 0.5,
    iouThreshold: 0.45,
    labels: .coco,
    imageSize: image.size
)

// Segmentation postprocessing
let mask = try SegmentationOutput.process(
    outputData: segmentationOutputData,
    format: .logits(height: 512, width: 512, numClasses: 21),
    labels: .voc
)
```

## 📖 Guide Descriptions

### [01 - Image Preprocessing Fundamentals](01-image-preprocessing-fundamentals.md)
Everything you need to know about preparing images for ML inference. Covers pixel extraction from various iOS/macOS image sources, color format conversions (RGB, BGR, HSV, YUV, LAB), resizing strategies with automatic letterbox transform metadata, normalization schemes (0-1, -1 to 1, ImageNet), data layouts (HWC, CHW, NHWC, NCHW), orientation handling, Float16 output, and batch processing.

### [02 - Quantization Guide](02-quantization-guide.md)
Deep dive into model quantization for efficient mobile inference. Explains quantization theory and math, different schemes (PTQ, QAT, dynamic), data types (INT8, UINT8, FP16/Float16, BF16, INT4), framework-specific quantization (TFLite, CoreML, PyTorch), preprocessing for quantized models, Float16 output for Apple Silicon, and accuracy considerations.

### [03 - Classification Output](03-classification-output.md)
Complete reference for image classification. Covers classification theory, softmax deep dive with numerical stability, top-K prediction extraction, popular architectures (MobileNet, EfficientNet, ResNet, Vision Transformer), transfer learning strategies, confidence calibration, and ImageNet label handling.

### [04 - Detection Output](04-detection-output.md)
Comprehensive YOLO and object detection guide. Explains YOLO architecture evolution (v1-v11, plus YOLOv9/v10), RT-DETR real-time transformers, detection pipelines, output tensor formats, bounding box coordinate systems, NMS algorithms (standard, soft, class-specific), anchor boxes, multi-scale detection, automatic letterbox coordinate correction, and COCO 80-class labels.

### [05 - Segmentation Output](05-segmentation-output.md)
Complete semantic segmentation reference. Covers segmentation types (semantic, instance, panoptic), encoder-decoder architectures, DeepLab family (V1-V3+), UNet, Mask2Former, SAM/SAM2, SegFormer, FCN, PSPNet, dilated convolutions, ASPP, output processing techniques, upsampling methods, color map generation, and Pascal VOC labels.

### [06 - Image Augmentation](06-image-augmentation.md)
Data augmentation techniques for training. Includes geometric transforms (rotation, crop, flip, shear, perspective, elastic), photometric transforms (brightness, contrast, saturation, hue, gamma), noise and blur techniques, occlusion methods (cutout, GridMask), advanced methods (MixUp, CutMix, Mosaic), and AutoAugment/RandAugment.

### [07 - Visualization Guide](07-visualization-guide.md)
Drawing and displaying ML results. Covers color theory for visualization, drawing bounding boxes with labels, segmentation mask overlays, classification result displays, Grad-CAM and attention heatmaps, debugging visualizations, video/animation techniques, and platform-specific rendering (UIKit, AppKit, SwiftUI, Metal).

### [08 - Label Database](08-label-database.md)
Complete class label reference for major datasets. Includes full ImageNet-1K labels (1000 classes), COCO 80 classes with category IDs, Pascal VOC 21 classes with color palette, Cityscapes 19/35 classes, ADE20K 150 classes, and cross-dataset mapping utilities.

### [10 - ONNX Runtime Integration](10-onnx-runtime-integration.md)
Comprehensive ONNX Runtime integration guide. Covers MLFramework ONNX variants (Float32, Float16, UInt8, Int8), tensor creation with `ONNXHelper`, pre-configured model configs (YOLOv8, RT-DETR, ResNet, ViT), output parsing for detection/classification/segmentation, batch inference, and complete working examples. **Note:** ONNX Runtime requires separate setup - see the guide for CocoaPods/SPM installation instructions.

## 🎯 Use Cases

### Image Classification
```
Guides: 01 → 02 (if quantized) → 03 → 07
```

### Object Detection (YOLO)
```
Guides: 01 → 02 (if quantized) → 04 → 07 → 08
```

### Semantic Segmentation
```
Guides: 01 → 02 (if quantized) → 05 → 07 → 08
```

### ONNX Runtime Inference
```
Guides: 01 → 10 → 04/03/05 (based on task) → 07
```

### Model Training
```
Guides: 01 → 06 → 03/04/05 (based on task)
```

### Debugging
```
Guides: 01 (verification) → 07 (debugging section)
```

## 🔧 SwiftPixelUtils Features

- **Pixel Extraction**: Extract pixels from UIImage, CGImage, CVPixelBuffer, CIImage
- **Preprocessing**: Resize, normalize, and format images for any ML model
- **Output Formats**: Float32, Float16, Int32, UInt8 arrays for different ML frameworks
- **Letterbox Transform**: Automatic transform metadata for reverse coordinate mapping
- **Orientation Handling**: Opt-in UIImage/EXIF orientation normalization
- **Quantization Support**: Handle INT8, UINT8, INT4, FP16 quantized models
- **Detection Decoding**: Parse YOLO outputs with NMS
- **Segmentation Decoding**: Process mask outputs with color mapping
- **Classification**: Top-K predictions with labels
- **Visualization**: Draw boxes, masks, and heatmaps
- **Labels**: Built-in ImageNet, COCO, VOC, Cityscapes labels
- **Synchronous API**: All functions use `throws` (not `async throws`)
- **Local Files Only**: Remote URLs not supported - download first with URLSession

## 📱 Platform Support

- iOS 15.0+
- macOS 12.0+
- tvOS 15.0+
- watchOS 8.0+
- Swift 5.9+

## 📄 License

MIT License - see [LICENSE](../../../LICENSE) for details.
