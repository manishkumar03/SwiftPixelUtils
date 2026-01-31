# SwiftPixelUtils - Implementation Summary

## Overview

SwiftPixelUtils is a high-performance Swift library for image preprocessing optimized for ML/AI inference pipelines on iOS/macOS. It provides raw pixel data extraction, color format conversion, resizing, normalization, bounding box utilities, label databases, image analysis, augmentation, tensor operations, quantization, and more.

## Architecture

The library is structured with the following key components:

### Core Files

1. **SwiftPixelUtils.swift** - Main entry point and version info
2. **Types.swift** - Complete type definitions including:
   - ImageSource (url, file, data, base64, cgImage, uiImage, nsImage)
   - ColorFormat (rgb, rgba, bgr, bgra, grayscale, hsv, hsl, lab, yuv, ycbcr)
   - ResizeStrategy (cover, contain, stretch, letterbox)
   - DataLayout (hwc, chw, nhwc, nchw)
   - All configuration structs (PixelDataOptions, ResizeOptions, etc.)

3. **Errors.swift** - Typed error handling with LocalizedError

4. **ModelPresets.swift** - Pre-configured settings for:
   - YOLO (yolo, yolov8)
   - MobileNet (mobilenet, mobilenet_v2, mobilenet_v3)
   - EfficientNet
   - ResNet (resnet, resnet50)
   - Vision Transformer (vit)
   - CLIP
   - SAM (Segment Anything Model)
   - DINO
   - DETR

5. **PixelExtractor.swift** - Core pixel data extraction with:
   - getPixelData() - Single image processing
   - batchGetPixelData() - Batch processing with concurrency
   - Image loading from multiple sources
   - ROI (Region of Interest) cropping
   - Multiple resize strategies
   - Color format conversion (10 formats)
   - Normalization (scale, ImageNet, TensorFlow, raw, custom)
   - Data layout transformation (HWC ↔ CHW ↔ NHWC ↔ NCHW)

6. **BoundingBox.swift** - Bounding box utilities:
   - Format conversion (xyxy ↔ xywh ↔ cxcywh)
   - Scaling between image dimensions
   - Clipping to image boundaries
   - IoU (Intersection over Union) calculation
   - Non-Maximum Suppression (NMS)

7. **LabelDatabase.swift** - Built-in label databases loaded from JSON:
   - COCO (80 classes)
   - COCO-91 (91 classes with background)
   - ImageNet (1000 classes)
   - ImageNet-21K (21,843 classes)
   - CIFAR-10 (10 classes)
   - CIFAR-100 (100 classes)
   - VOC (21 classes)
   - Places365 (365 classes)
   - ADE20K (150 classes)
   - Custom label loading support

### Feature Modules

8. **ImageAnalyzer.swift** - Image analysis utilities:
   - Image statistics (mean, std per channel)
   - Histogram calculation (per-channel, configurable bins)
   - Blur detection (Laplacian variance method)
   - Brightness/contrast analysis
   - Dominant color extraction

9. **ImageAugmentor.swift** - Image augmentation for training:
   - Rotation (arbitrary angles)
   - Flipping (horizontal, vertical, both)
   - Color jitter (brightness, contrast, saturation, hue)
   - Grayscale conversion
   - Gaussian blur
   - Cutout/random erasing
   - Combined augmentation pipeline

10. **Quantizer.swift** - Tensor quantization:
    - Float to int8/uint8/int16 conversion
    - Per-tensor and per-channel quantization
    - Symmetric and asymmetric modes
    - Dequantization (inverse transform)
    - Automatic scale/zero-point calculation

11. **TensorOperations.swift** - Tensor manipulation:
    - Channel extraction
    - Patch extraction (extractPatch)
    - Permutation/transpose
    - Batch assembly (concatenateToBatch)
    - Reshape, squeeze, unsqueeze
    - Softmax activation

12. **TensorToImage.swift** - Convert tensors back to images:
    - Support for all data layouts (HWC, CHW, NHWC, NCHW)
    - Denormalization (ImageNet, TensorFlow, custom)
    - CGImage and platform image output
    - Batch tensor unpacking

13. **TensorValidation.swift** - Tensor validation utilities:
    - Shape validation
    - Range checking (NaN, Inf, bounds)
    - Statistics computation
    - Model input compatibility checking

14. **Letterbox.swift** - Letterbox padding operations:
    - Add letterbox padding with configurable color
    - Automatic aspect ratio preservation
    - Reverse letterbox transform
    - Coordinate transformation for detections

15. **DrawingVisualization.swift** - Visualization utilities:
    - Draw bounding boxes with labels
    - Draw keypoints and skeletons
    - Overlay segmentation masks
    - Heatmap visualization
    - Text annotations

16. **VideoFrameExtractor.swift** - Video processing:
    - Extract frames at timestamps
    - Extract multiple frames (uniform/keyframes)
    - Thumbnail generation
    - Video metadata retrieval

17. **MultiCropOperations.swift** - Advanced cropping:
    - Grid extraction (patches)
    - Multi-scale crops
    - Random crops with seed support
    - Five-crop and ten-crop augmentation

18. **CameraFrameUtilities.swift** - Camera integration:
    - CMSampleBuffer to CGImage conversion
    - CVPixelBuffer processing
    - Real-time frame preprocessing
    - Orientation handling

### Resource Files (JSON)

- `coco_labels.json` - 80 COCO classes
- `coco91_labels.json` - 91 COCO classes with background
- `imagenet_labels.json` - 1000 ImageNet classes
- `imagenet21k_labels.json` - 21,843 ImageNet-21K classes
- `cifar10_labels.json` - 10 CIFAR-10 classes
- `cifar100_labels.json` - 100 CIFAR-100 classes
- `voc_labels.json` - 21 VOC classes
- `places365_labels.json` - 365 scene classes
- `ade20k_labels.json` - 150 segmentation classes

## Features Implemented

### ✅ Core Functionality
- [x] Raw pixel data extraction
- [x] Multiple image sources (URL, file, data, base64, CGImage, UIImage/NSImage)
- [x] 10 color formats
- [x] 4 resize strategies
- [x] ROI (Region of Interest) cropping
- [x] 4 normalization presets + custom
- [x] 4 data layouts
- [x] Async/await API

### ✅ Bounding Box Utilities
- [x] Format conversion (xyxy, xywh, cxcywh)
- [x] Scaling
- [x] Clipping
- [x] IoU calculation
- [x] Non-Maximum Suppression

### ✅ Model Support
- [x] 12 pre-configured model presets
- [x] YOLO preprocessing
- [x] MobileNet preprocessing
- [x] ResNet preprocessing
- [x] ViT preprocessing
- [x] CLIP preprocessing

### ✅ Label Database
- [x] COCO labels (80 classes)
- [x] COCO-91 labels (91 classes)
- [x] ImageNet labels (1000 classes)
- [x] ImageNet-21K labels (21,843 classes)
- [x] CIFAR-10/100 labels
- [x] VOC labels (21 classes)
- [x] Places365 labels (365 classes)
- [x] ADE20K labels (150 classes)
- [x] Top-K label prediction
- [x] Custom label loading

### ✅ Image Analysis
- [x] Image statistics (mean, std, histogram)
- [x] Blur detection (Laplacian variance)
- [x] Brightness/contrast analysis
- [x] Dominant color extraction

### ✅ Image Augmentation
- [x] Rotation, flips
- [x] Color jitter (brightness, contrast, saturation, hue)
- [x] Cutout/random erasing
- [x] Gaussian blur
- [x] Combined augmentation pipeline

### ✅ Tensor Operations
- [x] Channel extraction
- [x] Patch extraction
- [x] Permutation
- [x] Batch assembly
- [x] Reshape, squeeze, unsqueeze
- [x] Softmax activation
- [x] Tensor to image conversion

### ✅ Quantization
- [x] Float to int8/uint8/int16
- [x] Per-tensor and per-channel
- [x] Symmetric and asymmetric modes
- [x] Dequantization
- [x] Parameter calculation

### ✅ Letterbox & Drawing
- [x] Letterbox padding
- [x] Reverse letterbox transform
- [x] Draw bounding boxes with labels
- [x] Draw keypoints and skeletons
- [x] Overlay segmentation masks
- [x] Heatmap visualization

### ✅ Video & Camera
- [x] Video frame extraction (timestamps, uniform, keyframes)
- [x] Video metadata retrieval
- [x] CMSampleBuffer processing
- [x] CVPixelBuffer conversion
- [x] Real-time frame preprocessing

### ✅ Multi-Crop Operations
- [x] Grid/patch extraction
- [x] Multi-scale crops
- [x] Random crop with seed
- [x] Five-crop and ten-crop

### ✅ Testing
- [x] 24 comprehensive unit tests
- [x] Performance benchmarks
- [x] Error handling tests

## Apple Frameworks Used

- **CoreGraphics** - Image manipulation, context creation, drawing
- **CoreImage** - Advanced image processing, filters
- **Accelerate** - High-performance vectorized operations (vDSP, vImage)
- **Foundation** - Base utilities, async/await, JSON parsing
- **AVFoundation** - Video frame extraction, camera integration
- **UIKit** (iOS) / **AppKit** (macOS) - Platform image types

## Performance Optimizations

1. **Vectorized Operations** - Accelerate framework (vDSP) for tensor operations
2. **Async/Await** - Modern concurrency for non-blocking operations
3. **Batch Processing** - Concurrent image processing with configurable parallelism
4. **Memory Efficiency** - Pre-allocated arrays, in-place operations where possible
5. **Native Code** - No bridging overhead, pure Swift/CoreGraphics
6. **Lazy Loading** - Label databases loaded on-demand from JSON resources

## Usage Examples

### Basic Usage
```swift
let result = try await PixelExtractor.getPixelData(
    source: .url(URL(string: "https://example.com/image.jpg")!),
    options: PixelDataOptions()
)
```

### Model Presets
```swift
let yoloResult = try await PixelExtractor.getPixelData(
    source: .file(fileURL),
    options: ModelPresets.yolov8
)
```

### Bounding Boxes
```swift
let xyxyBox = BoundingBox.convertFormat(
    [[320, 240, 100, 80]],
    from: .cxcywh,
    to: .xyxy
)

let filtered = BoundingBox.nonMaxSuppression(
    detections: detections,
    iouThreshold: 0.5
)
```

### Label Database
```swift
// Get label for class index
let label = LabelDatabase.getLabel(281, dataset: .imagenet)  // "tabby cat"

// Get top-5 predictions
let top5 = LabelDatabase.getTopLabels(
    scores: modelOutput,
    dataset: .imagenet,
    k: 5
)
```

### Image Augmentation
```swift
let augmented = try ImageAugmentor.augment(
    image: cgImage,
    operations: [
        .rotate(degrees: 15),
        .colorJitter(brightness: 0.2, contrast: 0.2),
        .horizontalFlip
    ]
)
```

### Tensor Operations
```swift
// Extract patch from tensor
let patch = try TensorOperations.extractPatch(
    from: tensorData,
    options: PatchOptions(x: 100, y: 100, width: 64, height: 64),
    inputWidth: 640,
    inputHeight: 480,
    channels: 3
)

// Softmax activation
let probabilities = TensorOperations.softmax(logits)
```

### Video Processing
```swift
// Extract frames from video
let frames = try await VideoFrameExtractor.extractFrames(
    from: videoURL,
    count: 10,
    mode: .uniform
)
```

## Installation

Add to Package.swift:
```swift
dependencies: [
    .package(url: "https://github.com/user/SwiftPixelUtils.git", from: "1.0.0")
]
```

Or via Xcode: File → Add Package Dependencies

## Platform Requirements

- iOS 15.0+
- macOS 12.0+
- tvOS 15.0+
- watchOS 8.0+
- Swift 5.0+

## Testing

Run tests with:
```bash
swift test
```

All 24 tests pass covering:
- Core pixel extraction
- Bounding box operations
- Label database queries
- Tensor operations
- Quantization round-trips
- Performance benchmarks

## File Structure

```
SwiftPixelUtils/
├── Package.swift
├── Sources/
│   └── SwiftPixelUtils/
│       ├── SwiftPixelUtils.swift
│       ├── Types.swift
│       ├── Errors.swift
│       ├── ModelPresets.swift
│       ├── PixelExtractor.swift
│       ├── BoundingBox.swift
│       ├── LabelDatabase.swift
│       ├── ImageAnalyzer.swift
│       ├── ImageAugmentor.swift
│       ├── Quantizer.swift
│       ├── TensorOperations.swift
│       ├── TensorToImage.swift
│       ├── TensorValidation.swift
│       ├── Letterbox.swift
│       ├── DrawingVisualization.swift
│       ├── VideoFrameExtractor.swift
│       ├── MultiCropOperations.swift
│       ├── CameraFrameUtilities.swift
│       ├── Examples.swift
│       └── Resources/
│           ├── coco_labels.json
│           ├── coco91_labels.json
│           ├── imagenet_labels.json
│           ├── imagenet21k_labels.json
│           ├── cifar10_labels.json
│           ├── cifar100_labels.json
│           ├── voc_labels.json
│           ├── places365_labels.json
│           └── ade20k_labels.json
└── Tests/
    └── SwiftPixelUtilsTests/
        └── SwiftPixelUtilsTests.swift
```

## License

MIT License

## Author

Implemented as a native Swift alternative to react-native-vision-utils with full feature parity.
