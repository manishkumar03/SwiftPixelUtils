# SwiftPixelUtils

<p align="center">
  <strong>High-performance Swift library for image preprocessing optimized for ML/AI inference pipelines on iOS/macOS</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#features">Features</a> •
  <a href="Sources/SwiftPixelUtils/docs/README.md">📚 Docs</a>
</p>

---

High-performance Swift library for image preprocessing optimized for ML/AI inference pipelines. Native implementations using Core Image, Accelerate, and Core ML for pixel extraction, tensor conversion, quantization, augmentation, and model-specific preprocessing (YOLO, MobileNet, etc.)

## ✨ Features

- 🚀 **High Performance**: Native implementations using Apple frameworks (Core Image, Accelerate, vImage, Core ML)
- 🤖 **Simplified ML APIs**: One-line preprocessing (`getModelInput`) and postprocessing (`ClassificationOutput`, `DetectionOutput`, `SegmentationOutput`) for all major frameworks
- 🔢 **Raw Pixel Data**: Extract pixel values as typed arrays (Float, Int32, UInt8) ready for ML inference
- 🎨 **Multiple Color Formats**: RGB, RGBA, BGR, BGRA, Grayscale, HSV, HSL, LAB, YUV, YCbCr
- 📐 **Flexible Resizing**: Cover, contain, stretch, and letterbox strategies
- 🔢 **ML-Ready Normalization**: ImageNet, TensorFlow, custom presets
- 📊 **Multiple Data Layouts**: HWC, CHW, NHWC, NCHW (PyTorch/TensorFlow compatible)
- 📦 **Batch Processing**: Process multiple images with concurrency control
- 🖼️ **Multiple Sources**: URL, file, base64, assets, photo library
- 🤖 **Model Presets**: Pre-configured settings for YOLO, MobileNet, EfficientNet, ResNet, ViT, CLIP, SAM, DINO, DETR
- 🎯 **Framework Targets**: Automatic configuration for PyTorch, TensorFlow, TFLite, CoreML, ONNX, ExecuTorch, OpenCV
- 🔄 **Image Augmentation**: Rotation, flip, brightness, contrast, saturation, blur
- 🎨 **Color Jitter**: Granular brightness/contrast/saturation/hue control with range support and seeded randomness
- ✂️ **Cutout/Random Erasing**: Mask random regions with constant/noise fill for robustness training
- 📈 **Image Analysis**: Statistics, metadata, validation, blur detection
- 🧮 **Tensor Operations**: Channel extraction, patch extraction, permutation, batch concatenation
- 🔙 **Tensor to Image**: Convert processed tensors back to images
- 🎯 **Native Quantization**: Float→Int8/UInt8/Int16/INT4 with per-tensor and per-channel support (TFLite/ExecuTorch compatible)
- 🔢 **INT4 Quantization**: 4-bit quantization (8× compression) for LLM weights and edge deployment
- 📊 **Per-Channel Quantization**: Channel-wise scale/zeroPoint for higher accuracy (CNN, Transformer weights)
- 🏷️ **Label Database**: Built-in labels for COCO, ImageNet, VOC, CIFAR, Places365, ADE20K
- 📦 **Bounding Box Utilities**: Format conversion (xyxy/xywh/cxcywh), scaling, clipping, IoU, NMS
- 🖼️ **Letterbox Padding**: YOLO-style letterbox preprocessing with reverse coordinate transform
- 🎨 **Drawing/Visualization**: Draw boxes, keypoints, masks, and heatmaps for debugging
- 🔲 **Grid/Patch Extraction**: Extract image patches in grid patterns for sliding window inference
- 🎲 **Random Crop with Seed**: Reproducible random crops for data augmentation pipelines
- ✅ **Tensor Validation**: Validate tensor shapes, dtypes, and value ranges before inference
- 📦 **Batch Assembly**: Combine multiple images into NCHW/NHWC batch tensors

## 📱 Example App

A comprehensive iOS example app is included in the `Example/` directory, demonstrating all major features:

- **TensorFlow Lite Classification** - MobileNetV2 with TopK results
- **ExecuTorch Classification** - MobileNetV3 with TopK results
- **Object Detection** - YOLOv8 with NMS and bounding box visualization  
- **Semantic Segmentation** - DeepLabV3 with colored mask overlay
- **Pixel Extraction** - Model presets (YOLO, MobileNet, ResNet, ViT, CLIP) and custom options
- **Bounding Box Utilities** - Format conversion, IoU calculation, NMS, scaling, clipping
- **Image Augmentation** - Rotation, flip, brightness, contrast, saturation, blur
- **Tensor Operations** - Channel extraction, permutation, batch assembly
- **Drawing & Visualization** - Boxes, labels, masks, and overlays
- **Comprehensive UI Tests** - 50+ UI tests covering all features

<p align="center">
  <img src="SupportingFiles/example-app-screenshot.png" alt="Example App Screenshot" width="300">
</p>

To run the example app:
```bash
cd Example/SwiftPixelUtilsExampleApp
pod install
open SwiftPixelUtilsExampleApp.xcworkspace
```

To run UI tests, select the `SwiftPixelUtilsExampleAppUITests` target and press `⌘U`.

## �📦 Installation

### Swift Package Manager

Add SwiftPixelUtils to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/SwiftPixelUtils.git", from: "1.0.0")
]
```

Or add it via Xcode:
1. File → Add Package Dependencies
2. Enter the repository URL
3. Select version/branch

## 🚀 Quick Start

### Raw Pixel Data Extraction

```swift
import SwiftPixelUtils

// Load and process image
let result = try await PixelExtractor.getPixelData(
    source: .url(URL(string: "https://example.com/image.jpg")!),
    options: PixelDataOptions()
)

print(result.data) // Float array of pixel values
print(result.width) // Image width
print(result.height) // Image height
print(result.shape) // [height, width, channels]
```

### Using Model Presets

```swift
import SwiftPixelUtils

// Use pre-configured YOLO settings
let result = try await PixelExtractor.getPixelData(
    source: .file(URL(fileURLWithPath: "/path/to/image.jpg")),
    options: ModelPresets.yolov8
)
// Automatically configured: 640x640, letterbox resize, RGB, scale normalization, NCHW layout

// Or MobileNet
let mobileNetResult = try await PixelExtractor.getPixelData(
    source: .file(URL(fileURLWithPath: "/path/to/image.jpg")),
    options: ModelPresets.mobilenet
)
// Configured: 224x224, cover resize, RGB, ImageNet normalization, NHWC layout
```

### Available Model Presets

| Preset | Size | Resize | Normalization | Layout |
|--------|------|--------|---------------|--------|
| `yolo` / `yolov8` | 640×640 | letterbox | scale | NCHW |
| `mobilenet` | 224×224 | cover | ImageNet | NHWC |
| `mobilenet_v2` | 224×224 | cover | ImageNet | NHWC |
| `mobilenet_v3` | 224×224 | cover | ImageNet | NHWC |
| `efficientnet` | 224×224 | cover | ImageNet | NHWC |
| `resnet` | 224×224 | cover | ImageNet | NCHW |
| `resnet50` | 224×224 | cover | ImageNet | NCHW |
| `vit` | 224×224 | cover | ImageNet | NCHW |
| `clip` | 224×224 | cover | CLIP-specific | NCHW |
| `sam` | 1024×1024 | contain | ImageNet | NCHW |
| `dino` | 224×224 | cover | ImageNet | NCHW |
| `detr` | 800×800 | contain | ImageNet | NCHW |

### ExecuTorch Compatibility

All presets with **NCHW layout** work directly with ExecuTorch models exported from PyTorch. For quantized ExecuTorch models, use the output with `Quantizer` to convert to Int8:

```swift
// Preprocess for ExecuTorch quantized model
let pixels = try await PixelExtractor.getPixelData(
    source: .file(imageURL),
    options: PixelDataOptions(
        resize: .fit(width: 224, height: 224),
        normalization: .imagenet,
        dataLayout: .nchw  // PyTorch/ExecuTorch convention
    )
)

// Quantize for Int8 ExecuTorch model
let quantized = try Quantizer.quantize(
    data: pixels.pixelData,
    options: QuantizationOptions(
        mode: .perTensor,
        dtype: .int8,
        scale: [model_scale],
        zeroPoint: [0]  // Symmetric quantization
    )
)
// Pass quantized.int8Data to ExecuTorch tensor
```

## 🤖 Simplified ML APIs

SwiftPixelUtils provides high-level APIs that handle all preprocessing and postprocessing decisions automatically based on your target ML framework.

### `getModelInput()` - One-Line Preprocessing

Instead of manually configuring color format, normalization, layout, and output format, just specify your framework:

```swift
// TensorFlow Lite quantized model - one line!
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteQuantized,
    width: 224,
    height: 224
)
// input.data is raw Data containing UInt8 values in NHWC layout

// PyTorch model
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .pytorch,
    width: 224,
    height: 224
)
// input.data contains Float32 values in NCHW layout with ImageNet normalization
```

#### Available Frameworks

| Framework | Layout | Normalization | Output Type |
|-----------|--------|---------------|-------------|
| `.pytorch` | NCHW | ImageNet | Float32 |
| `.pytorchRaw` | NCHW | [0,1] scale | Float32 |
| `.tensorflow` | NHWC | [-1,1] | Float32 |
| `.tensorflowImageNet` | NHWC | ImageNet | Float32 |
| `.tfliteQuantized` | NHWC | raw [0,255] | UInt8 |
| `.tfliteFloat` | NHWC | [0,1] scale | Float32 |
| `.coreML` | NHWC | [0,1] scale | Float32 |
| `.coreMLImageNet` | NHWC | ImageNet | Float32 |
| `.onnx` | NCHW | ImageNet | Float32 |
| `.execuTorch` | NCHW | ImageNet | Float32 |
| `.execuTorchQuantized` | NCHW | raw [0,255] | Int8 |
| `.openCV` | HWC/BGR | [0,255] | UInt8 |

### `ClassificationOutput.process()` - One-Line Postprocessing

Process classification model output with automatic dequantization, softmax, and label mapping:

```swift
// Process TFLite quantized model output in one line
let result = try ClassificationOutput.process(
    outputData: outputTensor.data,
    quantization: .uint8(scale: quantParams.scale, zeroPoint: quantParams.zeroPoint),
    topK: 5,
    labels: .imagenet(hasBackgroundClass: true)
)

// Access results
for prediction in result.predictions {
    print("\(prediction.label): \(String(format: "%.1f%%", prediction.confidence * 100))")
}

// Or get the top prediction directly
if let top = result.topPrediction {
    print("Top: \(top.label) (\(top.confidence))")
}
```

#### Quantization Types
- `.none` - Float32 model output
- `.uint8(scale:zeroPoint:)` - TFLite quantized
- `.int8(scale:zeroPoint:)` - ExecuTorch/ONNX quantized

#### Label Sources
- `.imagenet(hasBackgroundClass:)` - ImageNet-1K (1000 classes)
- `.coco` - COCO (80 classes)
- `.cifar10` - CIFAR-10 (10 classes)
- `.cifar100` - CIFAR-100 (100 classes)
- `.custom([String])` - Your own label array
- `.none` - Returns "Class N" as labels

### `DetectionOutput.process()` - One-Line Detection Postprocessing

Process object detection model output (YOLO, SSD, etc.) with automatic parsing, NMS, and label mapping:

```swift
// Process YOLOv5 output in one line
let result = try DetectionOutput.process(
    floatOutput: outputArray,
    format: .yolov5(numClasses: 80),
    confidenceThreshold: 0.25,
    iouThreshold: 0.45,
    maxDetections: 20,
    labels: .coco,
    imageSize: originalImage.size,
    modelInputSize: CGSize(width: 320, height: 320),
    outputCoordinateSpace: .normalized  // TFLite YOLO outputs 0-1 coords
)

// Access detections
for detection in result.detections {
    print("\(detection.label): \(String(format: "%.1f%%", detection.confidence * 100))")
    print("  Box: \(detection.boundingBox)")        // Normalized 0-1
    print("  Pixels: \(detection.pixelBoundingBox!)") // Pixel coordinates
}
```

#### Supported Detection Formats
- `.yolov5(numClasses:)` - YOLOv5 format: [1, N, 5+classes]
- `.yolov8(numClasses:)` - YOLOv8 format: [1, 4+classes, N]
- `.yolov8Transposed(numClasses:)` - YOLOv8 already transposed
- `.ssd(numClasses:)` - SSD MobileNet format
- `.efficientDet(numClasses:)` - EfficientDet format

#### Output Coordinate Space

Different models output coordinates in different spaces. Use `outputCoordinateSpace` to handle this:

```swift
// TFLite YOLO models typically output normalized (0-1) coordinates
outputCoordinateSpace: .normalized

// Original PyTorch YOLO exports output pixel coordinates (0-640)
outputCoordinateSpace: .pixelSpace  // default
```

#### Converting to Drawable Boxes

Use `toDrawableBoxes()` to convert detections directly to visualization format:

```swift
// One-line conversion to drawable boxes
let boxes = result.toDrawableBoxes(imageSize: image.size)

// Draw on image
let annotated = try Drawing.drawBoxes(
    on: .uiImage(image),
    boxes: boxes,
    options: BoxDrawingOptions(lineWidth: 3, drawLabels: true, drawScores: true)
)
```

The `toDrawableBoxes()` method:
- Converts normalized coordinates to pixel coordinates
- Applies `DetectionColorPalette` for per-class colors (20 distinct colors)
- Returns `[DrawableBox]` ready for `Drawing.drawBoxes()`

### Complete TFLite Classification Example

Here's a complete example using both simplified APIs for image classification:

```swift
import SwiftPixelUtils
import TensorFlowLite

func classifyImage(_ image: UIImage) async throws -> [ClassificationPrediction] {
    // 1. Preprocess - one line
    let input = try await PixelExtractor.getModelInput(
        source: .uiImage(image),
        framework: .tfliteQuantized,
        width: 224,
        height: 224
    )
    
    // 2. Run inference
    let interpreter = try Interpreter(modelPath: modelPath)
    try interpreter.allocateTensors()
    try interpreter.copy(input.data, toInputAt: 0)
    try interpreter.invoke()
    
    // 3. Postprocess - one line
    let output = try interpreter.output(at: 0)
    let quantParams = output.quantizationParameters!
    
    let result = try ClassificationOutput.process(
        outputData: output.data,
        quantization: .uint8(scale: quantParams.scale, zeroPoint: quantParams.zeroPoint),
        topK: 5,
        labels: .imagenet(hasBackgroundClass: true)
    )
    
    return result.predictions
}
```

### Complete TFLite YOLO Detection Example

Here's a complete example for YOLOv5 object detection with visualization:

```swift
import SwiftPixelUtils
import TensorFlowLite

func detectObjects(_ image: UIImage) async throws -> UIImage? {
    let modelWidth = 320
    let modelHeight = 320
    
    // 1. Preprocess - use .stretch for YOLO (direct coordinate mapping)
    let input = try await PixelExtractor.getModelInput(
        source: .uiImage(image),
        framework: .tfliteFloat,
        width: modelWidth,
        height: modelHeight,
        resizeStrategy: .stretch
    )
    
    // 2. Run inference
    let interpreter = try Interpreter(modelPath: yoloModelPath)
    try interpreter.allocateTensors()
    try interpreter.copy(input.data, toInputAt: 0)
    try interpreter.invoke()
    
    // 3. Get output as float array
    let output = try interpreter.output(at: 0)
    let floatOutput = output.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    
    // 4. Postprocess - one line with outputCoordinateSpace
    let result = try DetectionOutput.process(
        floatOutput: floatOutput,
        format: .yolov5(numClasses: 80),
        confidenceThreshold: 0.25,
        iouThreshold: 0.45,
        maxDetections: 20,
        labels: .coco,
        imageSize: image.size,
        modelInputSize: CGSize(width: modelWidth, height: modelHeight),
        outputCoordinateSpace: .normalized  // TFLite YOLO outputs 0-1 coords
    )
    
    // 5. Draw boxes - one line conversion to drawable format
    let boxes = result.toDrawableBoxes(imageSize: image.size)
    
    let drawingResult = try Drawing.drawBoxes(
        on: .uiImage(image),
        boxes: boxes,
        options: BoxDrawingOptions(lineWidth: 4, drawLabels: true, drawScores: true)
    )
    
    return UIImage(cgImage: drawingResult.cgImage, scale: image.scale, orientation: image.imageOrientation)
}
```

### `SegmentationOutput.process()` - One-Line Segmentation Postprocessing

Process semantic segmentation model output (DeepLabV3, etc.) with automatic parsing and visualization:

```swift
// Process DeepLabV3 output in one line
let result = try SegmentationOutput.process(
    floatOutput: outputArray,
    format: .logits(height: 257, width: 257, numClasses: 21),
    labels: .voc
)

// Get detected classes with coverage percentages
for item in result.classSummary {
    print("\(item.label): \(String(format: "%.1f%%", item.percentage))")
}

// Access the class mask
let classAtCenter = result.classAt(x: 128, y: 128)
print("Class at center: \(result.labels?[classAtCenter] ?? "unknown")")
```

#### Supported Segmentation Formats
- `.logits(height:width:numClasses:)` - Raw logits in NHWC format (DeepLabV3)
- `.logitsNCHW(height:width:numClasses:)` - Raw logits in NCHW format (PyTorch)
- `.probabilities(height:width:numClasses:)` - Softmax probabilities (NHWC)
- `.probabilitiesNCHW(height:width:numClasses:)` - Softmax probabilities (NCHW)
- `.argmax(height:width:numClasses:)` - Pre-computed class indices

#### Label Sources for Segmentation
- `.voc` - Pascal VOC (21 classes with background)
- `.ade20k` - ADE20K (150 classes)
- `.cityscapes` - Cityscapes (19 classes)
- `.custom([String])` - Your own label array
- `.none` - Returns "class_N" as labels

#### Visualizing Segmentation Results

Overlay colored segmentation mask on the original image:

```swift
// Overlay segmentation on image
let overlay = try Drawing.overlaySegmentation(
    on: .uiImage(image),
    segmentation: result,
    palette: .voc,           // Pascal VOC colors
    alpha: 0.5,              // 50% opacity
    excludeBackground: true  // Don't color background pixels
)

// Or create a standalone colored mask
let coloredMask = result.toColoredCGImage(palette: .voc)
```

#### Color Palettes

Built-in color palettes for common datasets:
- `SegmentationColorPalette.voc` - Pascal VOC (21 colors)
- `SegmentationColorPalette.ade20k` - ADE20K (150 colors)
- `SegmentationColorPalette.cityscapes` - Cityscapes (19 colors)
- `SegmentationColorPalette.rainbow(numClasses:)` - Generate custom palette

### Complete TFLite Segmentation Example

Here's a complete example for DeepLabV3 semantic segmentation:

```swift
import SwiftPixelUtils
import TensorFlowLite

func segmentImage(_ image: UIImage) async throws -> UIImage? {
    let modelSize = 257
    
    // 1. Preprocess
    let input = try await PixelExtractor.getModelInput(
        source: .uiImage(image),
        framework: .tfliteFloat,
        width: modelSize,
        height: modelSize
    )
    
    // 2. Run inference
    let interpreter = try Interpreter(modelPath: deeplabModelPath)
    try interpreter.allocateTensors()
    try interpreter.copy(input.data, toInputAt: 0)
    try interpreter.invoke()
    
    // 3. Get output
    let output = try interpreter.output(at: 0)
    let floatOutput = output.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    
    // 4. Postprocess - one line
    let result = try SegmentationOutput.process(
        floatOutput: floatOutput,
        format: .logits(height: modelSize, width: modelSize, numClasses: 21),
        labels: .voc
    )
    
    // 5. Visualize - overlay on original image
    let overlay = try Drawing.overlaySegmentation(
        on: .uiImage(image),
        segmentation: result,
        palette: .voc,
        alpha: 0.5,
        excludeBackground: true
    )
    
    return UIImage(cgImage: overlay.cgImage, scale: image.scale, orientation: image.imageOrientation)
}
```

## 📖 API Reference

### 🔧 Core Functions

#### `getPixelData(source:options:)`

Extract pixel data from a single image.

```swift
let result = try await PixelExtractor.getPixelData(
    source: .file(fileURL),
    options: PixelDataOptions(
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        colorFormat: .rgb,
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
)
```

#### `batchGetPixelData(sources:options:concurrency:)`

Process multiple images with concurrency control.

```swift
let results = try await PixelExtractor.batchGetPixelData(
    sources: [
        .url(URL(string: "https://example.com/1.jpg")!),
        .url(URL(string: "https://example.com/2.jpg")!),
        .url(URL(string: "https://example.com/3.jpg")!)
    ],
    options: ModelPresets.mobilenet,
    concurrency: 4
)
```

### 🔍 Image Analysis

#### `ImageAnalyzer.getStatistics(source:)`

Calculate image statistics for analysis and preprocessing decisions.

```swift
let stats = try await ImageAnalyzer.getStatistics(source: .file(fileURL))
print(stats.mean) // [r, g, b] mean values (0-1)
print(stats.std) // [r, g, b] standard deviations
print(stats.min) // [r, g, b] minimum values
print(stats.max) // [r, g, b] maximum values
print(stats.histogram) // RGB histograms
```

#### `ImageAnalyzer.getMetadata(source:)`

Get image metadata without loading full pixel data.

```swift
let metadata = try await ImageAnalyzer.getMetadata(source: .file(fileURL))
print(metadata.width)
print(metadata.height)
print(metadata.channels)
print(metadata.colorSpace)
print(metadata.hasAlpha)
```

#### `ImageAnalyzer.validate(source:options:)`

Validate an image against specified criteria.

```swift
let validation = try await ImageAnalyzer.validate(
    source: .file(fileURL),
    options: ValidationOptions(
        minWidth: 224,
        minHeight: 224,
        maxWidth: 4096,
        maxHeight: 4096,
        requiredAspectRatio: 1.0,
        aspectRatioTolerance: 0.1
    )
)

print(validation.isValid)   // true if passes all checks
print(validation.issues)    // Array of validation issues
```

#### `ImageAnalyzer.detectBlur(source:threshold:downsampleSize:)`

Detect if an image is blurry using Laplacian variance analysis.

```swift
let result = try await ImageAnalyzer.detectBlur(
    source: .file(fileURL),
    threshold: 100,
    downsampleSize: 500
)

print(result.isBlurry) // true if blurry
print(result.score) // Laplacian variance score
```

### 🎨 Image Augmentation

#### `ImageAugmentor.applyAugmentations(to:options:)`

Apply image augmentations for data augmentation pipelines.

```swift
let augmented = try await ImageAugmentor.applyAugmentations(
    to: .file(fileURL),
    options: AugmentationOptions(
        rotation: 15, // Degrees
        horizontalFlip: true,
        verticalFlip: false,
        brightness: 1.2, // 1.0 = no change
        contrast: 1.1,
        saturation: 0.9,
        blur: BlurOptions(type: .gaussian, radius: 2)
    )
)
```

#### `ImageAugmentor.colorJitter(source:options:)`

Apply color jitter augmentation with granular control.

```swift
let result = try await ImageAugmentor.colorJitter(
    source: .file(fileURL),
    options: ColorJitterOptions(
        brightness: 0.2,     // Random in [-0.2, +0.2]
        contrast: 0.2,       // Random in [0.8, 1.2]
        saturation: 0.3,     // Random in [0.7, 1.3]
        hue: 0.1,            // Random in [-0.1, +0.1]
        seed: 42             // For reproducibility
    )
)
```

#### `ImageAugmentor.cutout(source:options:)`

Apply cutout (random erasing) augmentation.

```swift
let result = try await ImageAugmentor.cutout(
    source: .file(fileURL),
    options: CutoutOptions(
        numCutouts: 1,
        minSize: 0.02,
        maxSize: 0.33,
        fillMode: .constant,
        fillValue: [0, 0, 0],
        seed: 42
    )
)
```

### 📦 Bounding Box Utilities

#### `BoundingBox.convertFormat(_:from:to:)`

Convert between bounding box formats (xyxy, xywh, cxcywh).

```swift
let boxes = [[320.0, 240.0, 100.0, 80.0]] // [cx, cy, w, h]
let converted = BoundingBox.convertFormat(
    boxes,
    from: .cxcywh,
    to: .xyxy
)
// [[270, 200, 370, 280]] - [x1, y1, x2, y2]
```

#### `BoundingBox.scale(_:from:to:format:)`

Scale bounding boxes between different image dimensions.

```swift
let scaled = BoundingBox.scale(
    [[100, 100, 200, 200]],
    from: CGSize(width: 640, height: 640),
    to: CGSize(width: 1920, height: 1080),
    format: .xyxy
)
```

#### `BoundingBox.clip(_:imageSize:format:)`

Clip bounding boxes to image boundaries.

```swift
let clipped = BoundingBox.clip(
    [[-10, 50, 700, 500]],
    imageSize: CGSize(width: 640, height: 480),
    format: .xyxy
)
```

#### `BoundingBox.calculateIoU(_:_:format:)`

Calculate Intersection over Union between two boxes.

```swift
let iou = BoundingBox.calculateIoU(
    [100, 100, 200, 200],
    [150, 150, 250, 250],
    format: .xyxy
)
```

#### `BoundingBox.nonMaxSuppression(detections:iouThreshold:scoreThreshold:maxDetections:)`

Apply Non-Maximum Suppression to filter overlapping detections.

```swift
let detections = [
    Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
    Detection(box: [110, 110, 210, 210], score: 0.8, classIndex: 0),
    Detection(box: [300, 300, 400, 400], score: 0.7, classIndex: 1)
]

let filtered = BoundingBox.nonMaxSuppression(
    detections: detections,
    iouThreshold: 0.5,
    scoreThreshold: 0.3,
    maxDetections: 100  // Optional limit
)
```

### 🧮 Tensor Operations

#### `TensorOperations.extractChannel(data:width:height:channels:channelIndex:dataLayout:)`

Extract a single channel from pixel data.

```swift
let redChannel = TensorOperations.extractChannel(
    data: result.data,
    width: result.width,
    height: result.height,
    channels: result.channels,
    channelIndex: 0, // 0=R, 1=G, 2=B
    dataLayout: result.dataLayout
)
```

#### `TensorOperations.permute(data:shape:order:)`

Transpose/permute tensor dimensions.

```swift
// Convert HWC to CHW
let permuted = TensorOperations.permute(
    data: result.data,
    shape: result.shape,
    order: [2, 0, 1] // new order: C, H, W
)
```

#### `TensorOperations.assembleBatch(results:layout:padToSize:)`

Assemble multiple results into a batch tensor.

```swift
let batch = TensorOperations.assembleBatch(
    results: [result1, result2, result3],
    layout: .nchw,
    padToSize: 4
)
print(batch.shape) // [3, 3, 224, 224] for NCHW
```

### 🎯 Quantization

#### `Quantizer.quantize(data:options:)`

Quantize float data to int8/uint8/int16/int4 format.

```swift
// Per-tensor quantization (standard)
let quantized = try Quantizer.quantize(
    data: result.data,
    options: QuantizationOptions(
        mode: .perTensor,
        dtype: .uint8,
        scale: [0.0078125],
        zeroPoint: [128]
    )
)
```

#### Per-Channel Quantization

Quantize with per-channel scale/zeroPoint for higher accuracy (ideal for CNN and Transformer weights):

```swift
// Per-channel quantization with calibration
let (scales, zeroPoints) = Quantizer.calibratePerChannel(
    data: weightsData,
    numChannels: 64,
    dtype: .int8,
    layout: .chw  // or .hwc
)

let quantized = try Quantizer.quantize(
    data: weightsData,
    options: QuantizationOptions(
        mode: .perChannel(axis: 0),  // Channel axis
        dtype: .int8,
        scale: scales,
        zeroPoint: zeroPoints
    )
)
// Result: int8Data with per-channel parameters for TFLite/ExecuTorch
```

#### INT4 Quantization (LLM/Edge Deployment)

4-bit quantization for 8× compression, ideal for LLM weights and edge devices:

```swift
// INT4 quantization (2 values packed per byte)
let int4Result = try Quantizer.quantize(
    data: llmWeights,
    options: QuantizationOptions(
        mode: .perTensor,
        dtype: .int4,  // or .uint4
        scale: [scale],
        zeroPoint: [zeroPoint]
    )
)

// Access packed data (50% size of INT8)
let packedBytes = int4Result.packedInt4Data!  // [UInt8] - 2 values per byte
let originalCount = int4Result.originalCount!  // Original element count
let compression = int4Result.compressionRatio  // 8.0 vs Float32

// Dequantize back to float
let restored = try Quantizer.dequantize(
    packedInt4Data: packedBytes,
    originalCount: originalCount,
    scale: [scale],
    zeroPoint: [zeroPoint],
    dtype: .int4
)
```

#### Quantization Types Comparison

| Type | Range | Compression | Use Case |
|------|-------|-------------|----------|
| `uint8` | [0, 255] | 4× | TFLite, general inference |
| `int8` | [-128, 127] | 4× | ExecuTorch, symmetric |
| `int16` | [-32768, 32767] | 2× | High precision |
| `int4` | [-8, 7] | 8× | LLM weights, edge |
| `uint4` | [0, 15] | 8× | Activation quantization |

#### `Quantizer.dequantize(uint8Data:scale:zeroPoint:)` / `dequantize(int8Data:...)`

Convert quantized data back to float.

```swift
let dequantized = try Quantizer.dequantize(
    uint8Data: quantizedArray,
    scale: [0.0078125],
    zeroPoint: [128]
)
```

### 🖼️ Letterbox Padding

#### `Letterbox.apply(to:options:)`

Apply letterbox padding to an image.

```swift
let options = LetterboxOptions(
    targetWidth: 640,
    targetHeight: 640,
    fillColor: (114, 114, 114),  // YOLO gray
    scaleUp: true,
    center: true
)

let result = try await Letterbox.apply(
    to: .file(fileURL),
    options: options
)

print(result.cgImage)      // Letterboxed CGImage
print(result.scale)        // Scale factor applied
print(result.offsetX)      // X padding offset
print(result.offsetY)      // Y padding offset
```

#### `Letterbox.reverseTransformBoxes(boxes:scale:offsetX:offsetY:)`

Transform detection boxes back to original image coordinates.

```swift
let originalBoxes = Letterbox.reverseTransformBoxes(
    boxes: detectedBoxes,
    scale: letterboxResult.scale,
    offsetX: letterboxResult.offsetX,
    offsetY: letterboxResult.offsetY
)
```

### 🎨 Drawing & Visualization

#### `Drawing.drawBoxes(on:boxes:options:)`

Draw bounding boxes with labels on an image.

```swift
let result = try Drawing.drawBoxes(
    on: .uiImage(image),
    boxes: [
        DrawableBox(box: [100, 100, 200, 200], label: "person", score: 0.95, color: (255, 0, 0, 255)),
        DrawableBox(box: [300, 150, 400, 350], label: "dog", score: 0.87, color: (0, 255, 0, 255))
    ],
    options: BoxDrawingOptions(
        lineWidth: 3,
        fontSize: 14,
        drawLabels: true,
        drawScores: true
    )
)

let annotatedImage = result.cgImage
```

### 🏷️ Label Database

Built-in label databases for common ML classification and detection models.

#### `LabelDatabase.getLabel(_:dataset:)`

Get a label by its class index.

```swift
let label = LabelDatabase.getLabel(0, dataset: .coco) // Returns "person"
```

#### `LabelDatabase.getTopLabels(scores:dataset:k:minConfidence:)`

Get top-K labels from prediction scores.

```swift
let topLabels = LabelDatabase.getTopLabels(
    scores: modelOutput,
    dataset: .coco,
    k: 5,
    minConfidence: 0.1
)
```

#### Available Datasets

| Dataset | Classes | Description |
|---------|---------|-------------|
| `coco` | 80 | COCO 2017 object detection labels |
| `coco91` | 91 | COCO original labels with background |
| `imagenet` | 1000 | ImageNet ILSVRC 2012 classification |
| `imagenet21k` | 21841 | ImageNet-21K full classification |
| `voc` | 21 | PASCAL VOC with background |
| `cifar10` | 10 | CIFAR-10 classification |
| `cifar100` | 100 | CIFAR-100 classification |
| `places365` | 365 | Places365 scene recognition |
| `ade20k` | 150 | ADE20K semantic segmentation |

## 📝 Type Reference

### Image Source Types

```swift
public enum ImageSource {
    case url(URL)                    // URL-based image source
    case file(URL)                   // Local file path
    case data(Data)                  // Raw image data
    case base64(String)              // Base64 encoded image
    case cgImage(CGImage)            // CGImage instance
    case uiImage(UIImage)            // UIImage (iOS)
    case nsImage(NSImage)            // NSImage (macOS)
}
```

### Color Formats

```swift
public enum ColorFormat {
    case rgb        // 3 channels
    case rgba       // 4 channels
    case bgr        // 3 channels (OpenCV style)
    case bgra       // 4 channels
    case grayscale  // 1 channel
    case hsv        // 3 channels (Hue, Saturation, Value)
    case hsl        // 3 channels (Hue, Saturation, Lightness)
    case lab        // 3 channels (CIE LAB)
    case yuv        // 3 channels
    case ycbcr      // 3 channels
}
```

### Resize Strategies

```swift
public enum ResizeStrategy {
    case cover      // Fill target, crop excess (default)
    case contain    // Fit within target, padding
    case stretch    // Stretch to fill (may distort)
    case letterbox  // Fit within target, letterbox padding (YOLO-style)
}
```

### Normalization

```swift
public enum NormalizationPreset {
    case scale      // [0, 1] range
    case imagenet   // ImageNet mean/std
    case tensorflow // [-1, 1] range
    case raw        // No normalization (0-255)
    case custom(mean: [Float], std: [Float]) // Custom mean/std
}
```

### Data Layouts

```swift
public enum DataLayout {
    case hwc   // Height × Width × Channels (default)
    case chw   // Channels × Height × Width (PyTorch)
    case nhwc  // Batch × Height × Width × Channels (TensorFlow/TFLite)
    case nchw  // Batch × Channels × Height × Width (PyTorch/ExecuTorch)
}
```

### Output Formats

```swift
public enum OutputFormat {
    case array        // Default float array (result.data)
    case float32Array // Float array (result.data) - same as array
    case uint8Array   // UInt8 array (result.uint8Data) - for quantized models
    case int32Array   // Int32 array (result.int32Data) - for integer models
}
```

**Usage:**
- `result.data` - Always populated with `[Float]` values
- `result.uint8Data` - Populated when `outputFormat: .uint8Array` or `normalization: .raw`
- `result.int32Data` - Populated when `outputFormat: .int32Array`

## ⚠️ Error Handling

All functions throw typed errors:

```swift
public enum PixelUtilsError: Error {
    case invalidSource(String)
    case loadFailed(String)
    case invalidROI(String)
    case processingFailed(String)
    case invalidOptions(String)
    case invalidChannel(String)
    case invalidPatch(String)
    case dimensionMismatch(String)
    case emptyBatch(String)
    case unknown(String)
}
```

## ⚡ Performance Tips

| Tip | Description |
|-----|-------------|
| 🤖 Simplified APIs | Use `getModelInput()`, `ClassificationOutput.process()`, `DetectionOutput.process()`, and `SegmentationOutput.process()` for the easiest integration |
| 🎯 Resize Strategies | Use letterbox for YOLO, cover for classification models |
| 📦 Batch Processing | Process multiple images concurrently for better performance |
| ⚙️ Model Presets | Pre-configured settings are optimized for each model |
| 🔄 Data Layout | Choose the right dataLayout upfront to avoid conversions |
| 🚀 Accelerate Framework | Leverage Apple's Accelerate for optimal performance |
| 🤖 ExecuTorch | Use NCHW layout + Int8 quantization for PyTorch-exported models |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

---

Made with ❤️ for iOS/macOS ML developers
