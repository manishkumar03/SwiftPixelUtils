# SwiftPixelUtils

<p align="center">
  <strong>High-performance Swift library for image preprocessing optimized for ML/AI inference pipelines on iOS/macOS</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#features">Features</a>
</p>

---

High-performance Swift library for image preprocessing optimized for ML/AI inference pipelines. Native implementations using Core Image, Accelerate, and Core ML for pixel extraction, tensor conversion, quantization, augmentation, and model-specific preprocessing (YOLO, MobileNet, etc.)

## ✨ Features

- 🚀 **High Performance**: Native implementations using Apple frameworks (Core Image, Accelerate, vImage, Core ML)
- 🤖 **Simplified ML APIs**: One-line preprocessing (`getModelInput`) and postprocessing (`ClassificationOutput`) for all major frameworks
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
- 🎯 **Native Quantization**: Float→Int8/UInt8/Int16 with per-tensor and per-channel support (TFLite/ExecuTorch compatible)
- 🏷️ **Label Database**: Built-in labels for COCO, ImageNet, VOC, CIFAR, Places365, ADE20K
- 📦 **Bounding Box Utilities**: Format conversion (xyxy/xywh/cxcywh), scaling, clipping, IoU, NMS
- 🖼️ **Letterbox Padding**: YOLO-style letterbox preprocessing with reverse coordinate transform
- 🎨 **Drawing/Visualization**: Draw boxes, keypoints, masks, and heatmaps for debugging
- 🔲 **Grid/Patch Extraction**: Extract image patches in grid patterns for sliding window inference
- 🎲 **Random Crop with Seed**: Reproducible random crops for data augmentation pipelines
- ✅ **Tensor Validation**: Validate tensor shapes, dtypes, and value ranges before inference
- 📦 **Batch Assembly**: Combine multiple images into NCHW/NHWC batch tensors

## 📦 Installation

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

### Complete TFLite Example

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

#### `getImageStatistics(source:)`

Calculate image statistics for analysis and preprocessing decisions.

```swift
let stats = try await ImageAnalyzer.getStatistics(source: .file(fileURL))
print(stats.mean) // [r, g, b] mean values (0-1)
print(stats.std) // [r, g, b] standard deviations
print(stats.min) // [r, g, b] minimum values
print(stats.max) // [r, g, b] maximum values
print(stats.histogram) // RGB histograms
```

#### `getImageMetadata(source:)`

Get image metadata without loading full pixel data.

```swift
let metadata = try await ImageAnalyzer.getMetadata(source: .file(fileURL))
print(metadata.width)
print(metadata.height)
print(metadata.channels)
print(metadata.colorSpace)
print(metadata.hasAlpha)
```

#### `validateImage(source:options:)`

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
```

#### `detectBlur(source:threshold:downsampleSize:)`

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

#### `applyAugmentations(source:augmentations:)`

Apply image augmentations for data augmentation pipelines.

```swift
let augmented = try await ImageAugmentor.apply(
    source: .file(fileURL),
    augmentations: AugmentationOptions(
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

#### `colorJitter(source:options:)`

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

#### `cutout(source:options:)`

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

#### `convertBoxFormat(_:from:to:)`

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

#### `scaleBoxes(_:from:to:format:)`

Scale bounding boxes between different image dimensions.

```swift
let scaled = BoundingBox.scale(
    [[100, 100, 200, 200]],
    from: CGSize(width: 640, height: 640),
    to: CGSize(width: 1920, height: 1080),
    format: .xyxy
)
```

#### `clipBoxes(_:imageSize:format:)`

Clip bounding boxes to image boundaries.

```swift
let clipped = BoundingBox.clip(
    [[-10, 50, 700, 500]],
    imageSize: CGSize(width: 640, height: 480),
    format: .xyxy
)
```

#### `calculateIoU(_:_:format:)`

Calculate Intersection over Union between two boxes.

```swift
let iou = BoundingBox.calculateIoU(
    [100, 100, 200, 200],
    [150, 150, 250, 250],
    format: .xyxy
)
```

#### `nonMaxSuppression(detections:iouThreshold:scoreThreshold:maxDetections:)`

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
    maxDetections: 100
)
```

### 🧮 Tensor Operations

#### `extractChannel(data:width:height:channels:channelIndex:dataLayout:)`

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

#### `permute(data:shape:order:)`

Transpose/permute tensor dimensions.

```swift
// Convert HWC to CHW
let permuted = TensorOperations.permute(
    data: result.data,
    shape: result.shape,
    order: [2, 0, 1] // new order: C, H, W
)
```

#### `assembleBatch(results:layout:padToSize:)`

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

#### `quantize(data:mode:dtype:scale:zeroPoint:)`

Quantize float data to int8/uint8/int16 format.

```swift
let quantized = try Quantizer.quantize(
    data: result.data,
    mode: .perTensor,
    dtype: .uint8,
    scale: 0.0078125,
    zeroPoint: 128
)
```

#### `dequantize(data:mode:dtype:scale:zeroPoint:)`

Convert quantized data back to float.

```swift
let dequantized = try Quantizer.dequantize(
    data: quantized.data,
    mode: .perTensor,
    dtype: .int8,
    scale: 0.0078125,
    zeroPoint: 0
)
```

### 🖼️ Letterbox Padding

#### `letterbox(source:targetSize:fillColor:)`

Apply letterbox padding to an image.

```swift
let result = try await Letterbox.apply(
    source: .file(fileURL),
    targetSize: CGSize(width: 640, height: 640),
    fillColor: [114, 114, 114]
)

print(result.image) // Letterboxed UIImage/NSImage
print(result.info) // Transform info for reverse mapping
```

#### `reverseLetterbox(boxes:info:format:)`

Transform detection boxes back to original image coordinates.

```swift
let originalBoxes = Letterbox.reverse(
    boxes: detectedBoxes,
    info: result.info,
    format: .xyxy
)
```

### 🎨 Drawing & Visualization

#### `drawBoxes(image:boxes:options:)`

Draw bounding boxes with labels on an image.

```swift
let visualized = try Drawing.drawBoxes(
    image: image,
    boxes: [
        BoxAnnotation(box: [100, 100, 200, 200], label: "person", score: 0.95, classIndex: 0),
        BoxAnnotation(box: [300, 150, 400, 350], label: "dog", score: 0.87, classIndex: 16)
    ],
    options: DrawingOptions(
        lineWidth: 3,
        fontSize: 14,
        drawLabels: true
    )
)
```

### 🏷️ Label Database

Built-in label databases for common ML classification and detection models.

#### `getLabel(index:dataset:)`

Get a label by its class index.

```swift
let label = LabelDatabase.getLabel(0, dataset: .coco) // Returns "person"
```

#### `getTopLabels(scores:dataset:k:minConfidence:)`

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
| 🤖 Simplified APIs | Use `getModelInput()` and `ClassificationOutput.process()` for the easiest integration |
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
