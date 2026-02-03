# ONNX Runtime Integration Guide

SwiftPixelUtils provides comprehensive support for ONNX Runtime integration, making it easy to preprocess images and parse model outputs for ONNX models on iOS/macOS.

> **Note**: ONNX Runtime is not included in the SwiftPixelUtils example app due to symbol conflicts with TensorFlow Lite (both bundle XNNPACK internally). However, SwiftPixelUtils' ONNX helpers work seamlessly when you add ONNX Runtime to your own project.

## Table of Contents

1. [Overview](#overview)
2. [Adding ONNX Runtime to Your Project](#adding-onnx-runtime-to-your-project)
3. [MLFramework ONNX Variants](#mlframework-onnx-variants)
4. [Creating Tensor Data](#creating-tensor-data)
5. [Pre-configured Model Configs](#pre-configured-model-configs)
6. [Parsing Detection Output](#parsing-detection-output)
7. [Parsing Classification Output](#parsing-classification-output)
8. [Parsing Segmentation Output](#parsing-segmentation-output)
9. [Batch Inference](#batch-inference)
10. [Graph Optimization and Opsets](#graph-optimization-and-opsets)
11. [Decision Guide: ONNX Deployment Choices](#decision-guide-onnx-deployment-choices)
12. [Execution Providers and Threading](#execution-providers-and-threading)
13. [Complete Example](#complete-example)

## Overview

ONNX (Open Neural Network Exchange) is a popular format for sharing ML models across frameworks. ONNX Runtime provides high-performance inference on Apple Silicon. SwiftPixelUtils handles all the preprocessing complexity so you can focus on your application logic.

### Why SwiftPixelUtils for ONNX?

- **Correct Preprocessing**: NCHW layout, proper normalization, correct data types
- **Type Safety**: Strongly-typed tensor inputs with shape validation
- **Output Parsing**: Built-in parsers for common detection/classification formats
- **Performance**: Native implementations using Accelerate and vImage

## Adding ONNX Runtime to Your Project

### Option 1: CocoaPods

Add to your `Podfile`:

```ruby
pod 'onnxruntime-objc', '~> 1.16'
```

Then run:
```bash
pod install
```

> ⚠️ **Important**: Do NOT include `onnxruntime-objc` in projects that also use `TensorFlowLiteSwift`, as both frameworks bundle XNNPACK, causing duplicate symbol errors.

### Option 2: Swift Package Manager

Add to your `Package.swift` or in Xcode via File → Add Package Dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime", from: "1.16.0")
]
```

Or add directly in Xcode:
1. File → Add Package Dependencies
2. Enter: `https://github.com/microsoft/onnxruntime`
3. Select version: `1.16.0` or later

### Import Statement

```swift
import onnxruntime_objc
import SwiftPixelUtils
```

## MLFramework ONNX Variants

SwiftPixelUtils provides multiple ONNX framework variants for different model types:

```swift
public enum MLFramework {
    case onnx              // ImageNet normalization, Float32
    case onnxRaw           // [0,1] scale normalization, Float32
    case onnxQuantizedUInt8 // Raw [0,255], UInt8
    case onnxQuantizedInt8  // Raw [0,255], Int8
    case onnxFloat16       // ImageNet normalization, Float16
}
```

### Variant Comparison

| Variant | Layout | Normalization | Output Type | Use Case |
|---------|--------|---------------|-------------|----------|
| `.onnx` | NCHW | ImageNet | Float32 | Standard classification (ResNet, ViT) |
| `.onnxRaw` | NCHW | [0,1] scale | Float32 | YOLO detection models |
| `.onnxQuantizedUInt8` | NCHW | raw [0,255] | UInt8 | Quantized models (common) |
| `.onnxQuantizedInt8` | NCHW | raw [0,255] | Int8 | Signed quantized models |
| `.onnxFloat16` | NCHW | ImageNet | Float16 | Memory-optimized GPU models |

## Creating Tensor Data

### Using Model Config (Recommended)

```swift
import SwiftPixelUtils

// Create tensor data for YOLOv8 (synchronous)
let tensorInput = try ONNXHelper.createTensorData(
    from: .uiImage(image),
    config: .yolov8
)

print("Input name: \(tensorInput.name)")  // "images"
print("Shape: \(tensorInput.shape)")       // [1, 3, 640, 640]
print("Data type: \(tensorInput.dataType)") // .float32
print("Data size: \(tensorInput.data.count) bytes")
```

### Using Explicit Parameters

```swift
let tensorInput = try ONNXHelper.createTensorData(
    from: .uiImage(image),
    inputName: "input",
    width: 224,
    height: 224,
    framework: .onnx,
    dataType: .float32
)
```

### Using getModelInput()

For simple cases, you can use the unified API:

```swift
let input = try PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .onnx,
    width: 224,
    height: 224
)
// input.data contains Float32 values in NCHW layout with ImageNet normalization
```

## Pre-configured Model Configs

SwiftPixelUtils includes pre-configured settings for popular ONNX models:

### Detection Models

```swift
// YOLOv8 variants
ONNXModelConfig.yolov8   // [1, 3, 640, 640], Float32, [0,1] normalized
ONNXModelConfig.yolov8n
ONNXModelConfig.yolov8s
ONNXModelConfig.yolov8m
ONNXModelConfig.yolov8l
ONNXModelConfig.yolov8x

// RT-DETR
ONNXModelConfig.rtdetr   // [1, 3, 640, 640], Float32, ImageNet normalized
```

### Classification Models

```swift
ONNXModelConfig.resnet      // [1, 3, 224, 224], ImageNet normalized
ONNXModelConfig.mobilenetv2 // [1, 3, 224, 224], ImageNet normalized
ONNXModelConfig.vit         // [1, 3, 224, 224], ImageNet normalized
ONNXModelConfig.clip        // [1, 3, 224, 224], ImageNet normalized
```

### Creating Custom Configs

```swift
let customConfig = ONNXModelConfig(
    inputName: "input_tensor",
    inputShape: [1, 3, 512, 512],
    inputDataType: .float32,
    framework: .onnx,
    detectionFormat: nil  // nil for non-detection models
)
```

## Parsing Detection Output

### YOLOv8 Format

YOLOv8 outputs tensors in shape `[1, 84, 8400]` (for 80 COCO classes):

```swift
// Run ONNX inference...
let outputData: Data = runONNXInference(tensorInput)

// Parse detections
let detections = ONNXHelper.parseYOLOv8Output(
    data: outputData,
    numClasses: 80,
    confidenceThreshold: 0.25,
    nmsThreshold: 0.45
)

// Scale to original image coordinates
let scaledDetections = ONNXHelper.scaleDetections(
    detections,
    modelSize: CGSize(width: 640, height: 640),
    originalSize: image.size,
    letterboxInfo: letterboxInfo  // Optional, for letterboxed inputs
)

for detection in scaledDetections {
    print("Class \(detection.classIndex): \(detection.confidence)")
    print("Box: \(detection.box)")  // [x1, y1, x2, y2]
}
```

### RT-DETR Format

RT-DETR outputs `[1, 300, 4+num_classes]`:

```swift
let detections = ONNXHelper.parseRTDETROutput(
    data: outputData,
    numClasses: 80,
    confidenceThreshold: 0.5
)
```

### Generic Format

For YOLOv5 or other formats with shape `[1, num_boxes, 4+1+classes]`:

```swift
let detections = ONNXHelper.parseDetectionOutput(
    data: outputData,
    shape: [1, 25200, 85],
    format: .yoloV5,
    numClasses: 80,
    confidenceThreshold: 0.25,
    nmsThreshold: 0.45
)
```

### Detection Format Reference

| Format | Shape | Box Format | Scores |
|--------|-------|------------|--------|
| `.yoloV8` | [1, 4+C, N] | cx, cy, w, h | Needs sigmoid |
| `.yoloV5` | [1, N, 4+1+C] | cx, cy, w, h | objectness × class |
| `.rtdetr` | [1, 300, 4+C] | cx, cy, w, h | Softmax (no sigmoid) |
| `.ssd` | separate tensors | x1, y1, x2, y2 | Direct probabilities |

## Parsing Classification Output

```swift
let results = ONNXHelper.parseClassificationOutput(
    data: outputData,
    numClasses: 1000,
    topK: 5,
    applyingSoftmax: true  // Set false if model outputs probabilities
)

for (classIndex, probability) in results {
    let label = ImageNetLabels.labels[classIndex]
    print("\(label): \(probability * 100)%")
}
```

## Parsing Segmentation Output

For semantic segmentation models outputting `[1, num_classes, H, W]`:

```swift
let classIndices = ONNXHelper.parseSegmentationOutput(
    data: outputData,
    shape: [1, 21, 512, 512],  // 21 classes for Pascal VOC
    applyArgmax: true
)

// classIndices is [Int] of length H*W with class index per pixel
```

## Batch Inference

Process multiple images efficiently:

```swift
let images: [UIImage] = loadImages()
let sources = images.map { ImageSource.uiImage($0) }

let batchTensor = try ONNXHelper.createBatchTensorData(
    from: sources,
    config: .yolov8
)

print("Batch shape: \(batchTensor.shape)")  // [N, 3, 640, 640]

// Run batch inference with ONNX Runtime...
```

## Graph Optimization and Opsets

## Graph Optimization and Opsets

ONNX Runtime includes a powerful graph optimizer that rewrites your model graph at runtime for better performance.

### Optimization Levels
When configuring `ORTSessionOptions`, you can choose the optimization strategy:

1.  **Basic (`.basic`)**: Performs "safe" transformations like constant folding (calculating fixed values once), redundant node elimination, and identity removal. **Always enable this.**
2.  **Extended (`.extended`)**: Enables operator fusion (e.g., fusing `Conv` + `BatchNormalization` + `Relu` into a single kernel). This drastically reduces memory bandwidth usage.
3.  **All (`.all`)**: Enables hardware-specific optimizations, such as NCHWc memory layout transformations. This often yields the best performance but can increase initialization time.

### Opset Compatibility
The ONNX **Opset** version defines the available operators and their behavior.
- **Opset 11**: Considered the "stable" baseline. Most compatible with mobile frameworks.
- **Opset 17+**: Required for modern Transformer features (e.g., `LayerNormalization` specifics).
- **Pitfall**: Attempting to run a model exported with Opset 19 on an older ONNX Runtime version (e.g., 1.14) will fail. Always match your runtime library version to your export opset.

## Decision Guide: ONNX Deployment Choices

When should you use ONNX Runtime vs CoreML (Apple's native format)?

| Feature | Core ML (`.mlpackage`) | ONNX Runtime (`.onnx`) |
| :--- | :--- | :--- |
| **Hardware** | **Apple Neural Engine (ANE)**, GPU, CPU | Mostly CPU (on iOS), limited CoreML EP |
| **Power** | extremely Efficient | High (CPU usage) |
| **Workflow** | Requires conversion (`coremltools`) | Direct export from PyTorch (`torch.onnx.export`) |
| **Flexibility** | Rigid versions | Flexible, custom ops easier via C++ |

### Recommendation
1.  **Production iOS Apps**: Convert to **Core ML**. The power savings and ANE speedup (5x-10x) are critical for battery life.
2.  **Prototyping / Research**: Use **ONNX Runtime**. It allows you to test models immediately without fighting conversion errors.
3.  **Cross-Platform**: If you share code with Android/Windows, **ONNX Runtime** allows a single model file and unified preprocessing logic.

### Quantization Strategy
- **Float32**: Default. Best accuracy, largest size.
- **Dynamic Quantization (Weights)**: Compress weights to Int8, keep activations Float32. Great for LSTMs/Transformers. 2-4x smaller model.
- **Static Quantization**: Compress weights and activations. Requires a calibration dataset. Best for ConvNets on CPU.

## Execution Providers and Threading

ONNX Runtime abstracts hardware via **Execution Providers (EPs)**.

### The CPU Provider (Default)
Most efficient for general use on iOS if ANE is not accessible. Performance is sensitive to threading:

*   **`intra_op_num_threads`**: Controls parallelism *within* a single operator (e.g., splitting a large Matrix-Matrix multiplication across cores).
    *   *Tuning*: Set this to the number of **P-cores** (Performance cores) on the device. Setting it higher than physical cores causes context-switching thrashing and slows down inference.
    *   *Default*: ORT tries to guess, but often oversubscribes on iOS.
*   **`inter_op_num_threads`**: Controls parallelism of *independent* graph branches.
    *   *Tuning*: For sequential vision models (ResNet, YOLO), set to **1**. Higher values only help if the graph has many parallel paths.

### CoreML Execution Provider
You can force ONNX Runtime to delegate to Core ML:
```swift
let options = try ORTSessionOptions()
try options.appendExecutionProvider("CoreML") // Requires onnxruntime-c or similar
```
*   **Pros**: Access to ANE.
*   **Cons**: Compilation overhead on first run; "Silent Fallback" (if CoreML doesn't support an op, it falls back to CPU, causing expensive data copying).

## Complete Example

Here's a complete example for ResNet-50 image classification:

```swift
import SwiftPixelUtils
import onnxruntime_objc
import UIKit

class ResNetClassifier {
    private var session: ORTSession!
    private var env: ORTEnv!
    
    init(modelPath: String) throws {
        env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
    }
    
    func classify(image: UIImage) throws -> [(label: String, confidence: Float)] {
        // 1. Preprocess image using SwiftPixelUtils (synchronous)
        let modelInput = try PixelExtractor.getModelInput(
            source: .uiImage(image),
            framework: .onnx,  // Float32, NCHW, ImageNet normalized
            width: 224,
            height: 224
        )
        
        // 2. Create ONNX Runtime input tensor
        let inputShape: [NSNumber] = [1, 3, 224, 224]
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: modelInput.data),
            elementType: .float,
            shape: inputShape
        )
        
        // 3. Run inference
        let outputs = try session.run(
            withInputs: ["pixel_values": inputTensor],  // Check your model's input name
            outputNames: Set(["logits"]),               // Check your model's output name
            runOptions: nil
        )
        
        // 4. Get output data
        guard let outputTensor = outputs["logits"] else {
            throw NSError(domain: "ResNet", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "No output tensor"])
        }
        let outputData = try outputTensor.tensorData() as Data
        
        // 5. Parse classification output
        let predictions = ONNXHelper.parseClassificationOutput(
            data: outputData,
            numClasses: 1000,
            topK: 5,
            applyingSoftmax: true
        )
        
        // 6. Map to labels
        let labels = LabelDatabase.getAllLabels(for: .imagenet)
        return predictions.map { prediction in
            let label = prediction.classIndex < labels.count
                ? labels[prediction.classIndex]
                : "Class \(prediction.classIndex)"
            return (label: label, confidence: prediction.probability)
        }
    }
}

// Usage
let classifier = try ResNetClassifier(modelPath: "resnet50.onnx")
let results = try classifier.classify(image: myImage)

for (label, confidence) in results {
    print("\(label): \(confidence * 100)%")
}
```

### YOLOv8 Detection Example

```swift
import SwiftPixelUtils
import onnxruntime_objc
import UIKit

class YOLOv8Detector {
    private var session: ORTSession!
    private var env: ORTEnv!
    
    init(modelPath: String) throws {
        env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
    }
    
    func detect(image: UIImage) throws -> [Detection] {
        // 1. Preprocess image with letterbox (synchronous)
        let modelInput = try PixelExtractor.getModelInput(
            source: .uiImage(image),
            framework: .onnxRaw,  // Float32, NCHW, [0,1] scale
            width: 640,
            height: 640,
            resizeStrategy: .letterbox
        )
        
        // 2. Create ONNX Runtime input tensor
        let inputShape: [NSNumber] = [1, 3, 640, 640]
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: modelInput.data),
            elementType: .float,
            shape: inputShape
        )
        
        // 3. Run inference
        let outputs = try session.run(
            withInputs: ["images": inputTensor],
            outputNames: Set(["output0"]),
            runOptions: nil
        )
        
        // 4. Get output data
        guard let outputTensor = outputs["output0"] else {
            throw NSError(domain: "YOLOv8", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "No output tensor"])
        }
        let outputData = try outputTensor.tensorData() as Data
        
        // 5. Parse detections
        let detections = ONNXHelper.parseYOLOv8Output(
            data: outputData,
            numClasses: 80,
            confidenceThreshold: 0.25,
            nmsThreshold: 0.45
        )
        
        // 6. Scale to original image coordinates
        return ONNXHelper.scaleDetections(
            detections,
            modelSize: CGSize(width: 640, height: 640),
            originalSize: image.size,
            letterboxInfo: modelInput.letterboxInfo
        )
    }
}

// Usage
let detector = try YOLOv8Detector(modelPath: "yolov8n.onnx")
let detections = try detector.detect(image: myImage)

for detection in detections {
    let label = LabelDatabase.getLabel(detection.classIndex, dataset: .coco) ?? "Unknown"
    print("\(label): \(detection.score * 100)% at \(detection.box)")
}
```

## Model Presets

For quick preprocessing without `ONNXHelper`, use model presets:

```swift
// Use preset options directly (synchronous)
let result = try PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: ModelPresets.onnx_yolov8
)

// Available ONNX presets
ModelPresets.onnx_yolov8       // YOLOv8 detection
ModelPresets.onnx_rtdetr       // RT-DETR detection
ModelPresets.onnx_resnet       // ResNet classification
ModelPresets.onnx_mobilenetv2  // MobileNetV2 classification
ModelPresets.onnx_vit          // Vision Transformer
ModelPresets.onnx_clip         // CLIP vision encoder
ModelPresets.onnx_quantized_uint8  // Quantized UInt8
ModelPresets.onnx_quantized_int8   // Quantized Int8
ModelPresets.onnx_float16      // Float16 optimized
```

## Tips and Best Practices

### 1. Use the Right Framework Variant

```swift
// Classification with ImageNet normalization
framework: .onnx

// YOLO detection (0-1 scaling)
framework: .onnxRaw

// Quantized models
framework: .onnxQuantizedUInt8
```

### 2. Validate Tensor Data

```swift
let tensor = try ONNXHelper.createTensorData(from: source, config: config)

if !tensor.isValid {
    print("Expected \(tensor.expectedDataSize) bytes, got \(tensor.data.count)")
}
```

### 3. Handle Letterbox Correctly

When using letterbox resize, save the transform info for accurate coordinate scaling:

```swift
// Get letterbox info during preprocessing (synchronous)
let result = try PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: ModelPresets.onnx_yolov8
)

// Use letterbox info for scaling
let scaled = ONNXHelper.scaleDetections(
    detections,
    modelSize: CGSize(width: 640, height: 640),
    originalSize: image.size,
    letterboxInfo: result.letterboxInfo
)
```

### 4. Batch Processing for Throughput

For high-throughput scenarios, use batch processing:

```swift
let batchTensor = try ONNXHelper.createBatchTensorData(
    from: imageSources,
    config: .yolov8
)
```

## See Also

- [Image Preprocessing Fundamentals](01-image-preprocessing-fundamentals.md)
- [Detection Output](04-detection-output.md)
- [Classification Output](03-classification-output.md)
- [Segmentation Output](05-segmentation-output.md)
