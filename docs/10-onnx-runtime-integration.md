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
10. [Complete Example](#complete-example)

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

// Create tensor data for YOLOv8
let tensorInput = try await ONNXHelper.createTensorData(
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
let tensorInput = try await ONNXHelper.createTensorData(
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
let input = try await PixelExtractor.getModelInput(
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

let batchTensor = try await ONNXHelper.createBatchTensorData(
    from: sources,
    config: .yolov8
)

print("Batch shape: \(batchTensor.shape)")  // [N, 3, 640, 640]

// Run batch inference with ONNX Runtime...
```

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
    
    func classify(image: UIImage) async throws -> [(label: String, confidence: Float)] {
        // 1. Preprocess image using SwiftPixelUtils
        let modelInput = try await PixelExtractor.getModelInput(
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
let results = try await classifier.classify(image: myImage)

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
    
    func detect(image: UIImage) async throws -> [Detection] {
        // 1. Preprocess image with letterbox
        let modelInput = try await PixelExtractor.getModelInput(
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
let detections = try await detector.detect(image: myImage)

for detection in detections {
    let label = LabelDatabase.getLabel(detection.classIndex, dataset: .coco) ?? "Unknown"
    print("\(label): \(detection.score * 100)% at \(detection.box)")
}
```

## Model Presets

For quick preprocessing without `ONNXHelper`, use model presets:

```swift
// Use preset options directly
let result = try await PixelExtractor.getPixelData(
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
let tensor = try await ONNXHelper.createTensorData(from: source, config: config)

if !tensor.isValid {
    print("Expected \(tensor.expectedDataSize) bytes, got \(tensor.data.count)")
}
```

### 3. Handle Letterbox Correctly

When using letterbox resize, save the transform info for accurate coordinate scaling:

```swift
// Get letterbox info during preprocessing
let result = try await PixelExtractor.getPixelData(
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
let batchTensor = try await ONNXHelper.createBatchTensorData(
    from: imageSources,
    config: .yolov8
)
```

## See Also

- [Image Preprocessing Fundamentals](01-image-preprocessing-fundamentals.md)
- [Detection Output](04-detection-output.md)
- [Classification Output](03-classification-output.md)
- [Segmentation Output](05-segmentation-output.md)
