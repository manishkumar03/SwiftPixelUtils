# SwiftPixelUtils - Quick Start Guide

## Installation

### Swift Package Manager

1. In Xcode, go to **File → Add Package Dependencies**
2. Enter the repository URL: `https://github.com/yourusername/SwiftPixelUtils.git`
3. Choose version and add to your target

Or add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/SwiftPixelUtils.git", from: "1.0.0")
]
```

## Basic Usage

### 1. Import the Library

```swift
import SwiftPixelUtils
```

### 2. Extract Pixel Data from an Image

```swift
// From URL
let url = URL(string: "https://example.com/image.jpg")!
let result = try await PixelExtractor.getPixelData(
    source: .url(url),
    options: PixelDataOptions()
)

print("Width: \\(result.width)")
print("Height: \\(result.height)")
print("Channels: \\(result.channels)")
print("Shape: \\(result.shape)")
print("Processing time: \\(result.processingTimeMs)ms")

// Access pixel data
let pixels = result.data // [Float] array
```

### 3. Use Model Presets for Common ML Models

```swift
// YOLO preprocessing (640x640, letterbox, NCHW layout)
let yoloResult = try await PixelExtractor.getPixelData(
    source: .file(fileURL),
    options: ModelPresets.yolov8
)

// MobileNet preprocessing (224x224, cover, ImageNet normalization, NHWC)
let mobileNetResult = try await PixelExtractor.getPixelData(
    source: .file(fileURL),
    options: ModelPresets.mobilenet
)

// ResNet preprocessing (224x224, cover, ImageNet normalization, NCHW)
let resnetResult = try await PixelExtractor.getPixelData(
    source: .file(fileURL),
    options: ModelPresets.resnet50
)
```

### 4. Custom Preprocessing Options

```swift
let options = PixelDataOptions(
    colorFormat: .rgb,
    resize: ResizeOptions(
        width: 224,
        height: 224,
        strategy: .cover
    ),
    roi: ROI(x: 0, y: 0, width: 500, height: 500), // Optional crop
    normalization: .imagenet,
    dataLayout: .nchw,
    outputFormat: .float32Array
)

let result = try await PixelExtractor.getPixelData(
    source: .file(fileURL),
    options: options
)
```

### 5. Batch Processing

```swift
let sources: [ImageSource] = [
    .url(URL(string: "https://example.com/1.jpg")!),
    .url(URL(string: "https://example.com/2.jpg")!),
    .url(URL(string: "https://example.com/3.jpg")!)
]

let results = try await PixelExtractor.batchGetPixelData(
    sources: sources,
    options: ModelPresets.mobilenet,
    concurrency: 4 // Process up to 4 images simultaneously
)

for (index, result) in results.enumerated() {
    print("Image \\(index + 1): \\(result.width)x\\(result.height)\")
}
```

## Bounding Box Operations

### Convert Box Formats

```swift
// Convert from YOLO format (center x, y, width, height) to corners
let yoloBoxes: [[Float]] = [[320, 240, 100, 80]]
let cornerBoxes = BoundingBox.convertFormat(
    yoloBoxes,
    from: .cxcywh,
    to: .xyxy
)
// Result: [[270, 200, 370, 280]]
```

### Scale Boxes to Different Image Size

```swift
let boxes: [[Float]] = [[100, 100, 200, 200]]
let scaledBoxes = BoundingBox.scale(
    boxes,
    from: CGSize(width: 640, height: 640),
    to: CGSize(width: 1920, height: 1080),
    format: .xyxy
)
```

### Non-Maximum Suppression (NMS)

```swift
let detections = [
    Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0, label: "person"),
    Detection(box: [110, 110, 210, 210], score: 0.8, classIndex: 0, label: "person"),
    Detection(box: [300, 300, 400, 400], score: 0.7, classIndex: 1, label: "car")
]

let filtered = BoundingBox.nonMaxSuppression(
    detections: detections,
    iouThreshold: 0.5,
    scoreThreshold: 0.3,
    maxDetections: 100
)
// Keeps non-overlapping detections
```

## Label Database

### Get Label by Index

```swift
let label = LabelDatabase.getLabel(0, dataset: .coco)
// Returns: "person"
```

### Get Top Predictions

```swift
let scores: [Float] = [0.9, 0.05, 0.7, 0.3, 0.1] // Model output
let topLabels = LabelDatabase.getTopLabels(
    scores: scores,
    dataset: .coco,
    k: 3,
    minConfidence: 0.1
)

for (label, confidence, index) in topLabels {
    print("\\(label): \\(confidence * 100)%")
}
```

## Complete ML Pipeline Example

```swift
import SwiftPixelUtils

func detectObjects(in imageURL: URL) async throws {
    // 1. Preprocess image for YOLO
    let preprocessed = try await PixelExtractor.getPixelData(
        source: .url(imageURL),
        options: ModelPresets.yolov8
    )
    
    // 2. Run your ML model (example)
    // let predictions = try runYOLOModel(preprocessed.data)
    
    // 3. Post-process detections
    let detections = [
        Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
        Detection(box: [110, 110, 210, 210], score: 0.85, classIndex: 0),
        Detection(box: [300, 150, 400, 350], score: 0.7, classIndex: 16)
    ]
    
    // 4. Apply NMS
    let filtered = BoundingBox.nonMaxSuppression(
        detections: detections,
        iouThreshold: 0.5,
        scoreThreshold: 0.3
    )
    
    // 5. Get labels
    for detection in filtered {
        if let label = LabelDatabase.getLabel(detection.classIndex, dataset: .coco) {
            print("Found \\(label) with confidence \\(detection.score)")
        }
    }
}
```

## Image Source Types

```swift
// From URL
.url(URL(string: "https://example.com/image.jpg")!)

// From file path
.file(URL(fileURLWithPath: "/path/to/image.jpg"))

// From Data
.data(imageData)

// From base64 string
.base64("data:image/png;base64,iVBORw0KG...")

// From CGImage
.cgImage(cgImage)

// From UIImage (iOS)
.uiImage(uiImage)

// From NSImage (macOS)
.nsImage(nsImage)
```

## Color Formats

- `.rgb` - 3 channels (most common)
- `.rgba` - 4 channels with alpha
- `.bgr` - BGR format (OpenCV style)
- `.bgra` - BGRA with alpha
- `.grayscale` - Single channel
- `.hsv` - Hue, Saturation, Value
- `.hsl` - Hue, Saturation, Lightness
- `.lab` - CIE LAB color space
- `.yuv` - YUV color space
- `.ycbcr` - YCbCr color space

## Resize Strategies

- `.cover` - Fill target size, crop excess (most common)
- `.contain` - Fit within target, add padding
- `.stretch` - Stretch to fill (may distort aspect ratio)
- `.letterbox` - Fit within target, letterbox padding (YOLO style)

## Normalization Presets

- `.scale` - Normalize to [0, 1]
- `.imagenet` - ImageNet mean/std (most classification models)
- `.tensorflow` - Normalize to [-1, 1]
- `.raw` - No normalization, keep [0, 255]
- `.custom(mean: [Float], std: [Float])` - Custom normalization

## Data Layouts

- `.hwc` - Height × Width × Channels (default)
- `.chw` - Channels × Height × Width (PyTorch)
- `.nhwc` - Batch × Height × Width × Channels (TensorFlow)
- `.nchw` - Batch × Channels × Height × Width (PyTorch batched)

## Error Handling

```swift
do {
    let result = try await PixelExtractor.getPixelData(
        source: .file(fileURL),
        options: ModelPresets.yolov8
    )
    // Process result
} catch let error as PixelUtilsError {
    switch error {
    case .invalidSource(let message):
        print("Invalid source: \\(message)")
    case .loadFailed(let message):
        print("Failed to load: \\(message)")
    case .processingFailed(let message):
        print("Processing failed: \\(message)")
    default:
        print("Error: \\(error.localizedDescription)")
    }
} catch {
    print("Unexpected error: \\(error)")
}
```

## Performance Tips

1. **Use batch processing** for multiple images
2. **Choose appropriate concurrency** based on device capabilities
3. **Use model presets** - they're optimized for each model
4. **Select the right resize strategy** - letterbox for YOLO, cover for classification
5. **Consider data layout** - choose NCHW for PyTorch, NHWC for TensorFlow
6. **Reuse options** - create PixelDataOptions once and reuse

## Next Steps

- Read the full [README.md](README.md) for complete API documentation
- Check [IMPLEMENTATION.md](IMPLEMENTATION.md) for architecture details
- See [Examples.swift](Sources/SwiftPixelUtils/Examples.swift) for more examples
- Run tests with `swift test`

## Support

For issues, questions, or contributions, please visit the GitHub repository.

---

Made with ❤️ for iOS/macOS ML developers
