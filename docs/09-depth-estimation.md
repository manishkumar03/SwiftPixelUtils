# Depth Estimation Output Guide

Process depth estimation model outputs from MiDaS, DPT, ZoeDepth, and other monocular depth models.

## Table of Contents

1. [Overview](#overview)
2. [Supported Models](#supported-models)
3. [Model Downloading](#model-downloading)
4. [Processing Depth Output](#processing-depth-output)
5. [Visualization](#visualization)
6. [Depth Queries](#depth-queries)
7. [Colormaps](#colormaps)
8. [Complete Example](#complete-example)

## Overview

Monocular depth estimation predicts per-pixel depth from a single RGB image. SwiftPixelUtils provides:

- **Output processing** for common depth model formats
- **Visualization** with scientific colormaps (Viridis, Plasma, etc.)
- **Model downloading** with caching support
- **Depth queries** with bilinear interpolation

### Depth Types

| Type | Description | Models |
|------|-------------|--------|
| **Relative Inverse** | Higher values = closer, no metric scale | MiDaS, DPT |
| **Metric** | Depth in meters, lower values = closer | ZoeDepth |

## Supported Models

### MiDaS (Intel ISL)

| Variant | Input | Speed | Best For |
|---------|-------|-------|----------|
| MiDaS v2.1 Small | 256×256 | Fast | Mobile, real-time |
| MiDaS v2.1 | 384×384 | Medium | Balanced |
| MiDaS v3.0 DPT | 384×384 | Slow | High quality |

### DPT (Dense Prediction Transformer)

| Variant | Backbone | Quality |
|---------|----------|---------|
| DPT-Hybrid | ResNet50 + ViT | Good |
| DPT-Large | ViT-Large | Best |

### ZoeDepth (Zero-shot Transfer)

| Variant | Training Data | Use Case |
|---------|---------------|----------|
| ZoeD-N | NYU Depth v2 | Indoor |
| ZoeD-K | KITTI | Outdoor/driving |
| ZoeD-NK | Both | General |

## Model Downloading

SwiftPixelUtils can download and cache depth models automatically:

```swift
import SwiftPixelUtils

// Download MiDaS model
let modelURL = try await ModelDownloader.shared.download(
    model: .midasV21Small,
    progressHandler: { progress in
        print("Download: \(progress.percentComplete)%")
    }
)

// Load the model
let model = try MLModel(contentsOf: modelURL)
```

### Download and Load in One Call

```swift
let model = try await ModelDownloader.shared.loadModel(
    .midasV21Small,
    configuration: nil,
    progressHandler: { print("Progress: \($0.percentComplete)%") }
)
```

### Cache Management

```swift
// Check if model is cached
if await ModelDownloader.shared.isCached(model: .midasV21Small) {
    print("Model available offline")
}

// Get cache size
let size = await ModelDownloader.shared.cacheSize()
print("Cache: \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))")

// Clear cache
try await ModelDownloader.shared.clearCache()
```

### Custom Model URLs

```swift
let url = URL(string: "https://example.com/custom_depth.mlmodel")!
let modelURL = try await ModelDownloader.shared.download(
    from: url,
    modelId: "custom_depth"
)
```

## Processing Depth Output

### From Float Array

```swift
// After model inference
let depthOutput: [Float] = ... // Model output [H × W]

let result = try DepthEstimationOutput.process(
    output: depthOutput,
    width: 384,
    height: 384,
    modelType: .midas,
    originalWidth: 1920,  // For resize back to original
    originalHeight: 1080
)
```

### From 2D Array

```swift
let depthOutput2D: [[Float]] = ... // [height][width]

let result = try DepthEstimationOutput.process(
    output2D: depthOutput2D,
    modelType: .dptHybrid
)
```

### From Core ML MLMultiArray

```swift
// Direct from model output
let multiArray: MLMultiArray = modelOutput.featureValue(for: "depth")!.multiArrayValue!

let result = try DepthEstimationOutput.process(
    multiArray: multiArray,
    modelType: .midas,
    originalWidth: image.width,
    originalHeight: image.height
)
```

## Visualization

### Grayscale Depth Map

```swift
// Closer objects appear brighter
let grayscaleImage = result.toGrayscaleImage(invert: true)
```

### Colored Depth Map

```swift
// Using Viridis colormap (default)
let coloredImage = result.toColoredImage(colormap: .viridis)

// Using Plasma colormap
let plasmaImage = result.toColoredImage(colormap: .plasma, invert: true)

// Get platform image (UIImage/NSImage)
let image = result.toPlatformImage(colormap: .turbo)
```

### Resize to Original Dimensions

```swift
// Resize depth map to original image size
let fullSizeResult = result.resizedToOriginal()

// Or to specific size
let resized = result.resized(to: 1920, to: 1080)
```

## Depth Queries

### Point Queries

```swift
// Get depth at pixel coordinates
if let depth = result.depthAt(x: 100, y: 100) {
    print("Depth at (100, 100): \(depth)")
}

// Get depth at normalized coordinates (0-1)
if let depth = result.depthAtNormalized(normalizedX: 0.5, normalizedY: 0.5) {
    print("Depth at center: \(depth)")
}
```

### Statistics

```swift
let stats = result.statistics

print("Depth range: \(stats.min) - \(stats.max)")
print("Mean depth: \(stats.mean)")
print("Median: \(stats.median)")
print("Std dev: \(stats.stdDev)")
```

### Normalized Values

```swift
// Get 0-1 normalized depth
let normalized = result.normalized(invert: false)

// As 2D array
let depth2D = result.as2DArray()
```

## Colormaps

SwiftPixelUtils includes scientific colormaps for perceptually uniform depth visualization:

| Colormap | Description | Best For |
|----------|-------------|----------|
| **Viridis** | Blue → Green → Yellow | General purpose, colorblind-safe |
| **Plasma** | Blue → Purple → Orange → Yellow | High contrast |
| **Inferno** | Black → Purple → Orange → Yellow | Dark backgrounds |
| **Magma** | Black → Purple → Orange → White | Similar to Inferno |
| **Turbo** | Blue → Cyan → Green → Yellow → Red | Rainbow-like, improved |
| **Grayscale** | Black → White | Simple |
| **Jet** | Classic rainbow | Legacy (not perceptually uniform) |

```swift
// Available colormaps
let colormaps: [DepthColormap] = [
    .viridis,   // Recommended
    .plasma,
    .inferno,
    .magma,
    .turbo,
    .grayscale,
    .jet
]

// Use colormap
for colormap in colormaps {
    let image = result.toColoredImage(colormap: colormap)
    // Save or display image
}
```

### Colormap Comparison

```
Depth Value:  0.0 ──────────────────────────── 1.0

Viridis:      [Dark Blue] → [Teal] → [Green] → [Yellow]
Plasma:       [Dark Blue] → [Purple] → [Orange] → [Yellow]  
Inferno:      [Black] → [Purple] → [Orange] → [Yellow]
Turbo:        [Blue] → [Cyan] → [Green] → [Yellow] → [Red]
```

## Complete Example

### Depth Estimation Pipeline

```swift
import SwiftPixelUtils
import CoreML
import CoreImage

class DepthEstimator {
    private var model: MLModel?
    
    // Download and prepare model
    func setup() async throws {
        // Download model (cached after first download)
        model = try await ModelDownloader.shared.loadModel(
            .midasV21Small,
            progressHandler: { progress in
                print("Downloading: \(progress.percentComplete)%")
            }
        )
    }
    
    // Estimate depth for an image
    func estimateDepth(for image: CGImage) async throws -> DepthEstimationResult {
        guard let model = model else {
            throw PixelUtilsError.processingFailed("Model not loaded")
        }
        
        // Preprocess image (letterbox to model input size)
        let inputSize = (width: 256, height: 256)
        let preprocessed = try await Letterbox.apply(
            to: .cgImage(image),
            targetWidth: inputSize.width,
            targetHeight: inputSize.height
        )
        
        // Create MLFeatureProvider for input
        let input = try createInput(from: preprocessed.paddedImage)
        
        // Run inference
        let output = try model.prediction(from: input)
        
        // Get depth output
        guard let depthArray = output.featureValue(for: "depth")?.multiArrayValue else {
            throw PixelUtilsError.processingFailed("No depth output found")
        }
        
        // Process output
        return try DepthEstimationOutput.process(
            multiArray: depthArray,
            modelType: .midas,
            originalWidth: image.width,
            originalHeight: image.height
        )
    }
    
    // Visualize depth result
    func visualize(_ result: DepthEstimationResult) -> CGImage? {
        // Resize to original dimensions
        let fullSize = result.resizedToOriginal()
        
        // Create colored visualization
        return fullSize.toColoredImage(colormap: .viridis, invert: true)
    }
    
    private func createInput(from image: CGImage) throws -> MLFeatureProvider {
        // Implementation depends on model input format
        // Typically convert to CVPixelBuffer or MLMultiArray
        fatalError("Implement based on model requirements")
    }
}

// Usage
let estimator = DepthEstimator()
try await estimator.setup()

let image: CGImage = ... // Your input image
let depthResult = try await estimator.estimateDepth(for: image)

// Get depth at a point
if let centerDepth = depthResult.depthAtNormalized(normalizedX: 0.5, normalizedY: 0.5) {
    print("Center depth: \(centerDepth)")
}

// Create visualization
if let coloredDepth = estimator.visualize(depthResult) {
    // Display or save coloredDepth
}
```

### AR/3D Reconstruction Use Case

```swift
// For metric depth (ZoeDepth)
let result = try DepthEstimationOutput.process(
    output: modelOutput,
    width: 512,
    height: 384,
    modelType: .zoeDepth
)

// Get real-world distances
if result.isMetric {
    // Values are in meters
    for y in stride(from: 0, to: result.height, by: 10) {
        for x in stride(from: 0, to: result.width, by: 10) {
            if let depthMeters = result.depthAt(x: x, y: y) {
                // Use depth for 3D point cloud
                let point3D = (
                    x: Float(x),
                    y: Float(y),
                    z: depthMeters
                )
            }
        }
    }
}
```

## Model Comparison

| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| **MiDaS Small** | Fast, small | Lower resolution | Mobile, real-time |
| **MiDaS v2.1** | Good balance | Medium speed | General purpose |
| **DPT-Hybrid** | High quality | Slower, larger | Quality-focused |
| **DPT-Large** | Best quality | Slowest, ~350MB | Offline processing |
| **ZoeDepth** | Metric depth | Needs conversion | AR, 3D reconstruction |

## Performance Tips

1. **Cache models**: Download once, load from cache
2. **Batch processing**: Process multiple frames together if possible
3. **GPU acceleration**: Use `.all` compute units in MLModelConfiguration
4. **Resize wisely**: Process at model's native resolution, resize result
5. **Lazy loading**: Download models on-demand, not at app launch
