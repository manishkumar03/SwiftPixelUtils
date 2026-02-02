# Image Preprocessing Fundamentals for ML

A comprehensive reference for image preprocessing concepts, techniques, and implementation patterns for machine learning inference on Apple platforms.

## Table of Contents

- [Image Preprocessing Fundamentals for ML](#image-preprocessing-fundamentals-for-ml)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Why Preprocessing Matters](#why-preprocessing-matters)
    - [Common Preprocessing Failures](#common-preprocessing-failures)
    - [The Cost of Incorrect Preprocessing](#the-cost-of-incorrect-preprocessing)
  - [The ML Image Pipeline](#the-ml-image-pipeline)
    - [SwiftPixelUtils Simplifies This](#swiftpixelutils-simplifies-this)
  - [Pixel Data Extraction](#pixel-data-extraction)
    - [What is Pixel Data?](#what-is-pixel-data)
    - [Memory Representation](#memory-representation)
      - [Row-Major Order (Standard)](#row-major-order-standard)
      - [Interleaved vs Planar](#interleaved-vs-planar)
    - [Data Types](#data-types)
    - [Bit Depth and Dynamic Range](#bit-depth-and-dynamic-range)
    - [Alpha Channel Handling](#alpha-channel-handling)
  - [Color Formats](#color-formats)
    - [RGB and RGBA](#rgb-and-rgba)
    - [BGR and BGRA](#bgr-and-bgra)
    - [Grayscale](#grayscale)
    - [HSV/HSL Color Spaces](#hsvhsl-color-spaces)
    - [YUV/YCbCr Color Spaces](#yuvycbcr-color-spaces)
    - [LAB Color Space](#lab-color-space)
    - [Color Space Conversion Mathematics](#color-space-conversion-mathematics)
      - [RGB to Grayscale](#rgb-to-grayscale)
      - [RGB to HSV](#rgb-to-hsv)
      - [RGB to YUV (BT.601)](#rgb-to-yuv-bt601)
  - [Resizing Strategies](#resizing-strategies)
    - [Stretch (Distort)](#stretch-distort)
    - [Contain (Fit)](#contain-fit)
    - [Cover (Crop)](#cover-crop)
    - [Letterbox (YOLO-style)](#letterbox-yolo-style)
    - [Interpolation Methods](#interpolation-methods)
      - [Nearest Neighbor](#nearest-neighbor)
      - [Bilinear Interpolation](#bilinear-interpolation)
      - [Bicubic Interpolation](#bicubic-interpolation)
      - [Lanczos Resampling](#lanczos-resampling)
    - [Anti-Aliasing Considerations](#anti-aliasing-considerations)
  - [Normalization](#normalization)
    - [Why Normalize?](#why-normalize)
    - [Min-Max Normalization \[0, 1\]](#min-max-normalization-0-1)
    - [Symmetric Normalization \[-1, 1\]](#symmetric-normalization--1-1)
    - [ImageNet Normalization](#imagenet-normalization)
    - [Per-Channel Statistics](#per-channel-statistics)
    - [Batch Normalization vs Input Normalization](#batch-normalization-vs-input-normalization)
  - [Data Layouts](#data-layouts)
    - [HWC vs CHW](#hwc-vs-chw)
    - [NHWC vs NCHW](#nhwc-vs-nchw)
    - [Framework Layout Requirements](#framework-layout-requirements)
    - [Memory Layout Performance](#memory-layout-performance)
    - [Layout Conversion](#layout-conversion)
  - [Batch Processing](#batch-processing)
    - [Creating Batches](#creating-batches)
    - [Memory Management](#memory-management)
    - [Concurrency Control](#concurrency-control)
  - [Framework-Specific Requirements](#framework-specific-requirements)
    - [TensorFlow Lite](#tensorflow-lite)
    - [Core ML](#core-ml)
    - [PyTorch Mobile / ExecuTorch](#pytorch-mobile--executorch)
    - [ONNX Runtime](#onnx-runtime)
  - [Platform-Specific Image Sources](#platform-specific-image-sources)
    - [UIImage (iOS)](#uiimage-ios)
    - [NSImage (macOS)](#nsimage-macos)
    - [CGImage (Cross-platform)](#cgimage-cross-platform)
    - [CVPixelBuffer (Camera/Video)](#cvpixelbuffer-cameravideo)
    - [CIImage (Core Image)](#ciimage-core-image)
  - [Performance Optimization](#performance-optimization)
    - [Accelerate Framework](#accelerate-framework)
    - [Metal Performance Shaders](#metal-performance-shaders)
    - [Memory Alignment](#memory-alignment)
    - [Avoiding Copies](#avoiding-copies)
  - [Common Pitfalls and Debugging](#common-pitfalls-and-debugging)
    - [Checklist for Debugging Bad Results](#checklist-for-debugging-bad-results)
    - [Visual Debugging](#visual-debugging)
    - [Print Tensor Statistics](#print-tensor-statistics)
  - [SwiftPixelUtils API Reference](#swiftpixelutils-api-reference)
    - [PixelExtractor](#pixelextractor)
    - [ImageSource](#imagesource)
    - [MLFramework](#mlframework)
    - [PixelDataOptions](#pixeldataoptions)
  - [Mathematical Foundations](#mathematical-foundations)
    - [Linear Algebra of Image Transformations](#linear-algebra-of-image-transformations)
    - [Bilinear Interpolation Formula](#bilinear-interpolation-formula)
    - [Color Space Conversion Matrices](#color-space-conversion-matrices)

---

## Introduction

Image preprocessing is the critical bridge between how images exist in your application (UIImage, camera frames, files) and how machine learning models expect them (normalized float tensors in specific layouts). Getting preprocessing wrong is the #1 cause of poor model performance in production.

This guide serves as both a learning resource and a reference manual for image preprocessing in ML applications on Apple platforms.

---

## Why Preprocessing Matters

Machine learning models are extremely sensitive to input format. A model trained on ImageNet-normalized RGB images will produce garbage results if you feed it:

- **BGR instead of RGB** - Color channels swapped
- **Unnormalized [0-255] instead of [-1, 1]** - Wrong value range
- **Wrong dimensions or aspect ratio** - Spatial distortion
- **Incorrect data layout (HWC vs CHW)** - Misinterpreted tensor structure

### Common Preprocessing Failures

| Problem | Symptom | Root Cause |
|---------|---------|------------|
| Color channel mismatch | Random/wrong predictions | BGR/RGB confusion |
| Wrong normalization | Extremely high/low confidence | ImageNet vs [0,1] mismatch |
| Aspect ratio distortion | Poor accuracy on tall/wide images | Stretch instead of letterbox |
| Integer overflow | Crash or NaN values | UInt8 where Float expected |
| Layout mismatch | Garbled results | HWC vs CHW confusion |
| Missing alpha removal | Tinted predictions | RGBA fed as RGB |
| Wrong resolution | Poor accuracy | Model trained on different size |

### The Cost of Incorrect Preprocessing

```
Correct preprocessing:   Model accuracy: 94.2%
Wrong color channels:    Model accuracy: 12.1% (near random)
Wrong normalization:     Model accuracy: 31.4%
Wrong aspect ratio:      Model accuracy: 67.8%
```

---

## The ML Image Pipeline

```
┌─────────────────┐
│  Image Source   │  UIImage, CGImage, URL, Camera, CVPixelBuffer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Decode Image   │  Decompress JPEG/PNG/HEIC to raw pixels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Color Convert  │  RGBA → RGB, BGR, Grayscale
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Resize Image   │  Stretch, Contain, Cover, Letterbox
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Normalize      │  [0,255] → [0,1] or [-1,1] or ImageNet
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Layout Convert │  HWC → CHW, add batch dimension
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Quantize       │  Float32 → Int8/UInt8 (if needed)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Input    │  Tensor ready for inference
└─────────────────┘
```

### SwiftPixelUtils Simplifies This

```swift
// One-line preprocessing - handles all steps automatically
let input = try await PixelExtractor.getModelInput(
    source: .url(imageURL),
    framework: .tfliteFloat,
    width: 224,
    height: 224
)

// Run inference
interpreter.copy(input.data, toInputAt: 0)
try interpreter.invoke()

// One-line postprocessing
let results = try ClassificationOutput.process(outputData: output, labels: .imagenet)
```

---

## Pixel Data Extraction

### What is Pixel Data?

An image is a 2D grid of pixels. Each pixel contains color information, typically as Red, Green, and Blue (RGB) intensity values. Understanding this representation is fundamental to ML preprocessing.

```
Image (3×3 pixels, RGB):
┌───────────────┬───────────────┬───────────────┐
│  R:255 G:0    │  R:0   G:255  │  R:0   G:0    │
│  B:0          │  B:0          │  B:255        │
│  (Red)        │  (Green)      │  (Blue)       │
├───────────────┼───────────────┼───────────────┤
│  R:128 G:128  │  R:64  G:64   │  R:192 G:192  │
│  B:128        │  B:64         │  B:192        │
│  (Mid Gray)   │  (Dark Gray)  │  (Light Gray) │
├───────────────┼───────────────┼───────────────┤
│  R:0   G:0    │  R:255 G:255  │  R:128 G:0    │
│  B:0          │  B:255        │  B:128        │
│  (Black)      │  (White)      │  (Purple)     │
└───────────────┴───────────────┴───────────────┘
```

### Memory Representation

In memory, 2D image data becomes a 1D array. The layout determines how pixels are ordered:

#### Row-Major Order (Standard)
```
Pixels stored row by row, left to right, top to bottom:

Row 0: [P(0,0), P(0,1), P(0,2)]
Row 1: [P(1,0), P(1,1), P(1,2)]
Row 2: [P(2,0), P(2,1), P(2,2)]

Memory: [P(0,0), P(0,1), P(0,2), P(1,0), P(1,1), P(1,2), P(2,0), P(2,1), P(2,2)]
```

#### Interleaved vs Planar

**Interleaved (HWC):** All channels for one pixel together
```
[R₀₀,G₀₀,B₀₀, R₀₁,G₀₁,B₀₁, R₁₀,G₁₀,B₁₀, ...]
 └───pixel 0───┘ └───pixel 1───┘
```

**Planar (CHW):** All pixels for one channel, then next channel
```
[R₀₀,R₀₁,R₁₀,R₁₁, G₀₀,G₀₁,G₁₀,G₁₁, B₀₀,B₀₁,B₁₀,B₁₁]
 └───Red plane────┘ └───Green plane──┘ └───Blue plane───┘
```

### Data Types

| Type | Range | Size | Use Case |
|------|-------|------|----------|
| `UInt8` | 0-255 | 1 byte | Raw pixels, quantized models |
| `Int8` | -128 to 127 | 1 byte | Quantized TFLite models |
| `UInt16` | 0-65535 | 2 bytes | HDR images, medical imaging |
| `Float16` | ±65504 | 2 bytes | CoreML, Metal, GPU inference |
| `Float32` | ±3.4×10³⁸ | 4 bytes | Most ML models, training |
| `Float64` | ±1.8×10³⁰⁸ | 8 bytes | Scientific (rarely needed) |

### Bit Depth and Dynamic Range

**8-bit (256 levels):** Standard images
```
0 = black, 255 = white
256 possible values per channel
256³ = 16.7 million colors
```

**10-bit (1024 levels):** HDR displays
```
4× more levels than 8-bit
Used in ProRes, HEVC HDR
```

**16-bit (65536 levels):** Professional/medical
```
256× more levels than 8-bit
Scientific imaging, RAW photos
```

### Alpha Channel Handling

Many iOS images include an alpha (transparency) channel:

```
RGBA pixel: [R, G, B, A]
            [255, 128, 64, 255]  ← Fully opaque orange
            [255, 128, 64, 128]  ← 50% transparent orange
            [255, 128, 64, 0]    ← Fully transparent
```

**For ML preprocessing, you typically need to:**

1. **Remove alpha:** Most models expect RGB, not RGBA
2. **Handle premultiplied alpha:** iOS often stores premultiplied RGBA
3. **Composite over background:** Transparent pixels need a background color

```swift
// Remove alpha channel
let rgb = try await PixelExtractor.getPixelData(
    source: .uiImage(imageWithAlpha),
    options: PixelDataOptions(colorFormat: .rgb)  // Alpha removed
)

// Composite over white background
let rgb = try await PixelExtractor.getPixelData(
    source: .uiImage(imageWithAlpha),
    options: PixelDataOptions(
        colorFormat: .rgb,
        alphaHandling: .compositeOver(backgroundColor: (255, 255, 255))
    )
)
```

---

## Color Formats

### RGB and RGBA

**RGB (Red, Green, Blue)** - The most common format for ML models.

```
Channel order: [R, G, B]
Memory layout: [R₀, G₀, B₀, R₁, G₁, B₁, ...]

Example pixel values (orange):
R = 255, G = 165, B = 0
```

**RGBA (with Alpha)** - Native format for most iOS images.

```
Channel order: [R, G, B, A]
Memory layout: [R₀, G₀, B₀, A₀, R₁, G₁, B₁, A₁, ...]

A = 255 (opaque) or 0-254 (transparent)
```

**Usage:**
- TensorFlow/TFLite: RGB
- PyTorch: RGB
- Most classification/detection models: RGB
- iOS native images: RGBA (convert to RGB for ML)

### BGR and BGRA

**BGR (Blue, Green, Red)** - OpenCV's native format.

```
Channel order: [B, G, R]

Same pixel, different order:
RGB: [255, 165, 0]   ← Orange
BGR: [0, 165, 255]   ← Same orange, different representation
```

**Usage:**
- OpenCV-trained models
- Some face detection models
- Computer vision research code

**Why BGR exists:** Historical reasons from early video hardware.

```swift
// Convert to BGR for OpenCV-trained models
let bgr = try await PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: PixelDataOptions(colorFormat: .bgr)
)
```

### Grayscale

Single-channel intensity values.

```
Channel order: [Y]  (luminance)
Values: 0 (black) to 255 (white)

Conversion from RGB:
Y = 0.299×R + 0.587×G + 0.114×B  (ITU-R BT.601)
Y = 0.2126×R + 0.7152×G + 0.0722×B  (ITU-R BT.709, modern displays)
```

**Why these weights?** Human eyes are most sensitive to green, less to red, least to blue.

**Usage:**
- Document scanning
- Edge detection
- Some medical imaging
- Efficiency (1/3 the data)

```swift
let gray = try await PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: PixelDataOptions(colorFormat: .grayscale)
)
```

### HSV/HSL Color Spaces

**HSV (Hue, Saturation, Value):**

```
H = Hue (0-360°): Color position on color wheel
    0° = Red, 120° = Green, 240° = Blue
    
S = Saturation (0-100%): Color purity
    0% = Gray, 100% = Pure color
    
V = Value (0-100%): Brightness
    0% = Black, 100% = Full brightness
```

**HSL (Hue, Saturation, Lightness):**
```
Similar to HSV but L=50% is pure color, L=100% is white
```

**Usage:**
- Color-based segmentation (finding specific colors)
- Color filtering
- Image editing

```
HSV separates chromatic (H, S) from achromatic (V) information
```

### YUV/YCbCr Color Spaces

**YUV (Luminance, Chrominance):**

```
Y  = Luminance (brightness) - What black & white TVs showed
U  = Blue chrominance (Cb) 
V  = Red chrominance (Cr)

Conversion from RGB:
Y  =  0.299×R + 0.587×G + 0.114×B
U  = -0.147×R - 0.289×G + 0.436×B  (or 0.492×(B-Y))
V  =  0.615×R - 0.515×G - 0.100×B  (or 0.877×(R-Y))
```

**Usage:**
- Video compression (JPEG, MPEG, H.264)
- Camera sensor output (often YUV 4:2:0)
- Some real-time vision models
- Efficient chroma subsampling

**Chroma Subsampling:**
```
4:4:4 - Full resolution for Y, U, V
4:2:2 - Full Y, half horizontal U/V
4:2:0 - Full Y, quarter U/V (most common for video)
```

### LAB Color Space

**CIELAB (L*a*b*):**

```
L* = Lightness (0-100)
a* = Green-Red axis (-128 to +127)
b* = Blue-Yellow axis (-128 to +127)
```

**Why LAB?**
- Perceptually uniform: Equal distances = equal perceived differences
- Device-independent
- Used for color difference calculations (ΔE)

**Usage:**
- Color matching
- Image similarity
- Color correction

### Color Space Conversion Mathematics

#### RGB to Grayscale

```
// ITU-R BT.601 (SDTV)
Y = 0.299 × R + 0.587 × G + 0.114 × B

// ITU-R BT.709 (HDTV)
Y = 0.2126 × R + 0.7152 × G + 0.0722 × B

// Simple average (fast, less accurate)
Y = (R + G + B) / 3
```

#### RGB to HSV

```swift
func rgbToHsv(r: Float, g: Float, b: Float) -> (h: Float, s: Float, v: Float) {
    let maxVal = max(r, g, b)
    let minVal = min(r, g, b)
    let delta = maxVal - minVal
    
    // Value
    let v = maxVal
    
    // Saturation
    let s = maxVal == 0 ? 0 : delta / maxVal
    
    // Hue
    var h: Float = 0
    if delta != 0 {
        switch maxVal {
        case r: h = 60 * (((g - b) / delta).truncatingRemainder(dividingBy: 6))
        case g: h = 60 * (((b - r) / delta) + 2)
        case b: h = 60 * (((r - g) / delta) + 4)
        default: break
        }
    }
    if h < 0 { h += 360 }
    
    return (h, s, v)
}
```

#### RGB to YUV (BT.601)

```swift
func rgbToYuv(r: Float, g: Float, b: Float) -> (y: Float, u: Float, v: Float) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b
    let u = -0.14713 * r - 0.28886 * g + 0.436 * b
    let v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return (y, u, v)
}
```

---

## Resizing Strategies

Models have fixed input dimensions (e.g., 224×224, 640×640). Your input images are rarely that exact size. The resizing strategy significantly impacts model accuracy.

### Stretch (Distort)

Simply resize to target dimensions, ignoring aspect ratio.

```
Original (4:3 = 800×600)         Stretched (1:1 = 640×640)
┌────────────────────┐           ┌──────────────┐
│                    │           │              │
│   ○    ○           │           │  ◯  ◯        │
│      👃            │    →      │    👃       │  Objects appear
│   \_____/          │           │  \___/       │  horizontally
│                    │           │              │  squished
└────────────────────┘           └──────────────┘
```

**Pros:**
- Simple implementation
- Uses full input tensor area
- No padding artifacts

**Cons:**
- Distorts object shapes
- Hurts accuracy for shape-sensitive tasks
- Model may not recognize distorted objects

**Best for:**
- Textures and patterns (aspect ratio doesn't matter)
- When training data was also stretched
- Quick prototyping

```swift
let stretched = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteFloat,
    width: 224,
    height: 224,
    resizeStrategy: .stretch
)
```

### Contain (Fit)

Scale to fit entirely within target dimensions, maintaining aspect ratio. Pad remaining space.

```
Original (16:9)                  Contained (1:1)
┌──────────────────────┐         ┌──────────────┐
│                      │         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← Padding (gray/black)
│    🚗    🏠          │   →     │    🚗  🏠   │
│                      │         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← Padding
└──────────────────────┘         └──────────────┘
```

**Pros:**
- No shape distortion
- Entire image visible

**Cons:**
- Wasted tensor area (padding contains no information)
- May slightly reduce effective resolution

**Best for:**
- When shape matters (object detection, face recognition)
- General-purpose preprocessing
- When unsure which method to use

```swift
let contained = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteFloat,
    width: 224,
    height: 224,
    resizeStrategy: .contain(padding: .gray)  // or .black, .white, .reflect
)
```

### Cover (Crop)

Scale to completely cover target dimensions, cropping overflow.

```
Original (4:3)                    Covered (1:1)
┌────────────────────┐            ┌──────────────┐
│  ░░░░░░░░░░░░░░░   │            │              │
│  ░              ░  │            │              │
│  ░  🎯 Subject  ░  │     →      │  🎯 Subject  │  ← Center preserved
│  ░              ░  │            │              │
│  ░░░░░░░░░░░░░░░   │            │              │
└────────────────────┘            └──────────────┘
        ↑ Cropped areas
```

**Pros:**
- No padding, no distortion
- Full tensor utilization
- Maximum effective resolution

**Cons:**
- May crop important content at edges
- Assumes subject is centered

**Best for:**
- Classification where subject is centered
- Portrait/face photos
- When edges are unlikely to contain important content

```swift
let covered = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteFloat,
    width: 224,
    height: 224,
    resizeStrategy: .cover(alignment: .center)  // or .top, .bottom
)
```

### Letterbox (YOLO-style)

Special variant of contain optimized for object detection. Maintains aspect ratio with configurable padding color.

```
Original (16:9)                  Letterboxed (1:1)
┌──────────────────────┐         ┌──────────────┐
│                      │         │░░░░░░░░░░░░░░│ ← Gray (114,114,114)
│  🚗  🚶  🐕  🏠     │   →     │  🚗 🚶 🐕 🏠│
│                      │         │░░░░░░░░░░░░░░│ ← Gray (114,114,114)
└──────────────────────┘         └──────────────┘
```

**Critical for YOLO:** YOLO models are trained with letterboxing using gray (114, 114, 114) padding. Using stretch or different padding colors will significantly hurt detection accuracy.

**Pros:**
- Preserves aspect ratio
- Trained behavior for YOLO
- Predictable coordinate transformation

**Cons:**
- Requires reverse transformation for output coordinates
- Slightly more complex pipeline

**Letterbox Transformation:**
```swift
// Forward transform: original → letterboxed
let letterboxed = try Letterbox.apply(
    to: image,
    targetSize: CGSize(width: 640, height: 640),
    color: (114, 114, 114)  // YOLO standard gray
)

// letterboxed.image: the padded image
// letterboxed.info: transformation metadata

// After detection, reverse transform to get original coordinates
let originalCoords = Letterbox.reverseTransform(
    boxes: detections.boxes,
    letterboxInfo: letterboxed.info
)
```

**Automatic Letterbox Info with getPixelData:**

When using `getPixelData` with letterbox resize, transform metadata is automatically captured:

```swift
let result = try await PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: PixelDataOptions(
        resize: ResizeOptions(width: 640, height: 640, strategy: .letterbox),
        colorFormat: .rgb,
        normalization: .scale,
        dataLayout: .nchw
    )
)

// Transform info is now available in the result
if let info = result.letterboxInfo {
    print("Scale: \(info.scale)")           // e.g., 0.5 if image was scaled down
    print("Offset: \(info.offset)")         // e.g., (0, 80) for top/bottom padding
    print("Original: \(info.originalSize)") // Original image dimensions
    
    // Reverse transform detection coordinates
    func reverseTransform(x: Float, y: Float) -> (Float, Float) {
        let origX = (x - Float(info.offset.x)) / info.scale
        let origY = (y - Float(info.offset.y)) / info.scale
        return (origX, origY)
    }
}
```

### Interpolation Methods

When resizing, how do we calculate new pixel values?

#### Nearest Neighbor
```
Pick the closest pixel. Fast but blocky.

Original:    Upscaled 2×:
[A][B]       [A][A][B][B]
[C][D]   →   [A][A][B][B]
             [C][C][D][D]
             [C][C][D][D]
```

**Use for:** Pixel art, speed-critical applications, integer scale factors

#### Bilinear Interpolation
```
Weighted average of 4 nearest pixels.

For point P at (x, y):
P = (1-dx)(1-dy)×A + dx(1-dy)×B + (1-dx)dy×C + dxdy×D

where dx, dy are fractional distances
```

**Use for:** General purpose, good quality/speed tradeoff

#### Bicubic Interpolation
```
Weighted average of 16 nearest pixels using cubic function.
Smoother than bilinear, better edge preservation.
```

**Use for:** High-quality resizing, publication images

#### Lanczos Resampling
```
Uses sinc function windowed by Lanczos window.
Best quality for downsampling, preserves sharpness.
```

**Use for:** Maximum quality, photo editing

**SwiftPixelUtils Default:** Bilinear for upsampling, Lanczos for downsampling.

### Anti-Aliasing Considerations

When downsampling significantly, high-frequency details can cause aliasing (moiré patterns, jagged edges).

```
Without anti-aliasing:         With anti-aliasing:
[Striped pattern]              [Smooth gradient]
  ↓ Downsample                   ↓ Blur then downsample
[Moiré artifacts]              [Clean result]
```

SwiftPixelUtils applies appropriate anti-aliasing automatically based on scale factor.

---

## Normalization

### Why Normalize?

Raw pixel values (0-255) are not ideal for neural networks:

1. **Numerical stability:** Large values can cause overflow in activations
2. **Gradient flow:** Centered data helps gradient descent converge
3. **Model compatibility:** Must match training normalization exactly
4. **Feature scaling:** Ensures all features have similar magnitude

### Min-Max Normalization [0, 1]

The simplest normalization - divide by maximum value.

```
normalized = pixel / 255.0

Input:  [0, 128, 255]
Output: [0.0, 0.502, 1.0]

Range: [0.0, 1.0]
Mean: ~0.5 (for typical images)
```

**Mathematical form:**
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} = \frac{x}{255}$$

**Used by:**
- TensorFlow examples
- Many TFLite models
- Simple custom models

```swift
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteFloat,  // Uses [0,1] normalization
    width: 224,
    height: 224
)
```

### Symmetric Normalization [-1, 1]

Center values around zero for better gradient flow.

```
normalized = (pixel / 127.5) - 1.0
// or equivalently:
normalized = (pixel - 127.5) / 127.5

Input:  [0, 128, 255]
Output: [-1.0, 0.004, 1.0]

Range: [-1.0, 1.0]
Mean: ~0 (for typical images)
```

**Mathematical form:**
$$x_{norm} = \frac{2x}{255} - 1 = \frac{x - 127.5}{127.5}$$

**Used by:**
- TensorFlow official models
- StyleGAN and image generation models
- Many PyTorch models

```swift
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .custom(normalization: .symmetricMinusOneToOne),
    width: 224,
    height: 224
)
```

### ImageNet Normalization

Per-channel mean subtraction and standard deviation division, based on ImageNet dataset statistics.

```
ImageNet statistics (computed over 1.2M images):
mean = [0.485, 0.456, 0.406]  // RGB
std  = [0.229, 0.224, 0.225]  // RGB

normalized[channel] = (pixel[channel]/255.0 - mean[channel]) / std[channel]

Example (orange pixel [255, 165, 0]):
R: (255/255 - 0.485) / 0.229 = 2.249
G: (165/255 - 0.456) / 0.224 = 0.850
B: (0/255 - 0.406) / 0.225 = -1.804
```

**Mathematical form:**
$$x_{norm}^{(c)} = \frac{x^{(c)}/255 - \mu^{(c)}}{\sigma^{(c)}}$$

**Range:** Approximately [-2.5, 2.5] for typical images

**Used by:**
- **All PyTorch pretrained models** (ResNet, EfficientNet, ViT, etc.)
- torchvision transforms
- Models trained with transfer learning from ImageNet

```swift
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .pytorchMobile,  // Automatically uses ImageNet normalization
    width: 224,
    height: 224
)
```

### Per-Channel Statistics

Different datasets have different statistics:

| Dataset | Mean (RGB) | Std (RGB) |
|---------|------------|-----------|
| ImageNet | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
| CIFAR-10 | [0.4914, 0.4822, 0.4465] | [0.2470, 0.2435, 0.2616] |
| CIFAR-100 | [0.5071, 0.4867, 0.4408] | [0.2675, 0.2565, 0.2761] |
| COCO | [0.471, 0.448, 0.408] | [0.234, 0.239, 0.242] |

**Custom statistics:**
```swift
let customNorm = Normalization.custom(
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5]  // Results in [-1, 1] range
)

let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .custom(normalization: customNorm),
    width: 224,
    height: 224
)
```

### Batch Normalization vs Input Normalization

Don't confuse these:

**Input Normalization (what we're discussing):**
- Applied once to input data
- Fixed statistics from training set
- Happens in preprocessing

**Batch Normalization (model layer):**
- Applied to hidden layer activations
- Learned parameters (γ, β)
- Part of the model itself

Both can coexist and serve different purposes.

---

## Data Layouts

### HWC vs CHW

**HWC (Height × Width × Channels)** - Interleaved format

```
For a 2×2 RGB image:

Memory layout:
[R₀₀,G₀₀,B₀₀, R₀₁,G₀₁,B₀₁, R₁₀,G₁₀,B₁₀, R₁₁,G₁₁,B₁₁]
 ←─ pixel(0,0) ─→ ←─ pixel(0,1) ─→ ←─ pixel(1,0) ─→ ←─ pixel(1,1) ─→

Shape: [height, width, channels] = [2, 2, 3]

Index calculation: data[y * width * channels + x * channels + c]
```

**CHW (Channels × Height × Width)** - Planar format

```
For a 2×2 RGB image:

Memory layout:
[R₀₀,R₀₁,R₁₀,R₁₁, G₀₀,G₀₁,G₁₀,G₁₁, B₀₀,B₀₁,B₁₀,B₁₁]
 ←─── Red plane ────→ ←─── Green plane ──→ ←─── Blue plane ───→

Shape: [channels, height, width] = [3, 2, 2]

Index calculation: data[c * height * width + y * width + x]
```

### NHWC vs NCHW

Add batch dimension for multiple images:

**NHWC:** `[batch, height, width, channels]` - TensorFlow default
**NCHW:** `[batch, channels, height, width]` - PyTorch default

```
Batch of 2 images, 224×224 RGB:

NHWC shape: [2, 224, 224, 3]
NCHW shape: [2, 3, 224, 224]

Memory for NHWC: [img0_R₀₀,img0_G₀₀,img0_B₀₀,...,img1_R₀₀,img1_G₀₀,img1_B₀₀,...]
Memory for NCHW: [img0_R_plane,img0_G_plane,img0_B_plane,img1_R_plane,...]
```

### Framework Layout Requirements

| Framework | Layout | Notes |
|-----------|--------|-------|
| TensorFlow | NHWC | Default, most operations |
| TensorFlow Lite | NHWC | Most models |
| PyTorch | NCHW | Default, cuDNN optimized |
| PyTorch Mobile | NCHW | Same as PyTorch |
| ExecuTorch | NCHW | Same as PyTorch |
| CoreML | Varies | Check model input spec |
| ONNX | NCHW | Default convention |
| OpenCV | HWC | Row-major, no batch |

### Memory Layout Performance

**NCHW advantages:**
- Better memory locality for per-channel operations
- cuDNN optimization on NVIDIA GPUs
- Efficient for convolution implementations

**NHWC advantages:**
- Natural for pixel-wise operations
- Better for CPU SIMD operations (process all channels at once)
- iOS native image format

**In practice:** The difference is usually small. Match the model's expected format.

### Layout Conversion

```swift
// HWC to CHW
func hwcToChw(_ hwc: [Float], height: Int, width: Int, channels: Int) -> [Float] {
    var chw = [Float](repeating: 0, count: hwc.count)
    for y in 0..<height {
        for x in 0..<width {
            for c in 0..<channels {
                let hwcIndex = y * width * channels + x * channels + c
                let chwIndex = c * height * width + y * width + x
                chw[chwIndex] = hwc[hwcIndex]
            }
        }
    }
    return chw
}

// CHW to HWC
func chwToHwc(_ chw: [Float], channels: Int, height: Int, width: Int) -> [Float] {
    var hwc = [Float](repeating: 0, count: chw.count)
    for c in 0..<channels {
        for y in 0..<height {
            for x in 0..<width {
                let chwIndex = c * height * width + y * width + x
                let hwcIndex = y * width * channels + x * channels + c
                hwc[hwcIndex] = chw[chwIndex]
            }
        }
    }
    return hwc
}
```

SwiftPixelUtils handles layout conversion automatically based on the specified framework.

---

## Batch Processing

### Creating Batches

Processing multiple images together can improve throughput:

```swift
let images: [UIImage] = [image1, image2, image3, image4]

let batchResult = try await PixelExtractor.getBatchPixelData(
    sources: images.map { .uiImage($0) },
    options: PixelDataOptions(
        targetSize: CGSize(width: 224, height: 224),
        dataLayout: .nchw
    )
)

// batchResult.data: [Float] with shape [4, 3, 224, 224]
// batchResult.shape: [4, 3, 224, 224]
```

### Memory Management

Batch processing uses significant memory:

```
Single image:  224 × 224 × 3 × 4 bytes = 602 KB
Batch of 32:   32 × 224 × 224 × 3 × 4 bytes = 19.3 MB
Batch of 128:  128 × 224 × 224 × 3 × 4 bytes = 77.2 MB
```

For large batches, use streaming or chunked processing:

```swift
// Process in chunks to manage memory
func processLargeBatch(images: [UIImage], chunkSize: Int = 16) async throws -> [[Float]] {
    var results: [[Float]] = []
    
    for chunk in images.chunked(into: chunkSize) {
        autoreleasepool {
            let batchResult = try await PixelExtractor.getBatchPixelData(
                sources: chunk.map { .uiImage($0) },
                options: options
            )
            results.append(batchResult.data)
        }
    }
    
    return results
}
```

### Concurrency Control

Control parallel processing to balance speed and memory:

```swift
let results = try await PixelExtractor.getBatchPixelData(
    sources: hundredsOfImages,
    options: options,
    maxConcurrency: 4  // Process up to 4 images simultaneously
)
```

---

## Framework-Specific Requirements

### TensorFlow Lite

```swift
// TFLite Float32 model
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteFloat,  // RGB, [0,1], NHWC, Float32
    width: 224,
    height: 224
)

// Copy to interpreter
try interpreter.copy(Data(bytes: input.data, count: input.data.count * 4), toInputAt: 0)
```

**TFLite Quantized (UInt8):**
```swift
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteQuantized,  // RGB, [0,255], NHWC, UInt8
    width: 224,
    height: 224
)

// Copy to interpreter
try interpreter.copy(Data(input.dataUInt8), toInputAt: 0)
```

### Core ML

```swift
// CoreML typically expects MLMultiArray or CVPixelBuffer
let pixelBuffer = try await PixelExtractor.getCVPixelBuffer(
    source: .uiImage(image),
    targetSize: CGSize(width: 224, height: 224)
)

// Use directly with CoreML
let prediction = try model.prediction(image: pixelBuffer)
```

### PyTorch Mobile / ExecuTorch

```swift
// PyTorch expects ImageNet normalization, NCHW, Float32
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .pytorchMobile,  // RGB, ImageNet norm, NCHW, Float32
    width: 224,
    height: 224
)

// Shape: [1, 3, 224, 224]
```

### ONNX Runtime

```swift
// ONNX typically uses NCHW
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .onnx,  // RGB, [0,1], NCHW, Float32
    width: 224,
    height: 224
)
```

---

## Platform-Specific Image Sources

### UIImage (iOS)

```swift
// From UIImage
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(myUIImage),
    framework: .tfliteFloat,
    width: 224,
    height: 224
)
```

**UIImage orientation handling:**

UIImage may have EXIF orientation metadata that causes silent rotations when accessing `.cgImage`. SwiftPixelUtils provides built-in orientation normalization:

```swift
// Option 1: Enable automatic orientation normalization (recommended)
let result = try await PixelExtractor.getPixelData(
    source: .uiImage(myUIImage),
    options: PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalizeOrientation: true  // Fixes EXIF rotation automatically
    )
)

// Option 2: Manual normalization (if needed elsewhere)
func normalizeOrientation(_ image: UIImage) -> UIImage {
    if image.imageOrientation == .up { return image }
    
    UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
    image.draw(in: CGRect(origin: .zero, size: image.size))
    let normalized = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()
    return normalized
}
```

### NSImage (macOS)

```swift
#if os(macOS)
let input = try await PixelExtractor.getModelInput(
    source: .nsImage(myNSImage),
    framework: .tfliteFloat,
    width: 224,
    height: 224
)
#endif
```

### CGImage (Cross-platform)

```swift
// CGImage is the lowest-level, most reliable source
let input = try await PixelExtractor.getModelInput(
    source: .cgImage(myCGImage),
    framework: .tfliteFloat,
    width: 224,
    height: 224
)
```

### CVPixelBuffer (Camera/Video)

```swift
// Direct from camera or video
func captureOutput(_ output: AVCaptureOutput, 
                   didOutput sampleBuffer: CMSampleBuffer, 
                   from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    
    let input = try await PixelExtractor.getModelInput(
        source: .cvPixelBuffer(pixelBuffer),
        framework: .tfliteFloat,
        width: 640,
        height: 640
    )
    
    // Run inference on camera frame...
}
```

**CVPixelBuffer formats:**
```swift
// Common CVPixelBuffer formats from camera
kCVPixelFormatType_32BGRA      // Most common on iOS
kCVPixelFormatType_420YpCbCr8BiPlanarFullRange  // Video, efficient
kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
// RGB565 formats (16-bit)
kCVPixelFormatType_16LE565
kCVPixelFormatType_16BE565
```

### CIImage (Core Image)

```swift
// CIImage from Core Image pipeline
let ciImage = CIImage(image: myUIImage)!
    .applyingFilter("CIColorControls", parameters: ["inputBrightness": 0.1])

let input = try await PixelExtractor.getModelInput(
    source: .ciImage(ciImage),
    framework: .tfliteFloat,
    width: 224,
    height: 224
)
```

---

## Performance Optimization

### Accelerate Framework

SwiftPixelUtils uses Apple's Accelerate framework internally for SIMD-optimized operations:

```swift
// vImage for fast image operations
import Accelerate

// Color conversion
vImageConvert_RGBA8888toRGB888(...)

// Resize
vImageScale_ARGB8888(...)

// Type conversion
vDSP_vfltu8(...)  // UInt8 to Float
```

### Metal Performance Shaders

For GPU-accelerated preprocessing:

```swift
// Metal-based resize (much faster for large images)
let metalPreprocessor = MetalImagePreprocessor(device: MTLCreateSystemDefaultDevice()!)
let input = try metalPreprocessor.preprocess(
    image: image,
    targetSize: CGSize(width: 640, height: 640)
)
```

### Memory Alignment

Aligned memory access is faster:

```swift
// Allocate aligned memory for better performance
func allocateAligned<T>(count: Int, alignment: Int = 64) -> UnsafeMutablePointer<T> {
    let byteCount = count * MemoryLayout<T>.stride
    let pointer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
    return pointer.bindMemory(to: T.self, capacity: count)
}
```

### Avoiding Copies

Minimize memory copies for performance:

```swift
// Use withUnsafeBytes to avoid copies when possible
pixelData.withUnsafeBytes { rawBuffer in
    let floatBuffer = rawBuffer.bindMemory(to: Float.self)
    // Use floatBuffer directly...
}
```

---

## Common Pitfalls and Debugging

### Checklist for Debugging Bad Results

1. **Color channels:** Is the model expecting RGB or BGR?
2. **Normalization:** [0,1]? [-1,1]? ImageNet?
3. **Layout:** HWC or CHW? NHWC or NCHW?
4. **Data type:** Float32? UInt8? Int8?
5. **Resolution:** Does it match the model's input size?
6. **Aspect ratio:** Stretch? Letterbox? Contain?
7. **Alpha channel:** Is it being removed?
8. **Orientation:** Is the image rotated correctly?

### Visual Debugging

```swift
// Save intermediate results to inspect
func debugSaveImage(_ data: [Float], width: Int, height: Int, name: String) {
    // Convert normalized floats back to viewable image
    let uint8Data = data.map { UInt8(max(0, min(255, $0 * 255))) }
    
    // Create image and save
    let image = createImage(from: uint8Data, width: width, height: height)
    saveToDocuments(image, name: "\(name).png")
}
```

### Print Tensor Statistics

```swift
func printTensorStats(_ data: [Float], name: String) {
    print("\(name):")
    print("  Shape: \(data.count) elements")
    print("  Min: \(data.min()!)")
    print("  Max: \(data.max()!)")
    print("  Mean: \(data.reduce(0, +) / Float(data.count))")
    print("  Sample: \(data.prefix(10))")
}
```

---

## SwiftPixelUtils API Reference

### PixelExtractor

```swift
// Main preprocessing entry point
static func getModelInput(
    source: ImageSource,
    framework: MLFramework,
    width: Int,
    height: Int,
    resizeStrategy: ResizeStrategy = .contain(padding: .black)
) async throws -> ModelInput

// Get raw pixel data with options
static func getPixelData(
    source: ImageSource,
    options: PixelDataOptions
) async throws -> PixelDataResult

// Batch processing
static func getBatchPixelData(
    sources: [ImageSource],
    options: PixelDataOptions,
    maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount
) async throws -> BatchPixelDataResult

// CVPixelBuffer for CoreML
static func getCVPixelBuffer(
    source: ImageSource,
    targetSize: CGSize
) async throws -> CVPixelBuffer
```

### ImageSource

```swift
enum ImageSource {
    case uiImage(UIImage)
    case cgImage(CGImage)
    case ciImage(CIImage)
    case cvPixelBuffer(CVPixelBuffer)
    case url(URL)
    case data(Data)
    
    #if os(macOS)
    case nsImage(NSImage)
    #endif
}
```

### MLFramework

```swift
enum MLFramework {
    case tfliteFloat       // RGB, [0,1], NHWC, Float32
    case tfliteQuantized   // RGB, [0,255], NHWC, UInt8
    case coreML            // RGB, [0,1], NCHW, Float32
    case pytorchMobile     // RGB, ImageNet, NCHW, Float32
    case executorch        // RGB, ImageNet, NCHW, Float32
    case onnx              // RGB, [0,1], NCHW, Float32
    
    case custom(
        colorFormat: ColorFormat,
        normalization: Normalization,
        dataLayout: DataLayout,
        dataType: DataType
    )
}
```

### PixelDataOptions

```swift
struct PixelDataOptions {
    var colorFormat: ColorFormat = .rgb
    var resize: ResizeOptions? = nil
    var roi: ROI? = nil
    var normalization: Normalization = .scale
    var dataLayout: DataLayout = .hwc
    var outputFormat: OutputFormat = .float32Array
    var normalizeOrientation: Bool = false  // Fix UIImage EXIF rotation issues
}
```

#### Output Formats

```swift
enum OutputFormat {
    case array
    case float32Array    // [Float] - standard ML inference
    case float16Array    // [UInt16] - Float16 as bit patterns for Core ML/Metal
    case int32Array      // [Int32] - integer inference
    case uint8Array      // [UInt8] - quantized models
}
```

#### Orientation Handling

When loading from UIImage, EXIF orientation metadata can cause silent rotations:

```swift
// Enable orientation normalization to fix EXIF rotation issues
let options = PixelDataOptions(
    colorFormat: .rgb,
    normalizeOrientation: true  // Redraws image with .up orientation
)
```

### PixelDataResult

```swift
struct PixelDataResult {
    let data: [Float]              // Normalized pixel data (always populated)
    let uint8Data: [UInt8]?        // Raw 0-255 values (when outputFormat is .uint8Array)
    let int32Data: [Int32]?        // Int32 values (when outputFormat is .int32Array)
    let float16Data: [UInt16]?     // Float16 as bit patterns (when outputFormat is .float16Array)
    let width: Int
    let height: Int
    let channels: Int
    let colorFormat: ColorFormat
    let dataLayout: DataLayout
    let shape: [Int]
    let processingTimeMs: Double
    let letterboxInfo: LetterboxInfo?  // Transform metadata (when using .letterbox resize)
}
```

#### Letterbox Transform Info

When using letterbox resize, `letterboxInfo` is populated with transform metadata:

```swift
struct LetterboxInfo {
    let scale: Float           // Scale factor applied to image
    let offset: CGPoint        // Padding offset (x, y)
    let originalSize: CGSize   // Original image dimensions
    let letterboxedSize: CGSize // Final letterboxed dimensions
}

// Use to reverse-transform detection coordinates
if let info = result.letterboxInfo {
    let originalX = (modelOutputX - Float(info.offset.x)) / info.scale
    let originalY = (modelOutputY - Float(info.offset.y)) / info.scale
}
```

---

## Mathematical Foundations

### Linear Algebra of Image Transformations

**Scaling matrix:**
$$\begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Translation matrix:**
$$\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

**Rotation matrix:**
$$\begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### Bilinear Interpolation Formula

For point $(x, y)$ between pixels:
$$f(x, y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy$$

### Color Space Conversion Matrices

**RGB to YUV (BT.601):**
$$\begin{bmatrix} Y \\ U \\ V \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ -0.147 & -0.289 & 0.436 \\ 0.615 & -0.515 & -0.100 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$

