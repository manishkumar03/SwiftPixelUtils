//
//  Types.swift
//  SwiftPixelUtils
//
//  Core type definitions for image processing and ML preprocessing
//

/// # Types
///
/// This module defines the complete type system for SwiftPixelUtils, providing
/// strongly-typed configurations for every stage of the image processing pipeline.
///
/// ## Architecture Overview
///
/// ```
/// ImageSource → Resize → ROI → ColorFormat → Normalization → DataLayout → Output
/// ```
///
/// Each type is designed to be:
/// - **Codable**: Serialize configurations for persistence/transmission
/// - **Immutable**: Thread-safe by default
/// - **Composable**: Mix and match options freely
///
/// ## Key Type Categories
///
/// | Category | Types | Purpose |
/// |----------|-------|--------|
/// | Input | ``ImageSource`` | Abstract over image sources |
/// | Spatial | ``ResizeOptions``, ``ROI`` | Geometry transformations |
/// | Color | ``ColorFormat`` | Pixel color representation |
/// | ML | ``Normalization``, ``DataLayout`` | Neural network preparation |
/// | Detection | ``Detection``, ``BoxFormat`` | Object detection output |
/// | Quality | ``BlurDetectionResult``, ``ImageStatistics`` | Image analysis |

import Foundation
import CoreGraphics
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

// MARK: - Image Source

/// Image source specification
public enum ImageSource {
    case url(URL)
    case file(URL)
    case data(Data)
    case base64(String)
    case cgImage(CGImage)
    #if canImport(UIKit)
    case uiImage(UIImage)
    #endif
    #if canImport(AppKit)
    case nsImage(NSImage)
    #endif
}

// MARK: - Color Formats

/// Output color format for pixel data extraction.
///
/// ## Color Model Overview
///
/// | Format | Channels | Domain | Use Case |
/// |--------|----------|--------|----------|
/// | **RGB/RGBA** | 3/4 | Display | Standard image format |
/// | **BGR/BGRA** | 3/4 | Display | OpenCV, some ML models |
/// | **Grayscale** | 1 | Luminance | Document processing, edge detection |
/// | **HSV** | 3 | Perceptual | Color segmentation, tracking |
/// | **HSL** | 3 | Perceptual | Web colors, design tools |
/// | **LAB** | 3 | Perceptual | Color difference, correction |
/// | **YUV** | 3 | Broadcast | Video compression, analog TV |
/// | **YCbCr** | 3 | Digital | JPEG, MPEG, digital video |
///
/// ## Selecting a Color Format
///
/// ### For Neural Networks
/// - **RGB**: Most common (PyTorch ImageNet models, ResNet, VGG)
/// - **BGR**: OpenCV-trained models, some older Caffe models
/// - **Grayscale**: Document OCR, handwriting recognition
///
/// ### For Computer Vision
/// - **HSV**: Color-based object tracking (hue is lighting-invariant)
/// - **LAB**: Perceptually uniform color matching, skin detection
///
/// ### For Video/Compression
/// - **YCbCr**: JPEG encoding, video codecs (H.264, HEVC)
/// - **YUV**: Broadcast TV, analog video processing
public enum ColorFormat: String, Codable {
    case rgb
    case rgba
    case bgr
    case bgra
    case grayscale
    case hsv
    case hsl
    case lab
    case yuv
    case ycbcr
    
    /// Number of channels for this color format
    public var channelCount: Int {
        switch self {
        case .grayscale:
            return 1
        case .rgb, .bgr, .hsv, .hsl, .lab, .yuv, .ycbcr:
            return 3
        case .rgba, .bgra:
            return 4
        }
    }
}

// MARK: - Resize Strategy

/// Strategy for resizing images to target dimensions.
///
/// ## Strategy Comparison
///
/// For resizing a 1920×1080 image to 640×640:
///
/// | Strategy | Result | Aspect Ratio | Information Loss |
/// |----------|--------|--------------|------------------|
/// | **Cover** | 640×640 crop | Preserved | Crops edges |
/// | **Contain** | 640×360 + padding | Preserved | Adds padding |
/// | **Stretch** | 640×640 stretched | Distorted | None (but distorted) |
/// | **Letterbox** | 640×360 + gray bars | Preserved | Adds gray padding |
///
/// ## Visual Guide
///
/// ```
/// Original (16:9):     Target (1:1):
///  ┌────────────────┐   ┌────────┐
///  │                │   │        │
///  └────────────────┘   └────────┘
///
/// Cover (crop):         Contain (fit):
///  ┌────────┐           ┌────────┐
///  │ cropped │          │░░░░░░░░│
///  │  image  │          │ image  │
///  └────────┘           │░░░░░░░░│
///                       └────────┘
///
/// Stretch:              Letterbox:
///  ┌────────┐           ┌────────┐
///  │squeezed│           │▓▓▓▓▓▓▓▓│
///  │  image │           │ image  │
///  └────────┘           │▓▓▓▓▓▓▓▓│
///                       └────────┘
/// ```
///
/// ## When to Use Each Strategy
///
/// ### Cover
/// - **Best for**: Face detection, object classification
/// - **Rationale**: Maximizes resolution, object likely centered
/// - **Avoid when**: Objects may be near edges
///
/// ### Contain / Letterbox
/// - **Best for**: Object detection (YOLO), full scene analysis
/// - **Rationale**: Preserves all content, consistent aspect ratio
/// - **YOLO standard**: Gray padding (114, 114, 114) minimizes edge artifacts
///
/// ### Stretch
/// - **Best for**: Models trained on stretched data
/// - **Rationale**: Fills target exactly, simple implementation
/// - **Avoid when**: Aspect ratio distortion affects accuracy
public enum ResizeStrategy: String, Codable {
    case cover      // Fill target, crop excess
    case contain    // Fit within target, padding
    case stretch    // Stretch to fill
    case letterbox  // Fit within target, letterbox padding
}

/// Resize configuration options
public struct ResizeOptions: Codable {
    public let width: Int
    public let height: Int
    public let strategy: ResizeStrategy
    public let padColor: [Float]?
    public let letterboxColor: [Float]?
    
    public init(
        width: Int,
        height: Int,
        strategy: ResizeStrategy = .cover,
        padColor: [Float]? = nil,
        letterboxColor: [Float]? = [114, 114, 114]
    ) {
        self.width = width
        self.height = height
        self.strategy = strategy
        self.padColor = padColor
        self.letterboxColor = letterboxColor
    }
}

// MARK: - Normalization

/// Preset normalization strategies for different ML frameworks and models.
///
/// ## Normalization Purpose
///
/// Neural networks perform best when input values are centered around zero
/// with small magnitude. Raw pixel values [0, 255] cause:
/// - Large gradient magnitudes during backpropagation
/// - Slow convergence due to saturated activations
/// - Numerical instability in batch normalization
///
/// ## Preset Formulas
///
/// ### Scale (Default)
/// ```swift
/// output = pixel / 255.0  // Maps [0, 255] → [0, 1]
/// ```
/// **Use for**: Most custom models, simple preprocessing
///
/// ### ImageNet
/// ```swift
/// output = (pixel/255.0 - mean) / std
///
/// mean = [0.485, 0.456, 0.406]  // R, G, B
/// std  = [0.229, 0.224, 0.225]  // R, G, B
/// ```
/// **Origin**: Statistics from 1.2M ImageNet training images
/// **Use for**: ResNet, VGG, DenseNet, EfficientNet, most PyTorch vision models
///
/// ### TensorFlow
/// ```swift
/// output = (pixel / 127.5) - 1.0  // Maps [0, 255] → [-1, 1]
/// ```
/// **Use for**: MobileNet, Inception, NASNet, TensorFlow Hub models
///
/// ### Raw
/// ```swift
/// output = pixel  // No transformation, [0, 255]
/// ```
/// **Use for**: Models expecting integer inputs, quantized models
///
/// ## Why ImageNet Values?
///
/// The ImageNet mean values [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
/// were computed across ~1.2 million training images:
///
/// ```
/// mean_r = 1/N × Σ(all red pixels) / 255 ≈ 0.485
/// mean_g = 1/N × Σ(all green pixels) / 255 ≈ 0.456
/// mean_b = 1/N × Σ(all blue pixels) / 255 ≈ 0.406
/// ```
///
/// The slight red bias (R > G > B) reflects natural image statistics:
/// - Outdoor scenes: sky (blue) is less common than earth/vegetation
/// - Human subjects: skin tones skew toward red/yellow
///
/// ## Custom Normalization
///
/// For domain-specific models (medical imaging, satellite data), compute your
/// own mean/std from your training dataset:
///
/// ```swift
/// let customNorm = Normalization(
///     preset: .custom,
///     mean: [0.5, 0.5, 0.5],  // Your dataset mean
///     std: [0.2, 0.2, 0.2]   // Your dataset std
/// )
/// ```
public enum NormalizationPreset: String, Codable {
    case scale      // [0, 1] range
    case imagenet   // ImageNet mean/std
    case tensorflow // [-1, 1] range
    case raw        // No normalization (0-255)
    case custom     // Custom mean/std
}

/// Normalization configuration with optional custom mean and standard deviation.
///
/// ## Built-in Presets
///
/// Use the static properties for common configurations:
///
/// ```swift
/// let options = PixelDataOptions(normalization: .imagenet)  // For ResNet, VGG, etc.
/// let options = PixelDataOptions(normalization: .tensorflow) // For MobileNet, Inception
/// let options = PixelDataOptions(normalization: .scale)      // Simple [0,1] scaling
/// ```
///
/// ## Custom Normalization
///
/// For models trained on different datasets:
///
/// ```swift
/// // CLIP uses slightly different statistics
/// let clipNorm = Normalization(
///     preset: .custom,
///     mean: [0.48145466, 0.4578275, 0.40821073],
///     std: [0.26862954, 0.26130258, 0.27577711]
/// )
///
/// // Medical imaging often uses grayscale normalization
/// let medicalNorm = Normalization(
///     preset: .custom,
///     mean: [0.5],  // Single channel
///     std: [0.5]
/// )
/// ```
///
/// ## Mathematical Background
///
/// Z-score normalization (standardization):
/// ```
/// z = (x - μ) / σ
///
/// where:
///   x = input pixel value (0-1 range)
///   μ = mean (per-channel)
///   σ = standard deviation (per-channel)
///   z = normalized output (approximately -3 to +3 for most pixels)
/// ```
///
/// This transformation:
/// 1. **Centers** data around zero (subtracting mean)
/// 2. **Scales** to unit variance (dividing by std)
/// 3. **Preserves** relative differences between pixels
public struct Normalization: Codable, Equatable {
    public let preset: NormalizationPreset
    public let mean: [Float]?
    public let std: [Float]?
    
    public init(
        preset: NormalizationPreset = .scale,
        mean: [Float]? = nil,
        std: [Float]? = nil
    ) {
        self.preset = preset
        self.mean = mean
        self.std = std
    }
    
    public static let scale = Normalization(preset: .scale)
    public static let imagenet = Normalization(
        preset: .imagenet,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225]
    )
    public static let tensorflow = Normalization(preset: .tensorflow)
    public static let raw = Normalization(preset: .raw)
}

// MARK: - Data Layout

/// Memory layout format for tensor data in ML frameworks.
///
/// ## Layout Explanation
///
/// For a 224×224 RGB image, different layouts store pixels differently:
///
/// ### HWC (Height × Width × Channels)
/// ```
/// Shape: [224, 224, 3]
/// Memory: R₀₀, G₀₀, B₀₀, R₀₁, G₀₁, B₀₁, ... (pixels are contiguous)
/// Index:  [row, col, channel]
/// ```
/// **Frameworks**: TensorFlow, CoreML, image libraries
///
/// ### CHW (Channels × Height × Width)
/// ```
/// Shape: [3, 224, 224]
/// Memory: R₀₀, R₀₁, R₀₂, ..., G₀₀, G₀₁, ..., B₀₀, B₀₁, ... (channels are contiguous)
/// Index:  [channel, row, col]
/// ```
/// **Frameworks**: PyTorch, ONNX, Caffe
///
/// ### NHWC (Batch × Height × Width × Channels)
/// ```
/// Shape: [1, 224, 224, 3]  (batch size = 1)
/// Memory: Same as HWC with batch dimension prepended
/// Index:  [batch, row, col, channel]
/// ```
/// **Frameworks**: TensorFlow batch inference
///
/// ### NCHW (Batch × Channels × Height × Width)
/// ```
/// Shape: [1, 3, 224, 224]  (batch size = 1)
/// Memory: Same as CHW with batch dimension prepended
/// Index:  [batch, channel, row, col]
/// ```
/// **Frameworks**: PyTorch batch inference, ONNX Runtime
///
/// ## Why CHW for Deep Learning?
///
/// Most deep learning frameworks prefer CHW because:
///
/// 1. **Convolution efficiency**: Filters process entire spatial regions per channel,
///    making contiguous channel data cache-friendly
///
/// 2. **SIMD vectorization**: Channel-wise operations (BatchNorm, activation) benefit
///    from contiguous memory access patterns
///
/// 3. **GPU memory coalescing**: GPU threads accessing adjacent memory addresses
///    achieve better bandwidth utilization
///
/// ## Framework Requirements
///
/// | Framework | Default Layout | Common Input |
/// |-----------|----------------|---------------|
/// | PyTorch | NCHW | `torch.Tensor` |
/// | TensorFlow | NHWC | `tf.Tensor` |
/// | CoreML | NHWC/NCHW | `MLMultiArray` |
/// | ONNX Runtime | NCHW | `OrtValue` |
/// | OpenCV | HWC | `cv::Mat` |
/// | ExecuTorch | NCHW | PyTorch-exported models |
/// | TFLite | NHWC | `TfLiteTensor` |
public enum DataLayout: String, Codable {
    case hwc   // Height × Width × Channels
    case chw   // Channels × Height × Width
    case nhwc  // Batch × Height × Width × Channels
    case nchw  // Batch × Channels × Height × Width
}

// MARK: - Output Format

/// Output format for pixel data
public enum OutputFormat: String, Codable {
    case array
    case float32Array
    case int32Array
    case uint8Array
}

// MARK: - ML Framework Target

/// Target ML framework for automatic configuration.
///
/// Use this to get data in the exact format required by your ML framework,
/// without manually configuring normalization, layout, or output format.
///
/// ## Usage
///
/// ```swift
/// // Get data ready for TensorFlow Lite quantized model
/// let inputData = try await PixelExtractor.getModelInput(
///     source: .uiImage(image),
///     framework: .tfliteQuantized,
///     width: 224,
///     height: 224
/// )
/// // inputData is Data containing UInt8 values in NHWC layout
///
/// // Get data ready for PyTorch model
/// let inputData = try await PixelExtractor.getModelInput(
///     source: .uiImage(image),
///     framework: .pytorch,
///     width: 224,
///     height: 224
/// )
/// // inputData is Data containing Float32 values in NCHW layout with ImageNet normalization
/// ```
///
/// ## Framework Configurations
///
/// | Framework | Layout | Normalization | Output Type |
/// |-----------|--------|---------------|-------------|
/// | `.pytorch` | NCHW | ImageNet | Float32 |
/// | `.pytorchRaw` | NCHW | [0,1] scale | Float32 |
/// | `.tensorflow` | NHWC | [-1,1] | Float32 |
/// | `.tensorflowImageNet` | NHWC | ImageNet | Float32 |
/// | `.tfliteQuantized` | NHWC | raw [0,255] | UInt8 |
/// | `.tfliteFloat` | NHWC | [0,1] scale | Float32 |
/// | `.coreML` | NHWC | [0,1] scale | Float32 |
/// | `.coreMLImageNet` | NHWC | ImageNet | Float32 |
/// | `.onnx` | NCHW | ImageNet | Float32 |
/// | `.execuTorch` | NCHW | ImageNet | Float32 |
/// | `.execuTorchQuantized` | NCHW | raw [0,255] | Int8 |
/// | `.openCV` | HWC/BGR | [0,255] | UInt8 |
public enum MLFramework: String, Codable {
    /// PyTorch models (NCHW, ImageNet normalization, Float32)
    case pytorch
    /// PyTorch models with simple [0,1] scaling (NCHW, Float32)
    case pytorchRaw
    /// TensorFlow models (NHWC, [-1,1] normalization, Float32)
    case tensorflow
    /// TensorFlow models with ImageNet normalization (NHWC, Float32)
    case tensorflowImageNet
    /// TensorFlow Lite quantized models (NHWC, raw [0,255], UInt8)
    case tfliteQuantized
    /// TensorFlow Lite float models (NHWC, [0,1] scale, Float32)
    case tfliteFloat
    /// CoreML models (NHWC, [0,1] scale, Float32)
    case coreML
    /// CoreML models with ImageNet normalization (NHWC, Float32)
    case coreMLImageNet
    /// ONNX Runtime models (NCHW, ImageNet normalization, Float32)
    case onnx
    /// ExecuTorch models (NCHW, ImageNet normalization, Float32)
    case execuTorch
    /// ExecuTorch quantized models (NCHW, raw, Int8)
    case execuTorchQuantized
    /// OpenCV compatible (HWC, BGR, raw [0,255], UInt8)
    case openCV
    
    /// Get the pixel data options for this framework
    public var options: PixelDataOptions {
        switch self {
        case .pytorch:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .imagenet,
                dataLayout: .nchw,
                outputFormat: .float32Array
            )
        case .pytorchRaw:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .scale,
                dataLayout: .nchw,
                outputFormat: .float32Array
            )
        case .tensorflow:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .tensorflow,
                dataLayout: .nhwc,
                outputFormat: .float32Array
            )
        case .tensorflowImageNet:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .imagenet,
                dataLayout: .nhwc,
                outputFormat: .float32Array
            )
        case .tfliteQuantized:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .raw,
                dataLayout: .nhwc,
                outputFormat: .uint8Array
            )
        case .tfliteFloat:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .scale,
                dataLayout: .nhwc,
                outputFormat: .float32Array
            )
        case .coreML:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .scale,
                dataLayout: .nhwc,
                outputFormat: .float32Array
            )
        case .coreMLImageNet:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .imagenet,
                dataLayout: .nhwc,
                outputFormat: .float32Array
            )
        case .onnx:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .imagenet,
                dataLayout: .nchw,
                outputFormat: .float32Array
            )
        case .execuTorch:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .imagenet,
                dataLayout: .nchw,
                outputFormat: .float32Array
            )
        case .execuTorchQuantized:
            return PixelDataOptions(
                colorFormat: .rgb,
                normalization: .raw,
                dataLayout: .nchw,
                outputFormat: .int32Array
            )
        case .openCV:
            return PixelDataOptions(
                colorFormat: .bgr,
                normalization: .raw,
                dataLayout: .hwc,
                outputFormat: .uint8Array
            )
        }
    }
}

/// Result from getModelInput containing raw bytes ready for ML inference
public struct ModelInputResult {
    /// Raw bytes ready to be passed to the ML model
    public let data: Data
    /// Shape of the tensor [batch, ...] - format depends on framework
    public let shape: [Int]
    /// Width of the processed image
    public let width: Int
    /// Height of the processed image
    public let height: Int
    /// Number of channels
    public let channels: Int
    /// Data type description
    public let dataType: String
    /// Processing time in milliseconds
    public let processingTimeMs: Double
    
    public init(
        data: Data,
        shape: [Int],
        width: Int,
        height: Int,
        channels: Int,
        dataType: String,
        processingTimeMs: Double
    ) {
        self.data = data
        self.shape = shape
        self.width = width
        self.height = height
        self.channels = channels
        self.dataType = dataType
        self.processingTimeMs = processingTimeMs
    }
}

// MARK: - ROI (Region of Interest)

/// Region of interest for cropping
public struct ROI: Codable {
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int
    
    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

// MARK: - Pixel Data Options

/// Options for pixel data extraction
public struct PixelDataOptions {
    public var colorFormat: ColorFormat
    public var resize: ResizeOptions?
    public var roi: ROI?
    public var normalization: Normalization
    public var dataLayout: DataLayout
    public var outputFormat: OutputFormat
    
    public init(
        colorFormat: ColorFormat = .rgb,
        resize: ResizeOptions? = nil,
        roi: ROI? = nil,
        normalization: Normalization = .scale,
        dataLayout: DataLayout = .hwc,
        outputFormat: OutputFormat = .float32Array
    ) {
        self.colorFormat = colorFormat
        self.resize = resize
        self.roi = roi
        self.normalization = normalization
        self.dataLayout = dataLayout
        self.outputFormat = outputFormat
    }
}

// MARK: - Pixel Data Result

/// Result from pixel data extraction
public struct PixelDataResult {
    /// Float pixel data (always populated, normalized based on options)
    public let data: [Float]
    /// UInt8 pixel data (0-255 raw values, populated when outputFormat is .uint8Array or .raw normalization)
    public let uint8Data: [UInt8]?
    /// Int32 pixel data (populated when outputFormat is .int32Array)
    public let int32Data: [Int32]?
    public let width: Int
    public let height: Int
    public let channels: Int
    public let colorFormat: ColorFormat
    public let dataLayout: DataLayout
    public let shape: [Int]
    public let processingTimeMs: Double
    
    public init(
        data: [Float],
        uint8Data: [UInt8]? = nil,
        int32Data: [Int32]? = nil,
        width: Int,
        height: Int,
        channels: Int,
        colorFormat: ColorFormat,
        dataLayout: DataLayout,
        shape: [Int],
        processingTimeMs: Double
    ) {
        self.data = data
        self.uint8Data = uint8Data
        self.int32Data = int32Data
        self.width = width
        self.height = height
        self.channels = channels
        self.colorFormat = colorFormat
        self.dataLayout = dataLayout
        self.shape = shape
        self.processingTimeMs = processingTimeMs
    }
}

// MARK: - Image Statistics

/// Statistical information about an image
public struct ImageStatistics {
    public let mean: [Float]
    public let std: [Float]
    public let min: [Float]
    public let max: [Float]
    public let histogram: [[Int]]
    
    public init(
        mean: [Float],
        std: [Float],
        min: [Float],
        max: [Float],
        histogram: [[Int]]
    ) {
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.histogram = histogram
    }
}

// MARK: - Image Metadata

/// Metadata information about an image
public struct ImageMetadata {
    public let width: Int
    public let height: Int
    public let channels: Int
    public let colorSpace: String
    public let hasAlpha: Bool
    public let aspectRatio: Float
    
    public init(
        width: Int,
        height: Int,
        channels: Int,
        colorSpace: String,
        hasAlpha: Bool,
        aspectRatio: Float
    ) {
        self.width = width
        self.height = height
        self.channels = channels
        self.colorSpace = colorSpace
        self.hasAlpha = hasAlpha
        self.aspectRatio = aspectRatio
    }
}

// MARK: - Validation Options

/// Options for image validation
public struct ValidationOptions {
    public let minWidth: Int?
    public let minHeight: Int?
    public let maxWidth: Int?
    public let maxHeight: Int?
    public let requiredAspectRatio: Float?
    public let aspectRatioTolerance: Float?
    
    public init(
        minWidth: Int? = nil,
        minHeight: Int? = nil,
        maxWidth: Int? = nil,
        maxHeight: Int? = nil,
        requiredAspectRatio: Float? = nil,
        aspectRatioTolerance: Float? = nil
    ) {
        self.minWidth = minWidth
        self.minHeight = minHeight
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.requiredAspectRatio = requiredAspectRatio
        self.aspectRatioTolerance = aspectRatioTolerance
    }
}

/// Validation result
public struct ValidationResult {
    public let isValid: Bool
    public let issues: [String]
    
    public init(isValid: Bool, issues: [String]) {
        self.isValid = isValid
        self.issues = issues
    }
}

// MARK: - Blur Detection

/// Result from blur detection
public struct BlurDetectionResult {
    public let isBlurry: Bool
    public let score: Float
    public let threshold: Float
    public let processingTimeMs: Double
    
    public init(
        isBlurry: Bool,
        score: Float,
        threshold: Float,
        processingTimeMs: Double
    ) {
        self.isBlurry = isBlurry
        self.score = score
        self.threshold = threshold
        self.processingTimeMs = processingTimeMs
    }
}

// MARK: - Augmentation Options

/// Options for image augmentation
public struct AugmentationOptions {
    public let rotation: Float?
    public let horizontalFlip: Bool
    public let verticalFlip: Bool
    public let brightness: Float?
    public let contrast: Float?
    public let saturation: Float?
    public let blur: BlurOptions?
    
    public init(
        rotation: Float? = nil,
        horizontalFlip: Bool = false,
        verticalFlip: Bool = false,
        brightness: Float? = nil,
        contrast: Float? = nil,
        saturation: Float? = nil,
        blur: BlurOptions? = nil
    ) {
        self.rotation = rotation
        self.horizontalFlip = horizontalFlip
        self.verticalFlip = verticalFlip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blur = blur
    }
}

/// Blur options for augmentation
public struct BlurOptions {
    public let type: BlurType
    public let radius: Float
    
    public init(type: BlurType = .gaussian, radius: Float = 2) {
        self.type = type
        self.radius = radius
    }
}

/// Blur type
public enum BlurType: String {
    case gaussian
    case box
}

// MARK: - Color Jitter Options

/// Options for color jitter augmentation
public struct ColorJitterOptions {
    public let brightness: Float?
    public let contrast: Float?
    public let saturation: Float?
    public let hue: Float?
    public let seed: UInt64?
    
    public init(
        brightness: Float? = nil,
        contrast: Float? = nil,
        saturation: Float? = nil,
        hue: Float? = nil,
        seed: UInt64? = nil
    ) {
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.seed = seed
    }
}

/// Result from color jitter
public struct ColorJitterResult {
    public let image: PlatformImage
    public let appliedBrightness: Float
    public let appliedContrast: Float
    public let appliedSaturation: Float
    public let appliedHue: Float
    public let seed: UInt64
    public let processingTimeMs: Double
    
    public init(
        image: PlatformImage,
        appliedBrightness: Float,
        appliedContrast: Float,
        appliedSaturation: Float,
        appliedHue: Float,
        seed: UInt64,
        processingTimeMs: Double
    ) {
        self.image = image
        self.appliedBrightness = appliedBrightness
        self.appliedContrast = appliedContrast
        self.appliedSaturation = appliedSaturation
        self.appliedHue = appliedHue
        self.seed = seed
        self.processingTimeMs = processingTimeMs
    }
}

// MARK: - Cutout Options

/// Options for cutout augmentation
public struct CutoutOptions {
    public let numCutouts: Int
    public let minSize: Float
    public let maxSize: Float
    public let minAspect: Float
    public let maxAspect: Float
    public let fillMode: FillMode
    public let fillValue: [UInt8]
    public let probability: Float
    public let seed: UInt64?
    
    public init(
        numCutouts: Int = 1,
        minSize: Float = 0.02,
        maxSize: Float = 0.33,
        minAspect: Float = 0.3,
        maxAspect: Float = 3.3,
        fillMode: FillMode = .constant,
        fillValue: [UInt8] = [0, 0, 0],
        probability: Float = 1.0,
        seed: UInt64? = nil
    ) {
        self.numCutouts = numCutouts
        self.minSize = minSize
        self.maxSize = maxSize
        self.minAspect = minAspect
        self.maxAspect = maxAspect
        self.fillMode = fillMode
        self.fillValue = fillValue
        self.probability = probability
        self.seed = seed
    }
}

/// Fill mode for cutout
public enum FillMode: String {
    case constant
    case noise
    case random
}

// MARK: - Bounding Box

/// Bounding box coordinate format for object detection.
///
/// ## Format Definitions
///
/// For a box with top-left corner at (10, 20) and dimensions 100×50:
///
/// | Format | Values | Interpretation |
/// |--------|--------|----------------|
/// | **XYXY** | [10, 20, 110, 70] | (x1, y1, x2, y2) corners |
/// | **XYWH** | [10, 20, 100, 50] | (x, y, width, height) |
/// | **CXCYWH** | [60, 45, 100, 50] | (center_x, center_y, width, height) |
///
/// ## Format Usage by Model/Dataset
///
/// | Format | Used By |
/// |--------|--------|
/// | **XYXY** | Pascal VOC, Faster R-CNN outputs |
/// | **XYWH** | COCO annotations, SSD |
/// | **CXCYWH** | YOLO (all versions), anchor-based models |
///
/// ## Why CXCYWH for YOLO?
///
/// Anchor-based detectors like YOLO predict **offsets from anchor centers**:
///
/// ```
/// predicted_cx = anchor_cx + sigmoid(tx)
/// predicted_cy = anchor_cy + sigmoid(ty)
/// predicted_w  = anchor_w × exp(tw)
/// predicted_h  = anchor_h × exp(th)
/// ```
///
/// Center-based representation makes these offset calculations natural and
/// symmetric, unlike corner-based formats where offsets would be asymmetric.
///
/// ## Conversion Guidelines
///
/// - **For IoU calculation**: Convert to XYXY (simplest intersection math)
/// - **For display/annotation**: XYWH is most intuitive
/// - **For model training**: Match the model's expected format
public enum BoxFormat: String {
    case xyxy   // [x1, y1, x2, y2]
    case xywh   // [x, y, width, height]
    case cxcywh // [cx, cy, width, height]
}

/// Detection with bounding box
public struct Detection {
    public let box: [Float]
    public let score: Float
    public let classIndex: Int
    public let label: String?
    
    public init(
        box: [Float],
        score: Float,
        classIndex: Int,
        label: String? = nil
    ) {
        self.box = box
        self.score = score
        self.classIndex = classIndex
        self.label = label
    }
}

// MARK: - Quantization

/// Quantization mode
public enum QuantizationMode: String {
    case perTensor
    case perChannel
}

/// Data type for quantization
public enum QuantizationDType: String {
    case int8
    case uint8
    case int16
}

/// Quantization options
public struct QuantizationOptions {
    public let mode: QuantizationMode
    public let dtype: QuantizationDType
    public let scale: [Float]
    public let zeroPoint: [Int]
    
    public init(
        mode: QuantizationMode,
        dtype: QuantizationDType,
        scale: [Float],
        zeroPoint: [Int]
    ) {
        self.mode = mode
        self.dtype = dtype
        self.scale = scale
        self.zeroPoint = zeroPoint
    }
}

// MARK: - Letterbox

/// Letterbox transform info
public struct LetterboxInfo {
    public let scale: Float
    public let offset: CGPoint
    public let originalSize: CGSize
    public let letterboxedSize: CGSize
    
    public init(
        scale: Float,
        offset: CGPoint,
        originalSize: CGSize,
        letterboxedSize: CGSize
    ) {
        self.scale = scale
        self.offset = offset
        self.originalSize = originalSize
        self.letterboxedSize = letterboxedSize
    }
}

/// Letterbox result
public struct LetterboxResult {
    public let image: PlatformImage
    public let info: LetterboxInfo
    public let processingTimeMs: Double
    
    public init(
        image: PlatformImage,
        info: LetterboxInfo,
        processingTimeMs: Double
    ) {
        self.image = image
        self.info = info
        self.processingTimeMs = processingTimeMs
    }
}

// MARK: - Drawing Options

/// Options for drawing on images
public struct DrawingOptions {
    public let lineWidth: Float
    public let fontSize: Float
    public let drawLabels: Bool
    public let labelBackgroundAlpha: Float
    
    public init(
        lineWidth: Float = 2,
        fontSize: Float = 12,
        drawLabels: Bool = true,
        labelBackgroundAlpha: Float = 0.7
    ) {
        self.lineWidth = lineWidth
        self.fontSize = fontSize
        self.drawLabels = drawLabels
        self.labelBackgroundAlpha = labelBackgroundAlpha
    }
}

/// Box annotation for drawing
public struct BoxAnnotation {
    public let box: [Float]
    public let label: String?
    public let score: Float?
    public let classIndex: Int?
    public let color: [Float]?
    
    public init(
        box: [Float],
        label: String? = nil,
        score: Float? = nil,
        classIndex: Int? = nil,
        color: [Float]? = nil
    ) {
        self.box = box
        self.label = label
        self.score = score
        self.classIndex = classIndex
        self.color = color
    }
}
