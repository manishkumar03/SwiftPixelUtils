//
//  DepthEstimationOutput.swift
//  SwiftPixelUtils
//
//  High-level API for processing depth estimation model outputs
//

import Foundation
import CoreGraphics

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

#if canImport(CoreML)
import CoreML
#endif

// MARK: - Depth Estimation Result

/// Result from depth estimation model processing.
///
/// Contains the depth map and utilities for visualization and analysis.
///
/// ## Depth Values
///
/// - **MiDaS/DPT**: Outputs inverse relative depth (higher values = closer)
/// - **ZoeDepth**: Outputs metric depth in meters (lower values = closer)
///
/// Use ``isMetric`` to determine which interpretation applies.
public struct DepthEstimationResult {
    /// Raw depth values from the model
    /// Shape: [height × width] flattened
    public let depthMap: [Float]
    
    /// Width of the depth map
    public let width: Int
    
    /// Height of the depth map
    public let height: Int
    
    /// Minimum depth value in the map
    public let minDepth: Float
    
    /// Maximum depth value in the map
    public let maxDepth: Float
    
    /// Whether depth values are metric (meters) or relative
    public let isMetric: Bool
    
    /// Whether depth is inverse (higher = closer, like MiDaS)
    public let isInverse: Bool
    
    /// Processing time in milliseconds
    public let processingTimeMs: Double
    
    /// Original image dimensions (before model resize)
    public let originalWidth: Int?
    public let originalHeight: Int?
    
    public init(
        depthMap: [Float],
        width: Int,
        height: Int,
        minDepth: Float,
        maxDepth: Float,
        isMetric: Bool = false,
        isInverse: Bool = true,
        processingTimeMs: Double,
        originalWidth: Int? = nil,
        originalHeight: Int? = nil
    ) {
        self.depthMap = depthMap
        self.width = width
        self.height = height
        self.minDepth = minDepth
        self.maxDepth = maxDepth
        self.isMetric = isMetric
        self.isInverse = isInverse
        self.processingTimeMs = processingTimeMs
        self.originalWidth = originalWidth
        self.originalHeight = originalHeight
    }
    
    // MARK: - Depth Access
    
    /// Get the depth value at a specific position.
    /// - Parameters:
    ///   - x: X coordinate (0 to width-1)
    ///   - y: Y coordinate (0 to height-1)
    /// - Returns: Depth value at the position, or nil if out of bounds
    public func depthAt(x: Int, y: Int) -> Float? {
        guard x >= 0 && x < width && y >= 0 && y < height else { return nil }
        return depthMap[y * width + x]
    }
    
    /// Get depth value at normalized coordinates (0-1 range).
    /// - Parameters:
    ///   - normalizedX: X coordinate (0.0 to 1.0)
    ///   - normalizedY: Y coordinate (0.0 to 1.0)
    /// - Returns: Depth value with bilinear interpolation
    public func depthAtNormalized(normalizedX: Float, normalizedY: Float) -> Float? {
        let x = normalizedX * Float(width - 1)
        let y = normalizedY * Float(height - 1)
        return bilinearInterpolate(x: x, y: y)
    }
    
    /// Bilinear interpolation for sub-pixel depth queries
    private func bilinearInterpolate(x: Float, y: Float) -> Float? {
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        
        guard let d00 = depthAt(x: x0, y: y0),
              let d10 = depthAt(x: x1, y: y0),
              let d01 = depthAt(x: x0, y: y1),
              let d11 = depthAt(x: x1, y: y1) else {
            return nil
        }
        
        let xFrac = x - Float(x0)
        let yFrac = y - Float(y0)
        
        let d0 = d00 * (1 - xFrac) + d10 * xFrac
        let d1 = d01 * (1 - xFrac) + d11 * xFrac
        
        return d0 * (1 - yFrac) + d1 * yFrac
    }
    
    // MARK: - Normalization
    
    /// Returns depth map normalized to 0-1 range.
    /// - Parameter invert: If true, inverts the depth (closer = 1, farther = 0)
    /// - Returns: Normalized depth values
    public func normalized(invert: Bool = false) -> [Float] {
        let range = maxDepth - minDepth
        guard range > 0 else {
            return [Float](repeating: 0.5, count: depthMap.count)
        }
        
        if invert {
            return depthMap.map { 1.0 - ($0 - minDepth) / range }
        } else {
            return depthMap.map { ($0 - minDepth) / range }
        }
    }
    
    /// Returns 2D array of depth values.
    /// - Returns: Array of rows, each containing depth values
    public func as2DArray() -> [[Float]] {
        var result = [[Float]]()
        for y in 0..<height {
            let startIdx = y * width
            let endIdx = startIdx + width
            result.append(Array(depthMap[startIdx..<endIdx]))
        }
        return result
    }
    
    // MARK: - Statistics
    
    /// Statistical summary of the depth map
    public var statistics: DepthStatistics {
        let sorted = depthMap.sorted()
        let mean = depthMap.reduce(0, +) / Float(depthMap.count)
        
        // Variance and std dev
        let variance = depthMap.reduce(0) { $0 + pow($1 - mean, 2) } / Float(depthMap.count)
        let stdDev = sqrt(variance)
        
        // Percentiles
        let p25 = sorted[Int(Float(sorted.count) * 0.25)]
        let median = sorted[sorted.count / 2]
        let p75 = sorted[Int(Float(sorted.count) * 0.75)]
        
        return DepthStatistics(
            min: minDepth,
            max: maxDepth,
            mean: mean,
            stdDev: stdDev,
            median: median,
            percentile25: p25,
            percentile75: p75
        )
    }
    
    // MARK: - Visualization
    
    /// Convert depth map to grayscale image.
    /// - Parameter invert: If true, closer objects appear brighter
    /// - Returns: CGImage in grayscale
    public func toGrayscaleImage(invert: Bool = true) -> CGImage? {
        let normalizedDepth = normalized(invert: invert)
        var grayData = [UInt8](repeating: 0, count: width * height)
        
        for i in 0..<normalizedDepth.count {
            grayData[i] = UInt8(min(255, max(0, normalizedDepth[i] * 255)))
        }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        guard let provider = CGDataProvider(data: Data(grayData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: width,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: 0),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            return nil
        }
        
        return cgImage
    }
    
    /// Convert depth map to colored image using a colormap.
    /// - Parameters:
    ///   - colormap: Color palette for visualization
    ///   - invert: If true, closer objects appear in "hot" colors
    /// - Returns: CGImage with colormap applied
    public func toColoredImage(
        colormap: DepthColormap = .viridis,
        invert: Bool = true
    ) -> CGImage? {
        let normalizedDepth = normalized(invert: invert)
        var rgbaData = [UInt8](repeating: 255, count: width * height * 4)
        
        for i in 0..<normalizedDepth.count {
            let color = colormap.color(forValue: normalizedDepth[i])
            rgbaData[i * 4] = color.r
            rgbaData[i * 4 + 1] = color.g
            rgbaData[i * 4 + 2] = color.b
            rgbaData[i * 4 + 3] = 255
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(data: Data(rgbaData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            return nil
        }
        
        return cgImage
    }
    
    /// Create platform-specific image (UIImage on iOS, NSImage on macOS)
    public func toPlatformImage(
        colormap: DepthColormap = .viridis,
        invert: Bool = true
    ) -> PlatformImage? {
        guard let cgImage = toColoredImage(colormap: colormap, invert: invert) else {
            return nil
        }
        
        #if canImport(UIKit)
        return UIImage(cgImage: cgImage)
        #elseif canImport(AppKit)
        return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
        #endif
    }
    
    // MARK: - Resizing
    
    /// Resize depth map to new dimensions using bilinear interpolation.
    /// - Parameters:
    ///   - newWidth: Target width
    ///   - newHeight: Target height
    /// - Returns: New DepthEstimationResult with resized depth map
    public func resized(to newWidth: Int, to newHeight: Int) -> DepthEstimationResult {
        var resizedDepth = [Float](repeating: 0, count: newWidth * newHeight)
        
        let scaleX = Float(width - 1) / Float(newWidth - 1)
        let scaleY = Float(height - 1) / Float(newHeight - 1)
        
        for newY in 0..<newHeight {
            for newX in 0..<newWidth {
                let srcX = Float(newX) * scaleX
                let srcY = Float(newY) * scaleY
                resizedDepth[newY * newWidth + newX] = bilinearInterpolate(x: srcX, y: srcY) ?? 0
            }
        }
        
        // Recalculate min/max
        let newMin = resizedDepth.min() ?? 0
        let newMax = resizedDepth.max() ?? 1
        
        return DepthEstimationResult(
            depthMap: resizedDepth,
            width: newWidth,
            height: newHeight,
            minDepth: newMin,
            maxDepth: newMax,
            isMetric: isMetric,
            isInverse: isInverse,
            processingTimeMs: processingTimeMs,
            originalWidth: originalWidth ?? width,
            originalHeight: originalHeight ?? height
        )
    }
    
    /// Resize depth map to original image dimensions.
    /// - Returns: Resized result, or self if original dimensions unknown
    public func resizedToOriginal() -> DepthEstimationResult {
        guard let origWidth = originalWidth,
              let origHeight = originalHeight,
              origWidth != width || origHeight != height else {
            return self
        }
        return resized(to: origWidth, to: origHeight)
    }
}

// MARK: - Depth Statistics

/// Statistical summary of depth values
public struct DepthStatistics {
    public let min: Float
    public let max: Float
    public let mean: Float
    public let stdDev: Float
    public let median: Float
    public let percentile25: Float
    public let percentile75: Float
    
    /// Interquartile range (75th - 25th percentile)
    public var iqr: Float { percentile75 - percentile25 }
    
    /// Range (max - min)
    public var range: Float { max - min }
}

// MARK: - Depth Colormaps

/// Color palettes for depth visualization.
///
/// Scientific colormaps designed for perceptual uniformity and
/// colorblind-friendly representation of continuous data.
public struct DepthColormap {
    /// Name of the colormap
    public let name: String
    
    /// Color lookup table (256 RGB values)
    private let lut: [(r: UInt8, g: UInt8, b: UInt8)]
    
    public init(name: String, lut: [(r: UInt8, g: UInt8, b: UInt8)]) {
        self.name = name
        self.lut = lut
    }
    
    /// Get color for a normalized value (0-1).
    public func color(forValue value: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        let clampedValue = min(1.0, max(0.0, value))
        let index = Int(clampedValue * 255)
        return lut[min(index, 255)]
    }
    
    // MARK: - Predefined Colormaps
    
    /// Viridis colormap - perceptually uniform, colorblind-friendly (blue → green → yellow)
    public static let viridis = DepthColormap(
        name: "Viridis",
        lut: generateViridis()
    )
    
    /// Plasma colormap - perceptually uniform (blue → purple → orange → yellow)
    public static let plasma = DepthColormap(
        name: "Plasma",
        lut: generatePlasma()
    )
    
    /// Inferno colormap - perceptually uniform (black → purple → orange → yellow)
    public static let inferno = DepthColormap(
        name: "Inferno",
        lut: generateInferno()
    )
    
    /// Magma colormap - perceptually uniform (black → purple → orange → white)
    public static let magma = DepthColormap(
        name: "Magma",
        lut: generateMagma()
    )
    
    /// Turbo colormap - rainbow-like but improved (blue → cyan → green → yellow → red)
    public static let turbo = DepthColormap(
        name: "Turbo",
        lut: generateTurbo()
    )
    
    /// Grayscale colormap (black → white)
    public static let grayscale = DepthColormap(
        name: "Grayscale",
        lut: (0..<256).map { v in (UInt8(v), UInt8(v), UInt8(v)) }
    )
    
    /// Jet colormap - classic rainbow (blue → cyan → green → yellow → red)
    /// Note: Not perceptually uniform, prefer viridis for accuracy
    public static let jet = DepthColormap(
        name: "Jet",
        lut: generateJet()
    )
    
    // MARK: - Colormap Generation
    
    private static func generateViridis() -> [(r: UInt8, g: UInt8, b: UInt8)] {
        // Viridis colormap data points (sampled)
        let keyColors: [(Float, Float, Float)] = [
            (0.267, 0.004, 0.329),   // 0
            (0.282, 0.140, 0.458),   // 32
            (0.254, 0.265, 0.530),   // 64
            (0.207, 0.372, 0.553),   // 96
            (0.164, 0.471, 0.558),   // 128
            (0.128, 0.567, 0.551),   // 160
            (0.135, 0.659, 0.518),   // 192
            (0.267, 0.749, 0.441),   // 224
            (0.993, 0.906, 0.144)    // 255
        ]
        return interpolateColormap(keyColors: keyColors)
    }
    
    private static func generatePlasma() -> [(r: UInt8, g: UInt8, b: UInt8)] {
        let keyColors: [(Float, Float, Float)] = [
            (0.050, 0.030, 0.528),
            (0.295, 0.012, 0.615),
            (0.494, 0.012, 0.658),
            (0.665, 0.138, 0.618),
            (0.798, 0.280, 0.470),
            (0.899, 0.396, 0.301),
            (0.973, 0.580, 0.254),
            (0.940, 0.975, 0.131),
            (0.940, 0.975, 0.131)
        ]
        return interpolateColormap(keyColors: keyColors)
    }
    
    private static func generateInferno() -> [(r: UInt8, g: UInt8, b: UInt8)] {
        let keyColors: [(Float, Float, Float)] = [
            (0.001, 0.000, 0.014),
            (0.119, 0.047, 0.212),
            (0.316, 0.071, 0.485),
            (0.533, 0.135, 0.531),
            (0.735, 0.216, 0.330),
            (0.891, 0.349, 0.113),
            (0.976, 0.591, 0.124),
            (0.964, 0.843, 0.273),
            (0.988, 0.998, 0.645)
        ]
        return interpolateColormap(keyColors: keyColors)
    }
    
    private static func generateMagma() -> [(r: UInt8, g: UInt8, b: UInt8)] {
        let keyColors: [(Float, Float, Float)] = [
            (0.001, 0.000, 0.014),
            (0.113, 0.065, 0.276),
            (0.316, 0.071, 0.485),
            (0.533, 0.135, 0.531),
            (0.716, 0.215, 0.475),
            (0.868, 0.287, 0.409),
            (0.967, 0.439, 0.360),
            (0.995, 0.624, 0.427),
            (0.987, 0.991, 0.750)
        ]
        return interpolateColormap(keyColors: keyColors)
    }
    
    private static func generateTurbo() -> [(r: UInt8, g: UInt8, b: UInt8)] {
        let keyColors: [(Float, Float, Float)] = [
            (0.190, 0.072, 0.232),
            (0.136, 0.304, 0.773),
            (0.145, 0.536, 0.910),
            (0.220, 0.728, 0.750),
            (0.450, 0.863, 0.460),
            (0.710, 0.920, 0.280),
            (0.920, 0.820, 0.280),
            (0.995, 0.560, 0.220),
            (0.850, 0.180, 0.120)
        ]
        return interpolateColormap(keyColors: keyColors)
    }
    
    private static func generateJet() -> [(r: UInt8, g: UInt8, b: UInt8)] {
        let keyColors: [(Float, Float, Float)] = [
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
            (0.0, 0.5, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, 1.0, 0.5),
            (1.0, 1.0, 0.0),
            (1.0, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 0.0, 0.0)
        ]
        return interpolateColormap(keyColors: keyColors)
    }
    
    private static func interpolateColormap(
        keyColors: [(Float, Float, Float)]
    ) -> [(r: UInt8, g: UInt8, b: UInt8)] {
        var lut = [(r: UInt8, g: UInt8, b: UInt8)]()
        
        let numSegments = keyColors.count - 1
        let pointsPerSegment = 256 / numSegments
        
        for i in 0..<256 {
            let segment = min(i / pointsPerSegment, numSegments - 1)
            let t = Float(i % pointsPerSegment) / Float(pointsPerSegment)
            
            let c0 = keyColors[segment]
            let c1 = keyColors[min(segment + 1, keyColors.count - 1)]
            
            let r = c0.0 + (c1.0 - c0.0) * t
            let g = c0.1 + (c1.1 - c0.1) * t
            let b = c0.2 + (c1.2 - c0.2) * t
            
            lut.append((
                UInt8(min(255, max(0, r * 255))),
                UInt8(min(255, max(0, g * 255))),
                UInt8(min(255, max(0, b * 255)))
            ))
        }
        
        return lut
    }
}

// MARK: - DepthEstimationOutput Main API

/// High-level utilities for processing depth estimation model outputs.
///
/// Supports popular depth estimation models:
/// - **MiDaS** (v2.1, v3.0): Relative inverse depth
/// - **DPT** (Dense Prediction Transformer): High-quality relative depth
/// - **ZoeDepth**: Metric depth in meters
///
/// ## Basic Usage
///
/// ```swift
/// // Process MiDaS output
/// let result = try DepthEstimationOutput.process(
///     output: modelOutput,
///     width: 384,
///     height: 384,
///     modelType: .midas
/// )
///
/// // Visualize with colormap
/// let coloredImage = result.toColoredImage(colormap: .viridis)
///
/// // Get depth at specific point
/// let depth = result.depthAt(x: 100, y: 100)
/// ```
///
/// ## Model Output Formats
///
/// | Model | Output Shape | Value Range | Interpretation |
/// |-------|-------------|-------------|----------------|
/// | MiDaS | [1, H, W] | ~0-10000 | Inverse relative (higher = closer) |
/// | DPT | [1, H, W] | ~0-10000 | Inverse relative (higher = closer) |
/// | ZoeDepth | [1, H, W] | 0-10+ meters | Metric (lower = closer) |
public enum DepthEstimationOutput {
    
    // MARK: - Main Processing
    
    /// Process depth model output into a structured result.
    ///
    /// - Parameters:
    ///   - output: Raw model output as flattened Float array [H × W]
    ///   - width: Width of the depth map
    ///   - height: Height of the depth map
    ///   - modelType: Type of depth model for proper interpretation
    ///   - originalWidth: Original image width (for resize)
    ///   - originalHeight: Original image height (for resize)
    /// - Returns: Processed depth estimation result
    /// - Throws: ``PixelUtilsError`` if processing fails
    public static func process(
        output: [Float],
        width: Int,
        height: Int,
        modelType: DepthModelType = .midas,
        originalWidth: Int? = nil,
        originalHeight: Int? = nil
    ) throws -> DepthEstimationResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let expectedSize = width * height
        guard output.count == expectedSize else {
            throw PixelUtilsError.processingFailed(
                "Output size \(output.count) doesn't match expected \(width)×\(height) = \(expectedSize)"
            )
        }
        
        let minDepth = output.min() ?? 0
        let maxDepth = output.max() ?? 1
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return DepthEstimationResult(
            depthMap: output,
            width: width,
            height: height,
            minDepth: minDepth,
            maxDepth: maxDepth,
            isMetric: modelType.isMetric,
            isInverse: modelType.isInverse,
            processingTimeMs: processingTime,
            originalWidth: originalWidth,
            originalHeight: originalHeight
        )
    }
    
    /// Process depth model output with 2D array input.
    ///
    /// - Parameters:
    ///   - output2D: 2D array of depth values [height][width]
    ///   - modelType: Type of depth model
    ///   - originalWidth: Original image width (for resize)
    ///   - originalHeight: Original image height (for resize)
    /// - Returns: Processed depth estimation result
    public static func process(
        output2D: [[Float]],
        modelType: DepthModelType = .midas,
        originalWidth: Int? = nil,
        originalHeight: Int? = nil
    ) throws -> DepthEstimationResult {
        guard !output2D.isEmpty, !output2D[0].isEmpty else {
            throw PixelUtilsError.processingFailed("Empty depth output")
        }
        
        let height = output2D.count
        let width = output2D[0].count
        let flattened = output2D.flatMap { $0 }
        
        return try process(
            output: flattened,
            width: width,
            height: height,
            modelType: modelType,
            originalWidth: originalWidth,
            originalHeight: originalHeight
        )
    }
    
    #if canImport(CoreML)
    /// Process MLMultiArray output from Core ML depth model.
    ///
    /// Handles common Core ML output shapes:
    /// - [1, H, W] (batched single channel)
    /// - [H, W] (unbatched)
    /// - [1, 1, H, W] (batched with channel dim)
    ///
    /// - Parameters:
    ///   - multiArray: MLMultiArray from model inference
    ///   - modelType: Type of depth model
    ///   - originalWidth: Original image width
    ///   - originalHeight: Original image height
    /// - Returns: Processed depth estimation result
    public static func process(
        multiArray: MLMultiArray,
        modelType: DepthModelType = .midas,
        originalWidth: Int? = nil,
        originalHeight: Int? = nil
    ) throws -> DepthEstimationResult {
        let shape = multiArray.shape.map { $0.intValue }
        
        // Determine width and height from shape
        let (width, height): (Int, Int)
        switch shape.count {
        case 2:
            // [H, W]
            height = shape[0]
            width = shape[1]
        case 3:
            // [1, H, W] or [C, H, W]
            height = shape[1]
            width = shape[2]
        case 4:
            // [1, 1, H, W] or [N, C, H, W]
            height = shape[2]
            width = shape[3]
        default:
            throw PixelUtilsError.processingFailed("Unsupported depth output shape: \(shape)")
        }
        
        // Extract float values
        let count = multiArray.count
        var output = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            output[i] = ptr[i]
        }
        
        // Take only H×W values if batched
        let expectedSize = width * height
        let depthValues: [Float]
        if output.count > expectedSize {
            depthValues = Array(output.prefix(expectedSize))
        } else {
            depthValues = output
        }
        
        return try process(
            output: depthValues,
            width: width,
            height: height,
            modelType: modelType,
            originalWidth: originalWidth,
            originalHeight: originalHeight
        )
    }
    #endif
}

// MARK: - Depth Model Types

/// Supported depth estimation model types.
///
/// Each model type has different output characteristics:
///
/// | Model | Depth Type | Output Values |
/// |-------|-----------|---------------|
/// | Depth Anything | Relative inverse | Higher = closer |
/// | MiDaS | Relative inverse | Higher = closer |
/// | DPT | Relative inverse | Higher = closer |
/// | ZoeDepth | Metric (meters) | Lower = closer |
public enum DepthModelType: String, CaseIterable {
    /// Depth Anything models (inverse relative depth)
    case depthAnything
    
    /// MiDaS v2.1 or v3.0 models (inverse relative depth)
    case midas
    
    /// DPT (Dense Prediction Transformer) models
    case dpt
    
    /// DPT-Hybrid (ResNet50 + ViT backbone)
    case dptHybrid
    
    /// ZoeDepth models (metric depth in meters)
    case zoeDepth
    
    /// ZoeDepth trained on NYU Depth (indoor)
    case zoeDepthNYU
    
    /// ZoeDepth trained on KITTI (outdoor/driving)
    case zoeDepthKITTI
    
    /// Custom model (default to inverse relative)
    case custom
    
    /// Whether this model outputs metric depth (in meters)
    public var isMetric: Bool {
        switch self {
        case .zoeDepth, .zoeDepthNYU, .zoeDepthKITTI:
            return true
        default:
            return false
        }
    }
    
    /// Whether depth values are inverse (higher = closer)
    public var isInverse: Bool {
        switch self {
        case .zoeDepth, .zoeDepthNYU, .zoeDepthKITTI:
            return false // ZoeDepth: lower values = closer
        default:
            return true // MiDaS/DPT/DepthAnything: higher values = closer
        }
    }
    
    /// Recommended input size for this model type
    public var recommendedInputSize: (width: Int, height: Int) {
        switch self {
        case .depthAnything:
            return (518, 518)
        case .midas:
            return (384, 384)
        case .dpt, .dptHybrid:
            return (384, 384)
        case .zoeDepth, .zoeDepthNYU, .zoeDepthKITTI:
            return (512, 384)
        case .custom:
            return (384, 384)
        }
    }
    
    /// Display name for the model type
    public var displayName: String {
        switch self {
        case .depthAnything: return "Depth Anything"
        case .midas: return "MiDaS"
        case .dpt: return "DPT"
        case .dptHybrid: return "DPT-Hybrid"
        case .zoeDepth: return "ZoeDepth"
        case .zoeDepthNYU: return "ZoeDepth-NYU"
        case .zoeDepthKITTI: return "ZoeDepth-KITTI"
        case .custom: return "Custom"
        }
    }
}
