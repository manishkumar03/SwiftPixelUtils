//
//  ImageAnalyzer.swift
//  SwiftPixelUtils
//
//  Image analysis utilities for quality assessment and statistics
//

import Foundation
import CoreGraphics
import CoreImage
import Accelerate

#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

/// Image analysis utilities for quality assessment, blur detection, and statistics.
///
/// ## Overview
///
/// This module provides tools for analyzing image quality and extracting statistical
/// information. These are useful for:
/// - Filtering low-quality images before ML inference
/// - Computing dataset statistics for normalization
/// - Quality control in image processing pipelines
///
/// ## Features
///
/// | Feature | Purpose | Use Case |
/// |---------|---------|----------|
/// | **Blur Detection** | Detect out-of-focus images | Filter blurry uploads |
/// | **Statistics** | Compute mean, std, min, max | Dataset analysis |
/// | **Histograms** | Per-channel distributions | Exposure analysis |
/// | **Metadata** | Extract image properties | Format validation |
///
/// ## Usage
///
/// ```swift
/// // Check if image is blurry
/// let blurResult = try await ImageAnalyzer.detectBlur(
///     source: .cgImage(image),
///     threshold: 100.0
/// )
/// if blurResult.isBlurry {
///     print("Image is too blurry (score: \(blurResult.score))")
/// }
///
/// // Get image statistics
/// let stats = try await ImageAnalyzer.getStatistics(source: .cgImage(image))
/// print("Mean RGB: \(stats.mean)")
/// print("Std RGB: \(stats.std)")
/// ```
public enum ImageAnalyzer {
    
    // MARK: - Shared Resources
    
    private static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() {
            return CIContext(mtlDevice: device)
        }
        return CIContext()
    }()
    
    // MARK: - Blur Detection
    
    /// Detects blur in an image using the Laplacian variance method.
    ///
    /// ## Algorithm
    ///
    /// The Laplacian operator highlights edges and rapid intensity changes.
    /// Blurry images have fewer sharp edges, resulting in lower Laplacian variance.
    ///
    /// ### Laplacian Kernel
    /// ```
    /// [ 0  1  0]
    /// [ 1 -4  1]  ×  image  →  edge response
    /// [ 0  1  0]
    /// ```
    ///
    /// ### Variance Calculation
    /// ```
    /// variance = Σ(L - mean(L))² / N
    /// ```
    /// where L is the Laplacian response at each pixel.
    ///
    /// ## Threshold Guidelines
    ///
    /// | Threshold | Sensitivity | Use Case |
    /// |-----------|-------------|----------|
    /// | 50-80 | High | Strict quality control |
    /// | 100-150 | Medium | General use (default) |
    /// | 200-300 | Low | Lenient, allows some blur |
    ///
    /// The optimal threshold depends on image content:
    /// - **Detailed scenes**: Higher variance naturally, use higher threshold
    /// - **Simple backgrounds**: Lower variance naturally, use lower threshold
    ///
    /// ## Downsample Size
    ///
    /// Images are downsampled before analysis for performance. A size of 500px
    /// provides a good balance between accuracy and speed. Smaller values are
    /// faster but may miss fine blur; larger values are slower but more accurate.
    ///
    /// - Parameters:
    ///   - source: Image source to analyze
    ///   - threshold: Variance threshold below which image is considered blurry (default: 100)
    ///   - downsampleSize: Max dimension for analysis, larger images are downsampled (default: 500)
    /// - Returns: ``BlurDetectionResult`` with blur status, score, and timing
    /// - Throws: ``PixelUtilsError`` if analysis fails
    public static func detectBlur(
        source: ImageSource,
        threshold: Float = 100.0,
        downsampleSize: Int = 500
    ) throws -> BlurDetectionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load and optionally downsample
        var cgImage = try loadCGImage(from: source)
        
        let maxDim = max(cgImage.width, cgImage.height)
        if maxDim > downsampleSize {
            let scale = Float(downsampleSize) / Float(maxDim)
            let newWidth = Int(Float(cgImage.width) * scale)
            let newHeight = Int(Float(cgImage.height) * scale)
            cgImage = try downsample(cgImage, toWidth: newWidth, height: newHeight)
        }
        
        // Convert to grayscale float array
        let grayscale = try extractGrayscale(from: cgImage)
        
        // Apply Laplacian kernel
        let laplacian = applyLaplacian(to: grayscale, width: cgImage.width, height: cgImage.height)
        
        // Calculate variance of Laplacian
        let variance = calculateVariance(laplacian)
        
        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return BlurDetectionResult(
            isBlurry: variance < threshold,
            score: variance,
            threshold: threshold,
            processingTimeMs: processingTimeMs
        )
    }
    
    // MARK: - Image Statistics
    
    /// Computes comprehensive statistics for an image.
    ///
    /// ## Statistics Computed
    ///
    /// - **Mean**: Average pixel value per channel
    /// - **Std**: Standard deviation per channel
    /// - **Min**: Minimum pixel value per channel
    /// - **Max**: Maximum pixel value per channel
    /// - **Histogram**: 256-bin histogram per channel
    ///
    /// ## Use Cases
    ///
    /// ### Dataset Normalization
    /// Compute mean/std across your training dataset to use for normalization:
    /// ```swift
    /// var allMeans: [[Float]] = []
    /// for image in dataset {
    ///     let stats = try await ImageAnalyzer.getStatistics(source: image)
    ///     allMeans.append(stats.mean)
    /// }
    /// let datasetMean = allMeans.reduce([0,0,0]) { zip($0, $1).map(+) }
    ///     .map { $0 / Float(dataset.count) }
    /// ```
    ///
    /// ### Exposure Analysis
    /// Check if image is under/overexposed:
    /// ```swift
    /// let stats = try await ImageAnalyzer.getStatistics(source: image)
    /// let avgBrightness = stats.mean.reduce(0, +) / 3
    /// if avgBrightness < 0.2 {
    ///     print("Underexposed")
    /// } else if avgBrightness > 0.8 {
    ///     print("Overexposed")
    /// }
    /// ```
    ///
    /// - Parameter source: Image source to analyze
    /// - Returns: ``ImageStatistics`` with per-channel statistics and histograms
    /// - Throws: ``PixelUtilsError`` if analysis fails
    public static func getStatistics(source: ImageSource) throws -> ImageStatistics {
        let cgImage = try loadCGImage(from: source)
        
        // Extract RGBA pixel data
        let width = cgImage.width
        let height = cgImage.height
        let pixelCount = width * height
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let data = context.data else {
            throw PixelUtilsError.processingFailed("Failed to get pixel data")
        }
        
        let buffer = data.assumingMemoryBound(to: UInt8.self)
        
        // Compute statistics for RGB channels
        var sums: [Double] = [0, 0, 0]
        var sumSquares: [Double] = [0, 0, 0]
        var mins: [Float] = [255, 255, 255]
        var maxs: [Float] = [0, 0, 0]
        var histograms: [[Int]] = [Array(repeating: 0, count: 256),
                                   Array(repeating: 0, count: 256),
                                   Array(repeating: 0, count: 256)]
        
        for i in 0..<pixelCount {
            let offset = i * 4
            for c in 0..<3 {
                let value = Float(buffer[offset + c])
                let normalizedValue = value / 255.0
                
                sums[c] += Double(normalizedValue)
                sumSquares[c] += Double(normalizedValue * normalizedValue)
                mins[c] = min(mins[c], normalizedValue)
                maxs[c] = max(maxs[c], normalizedValue)
                histograms[c][Int(value)] += 1
            }
        }
        
        let count = Double(pixelCount)
        let means = sums.map { Float($0 / count) }
        let stds = zip(sums, sumSquares).map { sum, sumSq in
            let mean = sum / count
            let variance = (sumSq / count) - (mean * mean)
            return Float(sqrt(max(0, variance)))
        }
        
        return ImageStatistics(
            mean: means,
            std: stds,
            min: mins,
            max: maxs,
            histogram: histograms
        )
    }
    
    // MARK: - Image Metadata
    
    /// Extracts metadata information from an image.
    ///
    /// Retrieves basic properties without loading full pixel data:
    /// - Dimensions (width, height)
    /// - Channel count
    /// - Color space
    /// - Alpha channel presence
    /// - Aspect ratio
    ///
    /// - Parameter source: Image source to analyze
    /// - Returns: ``ImageMetadata`` with image properties
    /// - Throws: ``PixelUtilsError`` if metadata extraction fails
    public static func getMetadata(source: ImageSource) throws -> ImageMetadata {
        let cgImage = try loadCGImage(from: source)
        
        let width = cgImage.width
        let height = cgImage.height
        let hasAlpha = cgImage.alphaInfo != .none && cgImage.alphaInfo != .noneSkipLast && cgImage.alphaInfo != .noneSkipFirst
        let channels = hasAlpha ? 4 : 3
        
        // Simplify color space name
        var colorSpace = "Unknown"
        if let csName = cgImage.colorSpace?.name as String? {
            let csNameLower = csName.lowercased()
            if csName == "kCGColorSpaceSRGB" || csNameLower.contains("srgb") {
                colorSpace = "sRGB"
            } else if csName == "kCGColorSpaceDisplayP3" || csNameLower.contains("display p3") || csNameLower.contains("displayp3") {
                colorSpace = "Display P3"
            } else if csName == "kCGColorSpaceAdobeRGB1998" || csNameLower.contains("adobe") {
                colorSpace = "Adobe RGB"
            } else if csName == "kCGColorSpaceGenericGrayGamma2_2" || csNameLower.contains("gray") {
                colorSpace = "Grayscale"
            } else if csName == "kCGColorSpaceGenericCMYK" || csNameLower.contains("cmyk") {
                colorSpace = "CMYK"
            } else {
                colorSpace = csName
            }
        }
        
        let aspectRatio = Float(width) / Float(height)
        
        return ImageMetadata(
            width: width,
            height: height,
            channels: channels,
            colorSpace: colorSpace,
            hasAlpha: hasAlpha,
            aspectRatio: aspectRatio
        )
    }
    
    // MARK: - Image Validation
    
    /// Validates an image against specified criteria.
    ///
    /// Checks if an image meets requirements for:
    /// - Minimum/maximum dimensions
    /// - Aspect ratio constraints
    ///
    /// - Parameters:
    ///   - source: Image source to validate
    ///   - options: Validation criteria
    /// - Returns: ``ValidationResult`` indicating validity and any issues
    /// - Throws: ``PixelUtilsError`` if validation fails
    public static func validate(
        source: ImageSource,
        options: ValidationOptions
    ) throws -> ValidationResult {
        let metadata = try getMetadata(source: source)
        var issues: [String] = []
        
        // Check minimum dimensions
        if let minWidth = options.minWidth, metadata.width < minWidth {
            issues.append("Width \(metadata.width) is below minimum \(minWidth)")
        }
        if let minHeight = options.minHeight, metadata.height < minHeight {
            issues.append("Height \(metadata.height) is below minimum \(minHeight)")
        }
        
        // Check maximum dimensions
        if let maxWidth = options.maxWidth, metadata.width > maxWidth {
            issues.append("Width \(metadata.width) exceeds maximum \(maxWidth)")
        }
        if let maxHeight = options.maxHeight, metadata.height > maxHeight {
            issues.append("Height \(metadata.height) exceeds maximum \(maxHeight)")
        }
        
        // Check aspect ratio
        if let requiredRatio = options.requiredAspectRatio {
            let tolerance = options.aspectRatioTolerance ?? 0.1
            let diff = abs(metadata.aspectRatio - requiredRatio)
            if diff > tolerance {
                issues.append("Aspect ratio \(metadata.aspectRatio) differs from required \(requiredRatio) by more than \(tolerance)")
            }
        }
        
        return ValidationResult(
            isValid: issues.isEmpty,
            issues: issues
        )
    }
    
    // MARK: - Private Helpers - Image Loading
    
    private static func loadCGImage(from source: ImageSource) throws -> CGImage {
        switch source {
        case .cgImage(let cgImage):
            return cgImage
            
        case .data(let data):
            guard let ciImage = CIImage(data: data),
                  let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                throw PixelUtilsError.loadFailed("Failed to create CGImage from data")
            }
            return cgImage
            
        case .file(let url):
            // Check if URL is remote and throw a friendly error
            if URLUtilities.isRemoteURL(url) {
                throw PixelUtilsError.invalidSource(
                    URLUtilities.remoteURLErrorMessage(example: "let result = try ImageAnalyzer.getStatistics(from: .data(data))")
                )
            }
            guard let ciImage = CIImage(contentsOf: url),
                  let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                throw PixelUtilsError.loadFailed("Failed to load image from file: \(url)")
            }
            return cgImage
            
        case .base64(let base64String):
            guard let data = Data(base64Encoded: base64String) else {
                throw PixelUtilsError.loadFailed("Invalid base64 string")
            }
            return try loadCGImage(from: .data(data))
            
        #if canImport(UIKit)
        case .uiImage(let uiImage):
            guard let cgImage = uiImage.cgImage else {
                throw PixelUtilsError.invalidSource("UIImage has no CGImage")
            }
            return cgImage
        #endif
            
        #if canImport(AppKit)
        case .nsImage(let nsImage):
            guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                throw PixelUtilsError.invalidSource("NSImage has no CGImage")
            }
            return cgImage
        #endif
        }
    }
    
    // MARK: - Private Helpers - Blur Detection
    
    private static func downsample(_ image: CGImage, toWidth width: Int, height: Int) throws -> CGImage {
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create downsample context")
        }
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create downsampled image")
        }
        
        return result
    }
    
    private static func extractGrayscale(from cgImage: CGImage) throws -> [Float] {
        let width = cgImage.width
        let height = cgImage.height
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let data = context.data else {
            throw PixelUtilsError.processingFailed("Failed to get pixel data")
        }
        
        let buffer = data.assumingMemoryBound(to: UInt8.self)
        let pixelCount = width * height
        
        // Convert to grayscale using BT.601 luma coefficients
        var grayscale = [Float](repeating: 0, count: pixelCount)
        for i in 0..<pixelCount {
            let offset = i * 4
            let r = Float(buffer[offset]) / 255.0
            let g = Float(buffer[offset + 1]) / 255.0
            let b = Float(buffer[offset + 2]) / 255.0
            grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }
        
        return grayscale
    }
    
    /// Applies Laplacian kernel to detect edges.
    ///
    /// The Laplacian is a second-order derivative operator that highlights
    /// regions of rapid intensity change (edges). The kernel used is:
    /// ```
    /// [ 0  1  0]
    /// [ 1 -4  1]
    /// [ 0  1  0]
    /// ```
    private static func applyLaplacian(to grayscale: [Float], width: Int, height: Int) -> [Float] {
        var laplacian = [Float](repeating: 0, count: grayscale.count)
        
        // Laplacian kernel: [0, 1, 0], [1, -4, 1], [0, 1, 0]
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let idx = y * width + x
                
                let center = grayscale[idx]
                let top = grayscale[(y - 1) * width + x]
                let bottom = grayscale[(y + 1) * width + x]
                let left = grayscale[y * width + (x - 1)]
                let right = grayscale[y * width + (x + 1)]
                
                laplacian[idx] = top + bottom + left + right - 4 * center
            }
        }
        
        return laplacian
    }
    
    /// Calculates variance of an array.
    ///
    /// Variance measures how spread out the values are:
    /// ```
    /// variance = Σ(x - μ)² / N = E[x²] - E[x]²
    /// ```
    private static func calculateVariance(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        
        var sum: Float = 0
        var sumSquares: Float = 0
        
        for value in values {
            sum += value
            sumSquares += value * value
        }
        
        let count = Float(values.count)
        let mean = sum / count
        let variance = (sumSquares / count) - (mean * mean)
        
        return variance
    }
}
