//
//  ImageAugmentor.swift
//  SwiftPixelUtils
//
//  Image augmentation utilities for ML training pipelines
//

import Foundation
import CoreGraphics
import CoreImage

#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

/// Image augmentation utilities for ML training and data augmentation pipelines.
///
/// ## Overview
///
/// Data augmentation is crucial for training robust ML models. This module provides
/// common augmentation techniques that can be applied during training to increase
/// dataset diversity and reduce overfitting.
///
/// ## Augmentation Techniques
///
/// | Technique | Purpose | Use Case |
/// |-----------|---------|----------|
/// | **Rotation** | Orientation invariance | Objects at angles |
/// | **Flip** | Mirror invariance | Symmetric objects |
/// | **Brightness** | Lighting invariance | Indoor/outdoor |
/// | **Contrast** | Dynamic range adaptation | Shadows/highlights |
/// | **Saturation** | Color intensity variation | Different cameras |
/// | **Color Jitter** | Combined color augmentation | General robustness |
/// | **Cutout** | Occlusion robustness | Partial visibility |
/// | **Blur** | Focus variation | Motion/defocus |
///
/// ## Usage
///
/// ```swift
/// // Apply multiple augmentations
/// let augmented = try await ImageAugmentor.applyAugmentations(
///     to: .cgImage(image),
///     options: AugmentationOptions(
///         rotation: 15,
///         horizontalFlip: true,
///         brightness: 0.2,
///         contrast: 0.1
///     )
/// )
///
/// // Color jitter with reproducible seed
/// let jittered = try await ImageAugmentor.colorJitter(
///     source: .cgImage(image),
///     options: ColorJitterOptions(
///         brightness: 0.3,
///         contrast: 0.3,
///         saturation: 0.3,
///         hue: 0.1,
///         seed: 42
///     )
/// )
/// ```
///
/// ## Mathematical Background
///
/// ### Brightness Adjustment
/// ```
/// output = input + brightness × 255
/// ```
/// Positive values brighten, negative darken.
///
/// ### Contrast Adjustment
/// ```
/// output = (input - 128) × (1 + contrast) + 128
/// ```
/// Expands or compresses around middle gray.
///
/// ### Saturation Adjustment
/// ```
/// gray = 0.299R + 0.587G + 0.114B
/// output = gray + (input - gray) × (1 + saturation)
/// ```
/// Interpolates between grayscale and original.
///
/// ### Hue Rotation
/// Rotates colors around the HSV cylinder:
/// ```
/// H_new = (H + hue × 360) mod 360
/// ```
public enum ImageAugmentor {
    
    // MARK: - Core Image Context
    
    /// Shared CIContext for efficient filter operations
    private static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() {
            return CIContext(mtlDevice: device)
        }
        return CIContext()
    }()
    
    // MARK: - Combined Augmentations
    
    /// Applies multiple augmentations to an image.
    ///
    /// Augmentations are applied in a specific order to ensure consistent results:
    /// 1. Geometric transforms (rotation, flip)
    /// 2. Color adjustments (brightness, contrast, saturation)
    /// 3. Blur (if specified)
    ///
    /// - Parameters:
    ///   - source: Image source to augment
    ///   - options: Augmentation configuration
    /// - Returns: Augmented platform image
    /// - Throws: ``PixelUtilsError`` if augmentation fails
    public static func applyAugmentations(
        to source: ImageSource,
        options: AugmentationOptions
    ) throws -> PlatformImage {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load the source image
        var ciImage = try loadCIImage(from: source)
        
        // Apply rotation
        if let rotation = options.rotation, rotation != 0 {
            ciImage = applyRotation(to: ciImage, degrees: rotation)
        }
        
        // Apply flips
        if options.horizontalFlip {
            ciImage = applyHorizontalFlip(to: ciImage)
        }
        if options.verticalFlip {
            ciImage = applyVerticalFlip(to: ciImage)
        }
        
        // Apply brightness
        if let brightness = options.brightness, brightness != 0 {
            ciImage = applyBrightness(to: ciImage, value: brightness)
        }
        
        // Apply contrast
        if let contrast = options.contrast, contrast != 0 {
            ciImage = applyContrast(to: ciImage, value: contrast)
        }
        
        // Apply saturation
        if let saturation = options.saturation, saturation != 0 {
            ciImage = applySaturation(to: ciImage, value: saturation)
        }
        
        // Apply blur
        if let blurOptions = options.blur {
            ciImage = applyBlur(to: ciImage, options: blurOptions)
        }
        
        // Convert back to platform image
        let _ = CFAbsoluteTimeGetCurrent() - startTime
        return try renderToPlatformImage(ciImage)
    }
    
    // MARK: - Color Jitter
    
    /// Applies color jitter augmentation with granular control.
    ///
    /// Color jitter randomly adjusts brightness, contrast, saturation, and hue
    /// within specified ranges. This is one of the most effective augmentations
    /// for improving model robustness to lighting and color variations.
    ///
    /// ## Random Sampling
    ///
    /// Each parameter defines a maximum deviation. The actual adjustment is
    /// sampled uniformly from [-value, +value]:
    ///
    /// ```
    /// actual_brightness = random(-brightness, +brightness)
    /// actual_contrast = random(-contrast, +contrast)
    /// actual_saturation = random(-saturation, +saturation)
    /// actual_hue = random(-hue, +hue)
    /// ```
    ///
    /// ## Reproducibility
    ///
    /// Use the `seed` parameter for reproducible augmentations during debugging
    /// or when you need deterministic training runs.
    ///
    /// - Parameters:
    ///   - source: Image source to augment
    ///   - options: Color jitter configuration with ranges and optional seed
    /// - Returns: ``ColorJitterResult`` with augmented image and applied values
    /// - Throws: ``PixelUtilsError`` if augmentation fails
    public static func colorJitter(
        source: ImageSource,
        options: ColorJitterOptions
    ) throws -> ColorJitterResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Initialize random number generator
        let seed = options.seed ?? UInt64.random(in: 0...UInt64.max)
        var rng = SeededRandomNumberGenerator(seed: seed)
        
        // Sample random values within ranges
        let appliedBrightness = sampleUniform(range: options.brightness, using: &rng)
        let appliedContrast = sampleUniform(range: options.contrast, using: &rng)
        let appliedSaturation = sampleUniform(range: options.saturation, using: &rng)
        let appliedHue = sampleUniform(range: options.hue, using: &rng)
        
        // Load and process image
        var ciImage = try loadCIImage(from: source)
        
        // Apply adjustments in order: brightness, contrast, saturation, hue
        if appliedBrightness != 0 {
            ciImage = applyBrightness(to: ciImage, value: appliedBrightness)
        }
        if appliedContrast != 0 {
            ciImage = applyContrast(to: ciImage, value: appliedContrast)
        }
        if appliedSaturation != 0 {
            ciImage = applySaturation(to: ciImage, value: appliedSaturation)
        }
        if appliedHue != 0 {
            ciImage = applyHueRotation(to: ciImage, value: appliedHue)
        }
        
        let platformImage = try renderToPlatformImage(ciImage)
        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return ColorJitterResult(
            image: platformImage,
            appliedBrightness: appliedBrightness,
            appliedContrast: appliedContrast,
            appliedSaturation: appliedSaturation,
            appliedHue: appliedHue,
            seed: seed,
            processingTimeMs: processingTimeMs
        )
    }
    
    // MARK: - Cutout (Random Erasing)
    
    /// Applies cutout (random erasing) augmentation.
    ///
    /// Cutout randomly masks rectangular regions of the image, forcing the model
    /// to learn from partial information. This improves robustness to occlusion
    /// and encourages the model to use more distributed features.
    ///
    /// ## Algorithm
    ///
    /// For each cutout:
    /// 1. Sample area ratio from [minSize, maxSize] × image area
    /// 2. Sample aspect ratio from [minAspect, maxAspect]
    /// 3. Compute width and height from area and aspect ratio
    /// 4. Sample random position within image bounds
    /// 5. Fill the region according to fillMode
    ///
    /// ## Fill Modes
    ///
    /// | Mode | Description | Use Case |
    /// |------|-------------|----------|
    /// | **constant** | Solid color (default: black) | Standard cutout |
    /// | **noise** | Random noise | More variation |
    /// | **random** | Random solid color | Color invariance |
    ///
    /// ## Research Background
    ///
    /// Cutout was introduced in "Improved Regularization of Convolutional Neural
    /// Networks with Cutout" (DeVries & Taylor, 2017). It's particularly effective
    /// for image classification and can improve accuracy by 1-2% on CIFAR/ImageNet.
    ///
    /// - Parameters:
    ///   - source: Image source to augment
    ///   - options: Cutout configuration
    /// - Returns: Augmented platform image with cutout regions
    /// - Throws: ``PixelUtilsError`` if augmentation fails
    public static func cutout(
        source: ImageSource,
        options: CutoutOptions
    ) throws -> PlatformImage {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Check probability
        let seed = options.seed ?? UInt64.random(in: 0...UInt64.max)
        var rng = SeededRandomNumberGenerator(seed: seed)
        
        if Float.random(in: 0...1, using: &rng) > options.probability {
            // Don't apply cutout, return original
            return try loadPlatformImage(from: source)
        }
        
        // Load image as CGImage for pixel manipulation
        let cgImage = try loadCGImage(from: source)
        let width = cgImage.width
        let height = cgImage.height
        let imageArea = Float(width * height)
        
        // Create mutable context
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
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Apply cutouts
        for _ in 0..<options.numCutouts {
            // Sample area and aspect ratio
            let areaRatio = Float.random(in: options.minSize...options.maxSize, using: &rng)
            let aspectRatio = Float.random(in: options.minAspect...options.maxAspect, using: &rng)
            
            let cutoutArea = imageArea * areaRatio
            let cutoutWidth = Int(sqrt(cutoutArea * aspectRatio))
            let cutoutHeight = Int(sqrt(cutoutArea / aspectRatio))
            
            // Sample position
            let x = Int.random(in: 0..<max(1, width - cutoutWidth), using: &rng)
            let y = Int.random(in: 0..<max(1, height - cutoutHeight), using: &rng)
            
            // Determine fill color
            let fillColor: CGColor
            switch options.fillMode {
            case .constant:
                let r = CGFloat(options.fillValue[0]) / 255.0
                let g = CGFloat(options.fillValue.count > 1 ? options.fillValue[1] : options.fillValue[0]) / 255.0
                let b = CGFloat(options.fillValue.count > 2 ? options.fillValue[2] : options.fillValue[0]) / 255.0
                fillColor = CGColor(red: r, green: g, blue: b, alpha: 1.0)
            case .random:
                fillColor = CGColor(
                    red: CGFloat.random(in: 0...1, using: &rng),
                    green: CGFloat.random(in: 0...1, using: &rng),
                    blue: CGFloat.random(in: 0...1, using: &rng),
                    alpha: 1.0
                )
            case .noise:
                // For noise, we'll fill with a base color and add noise below
                fillColor = CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
            }
            
            // Fill the cutout region
            context.setFillColor(fillColor)
            context.fill(CGRect(x: x, y: y, width: cutoutWidth, height: cutoutHeight))
            
            // Add noise if needed
            if options.fillMode == .noise {
                addNoiseToRegion(context: context, x: x, y: y, width: cutoutWidth, height: cutoutHeight, rng: &rng)
            }
        }
        
        // Create output image
        guard let outputCGImage = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create output image")
        }
        
        let _ = CFAbsoluteTimeGetCurrent() - startTime
        
        #if canImport(UIKit)
        return UIImage(cgImage: outputCGImage)
        #else
        return NSImage(cgImage: outputCGImage, size: NSSize(width: width, height: height))
        #endif
    }
    
    // MARK: - Individual Transforms
    
    /// Applies rotation to an image.
    ///
    /// ## Rotation Mathematics
    ///
    /// Rotation is performed around the image center using an affine transform:
    /// ```
    /// [x']   [cos θ  -sin θ] [x - cx]   [cx]
    /// [y'] = [sin θ   cos θ] [y - cy] + [cy]
    /// ```
    /// where (cx, cy) is the image center and θ is the rotation angle in radians.
    ///
    /// - Parameters:
    ///   - source: Image source to rotate
    ///   - degrees: Rotation angle in degrees (positive = counter-clockwise)
    /// - Returns: Rotated platform image
    /// - Throws: ``PixelUtilsError`` if rotation fails
    public static func rotate(
        source: ImageSource,
        degrees: Float
    ) throws -> PlatformImage {
        var ciImage = try loadCIImage(from: source)
        ciImage = applyRotation(to: ciImage, degrees: degrees)
        return try renderToPlatformImage(ciImage)
    }
    
    /// Applies horizontal flip to an image.
    ///
    /// Mirrors the image along the vertical axis (left becomes right).
    ///
    /// - Parameter source: Image source to flip
    /// - Returns: Horizontally flipped platform image
    /// - Throws: ``PixelUtilsError`` if flip fails
    public static func flipHorizontal(source: ImageSource) throws -> PlatformImage {
        var ciImage = try loadCIImage(from: source)
        ciImage = applyHorizontalFlip(to: ciImage)
        return try renderToPlatformImage(ciImage)
    }
    
    /// Applies vertical flip to an image.
    ///
    /// Mirrors the image along the horizontal axis (top becomes bottom).
    ///
    /// - Parameter source: Image source to flip
    /// - Returns: Vertically flipped platform image
    /// - Throws: ``PixelUtilsError`` if flip fails
    public static func flipVertical(source: ImageSource) throws -> PlatformImage {
        var ciImage = try loadCIImage(from: source)
        ciImage = applyVerticalFlip(to: ciImage)
        return try renderToPlatformImage(ciImage)
    }
    
    /// Adjusts image brightness.
    ///
    /// - Parameters:
    ///   - source: Image source to adjust
    ///   - value: Brightness adjustment (-1 to 1, 0 = no change)
    /// - Returns: Brightness-adjusted platform image
    /// - Throws: ``PixelUtilsError`` if adjustment fails
    public static func adjustBrightness(
        source: ImageSource,
        value: Float
    ) throws -> PlatformImage {
        var ciImage = try loadCIImage(from: source)
        ciImage = applyBrightness(to: ciImage, value: value)
        return try renderToPlatformImage(ciImage)
    }
    
    /// Adjusts image contrast.
    ///
    /// - Parameters:
    ///   - source: Image source to adjust
    ///   - value: Contrast adjustment (-1 to 1, 0 = no change)
    /// - Returns: Contrast-adjusted platform image
    /// - Throws: ``PixelUtilsError`` if adjustment fails
    public static func adjustContrast(
        source: ImageSource,
        value: Float
    ) throws -> PlatformImage {
        var ciImage = try loadCIImage(from: source)
        ciImage = applyContrast(to: ciImage, value: value)
        return try renderToPlatformImage(ciImage)
    }
    
    /// Adjusts image saturation.
    ///
    /// - Parameters:
    ///   - source: Image source to adjust
    ///   - value: Saturation adjustment (-1 to 1, 0 = no change)
    /// - Returns: Saturation-adjusted platform image
    /// - Throws: ``PixelUtilsError`` if adjustment fails
    public static func adjustSaturation(
        source: ImageSource,
        value: Float
    ) throws -> PlatformImage {
        var ciImage = try loadCIImage(from: source)
        ciImage = applySaturation(to: ciImage, value: value)
        return try renderToPlatformImage(ciImage)
    }
    
    /// Applies Gaussian blur to an image.
    ///
    /// - Parameters:
    ///   - source: Image source to blur
    ///   - radius: Blur radius in pixels
    /// - Returns: Blurred platform image
    /// - Throws: ``PixelUtilsError`` if blur fails
    public static func blur(
        source: ImageSource,
        radius: Float
    ) throws -> PlatformImage {
        var ciImage = try loadCIImage(from: source)
        ciImage = applyBlur(to: ciImage, options: BlurOptions(type: .gaussian, radius: radius))
        return try renderToPlatformImage(ciImage)
    }
    
    // MARK: - Private Helpers - Image Loading
    
    private static func loadCIImage(from source: ImageSource) throws -> CIImage {
        switch source {
        case .cgImage(let cgImage):
            return CIImage(cgImage: cgImage)
            
        case .data(let data):
            guard let ciImage = CIImage(data: data) else {
                throw PixelUtilsError.loadFailed("Failed to create CIImage from data")
            }
            return ciImage
            
        case .file(let url):
            // Check if URL is remote and throw a friendly error
            if URLUtilities.isRemoteURL(url) {
                throw PixelUtilsError.invalidSource(
                    URLUtilities.remoteURLErrorMessage(example: "let result = try ImageAugmentor.rotate(source: .data(data), degrees: 45)")
                )
            }
            guard let ciImage = CIImage(contentsOf: url) else {
                throw PixelUtilsError.loadFailed("Failed to load image from file: \(url)")
            }
            return ciImage
            
        case .base64(let base64String):
            guard let data = Data(base64Encoded: base64String),
                  let ciImage = CIImage(data: data) else {
                throw PixelUtilsError.loadFailed("Failed to decode base64 image")
            }
            return ciImage
            
        #if canImport(UIKit)
        case .uiImage(let uiImage):
            guard let cgImage = uiImage.cgImage else {
                throw PixelUtilsError.invalidSource("UIImage has no CGImage")
            }
            return CIImage(cgImage: cgImage)
        #endif
            
        #if canImport(AppKit)
        case .nsImage(let nsImage):
            guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                throw PixelUtilsError.invalidSource("NSImage has no CGImage")
            }
            return CIImage(cgImage: cgImage)
        #endif
        }
    }
    
    private static func loadCGImage(from source: ImageSource) throws -> CGImage {
        switch source {
        case .cgImage(let cgImage):
            return cgImage
            
        case .data(let data):
            guard let provider = CGDataProvider(data: data as CFData),
                  let cgImage = CGImage(
                    pngDataProviderSource: provider,
                    decode: nil,
                    shouldInterpolate: true,
                    intent: .defaultIntent
                  ) ?? CGImage(
                    jpegDataProviderSource: provider,
                    decode: nil,
                    shouldInterpolate: true,
                    intent: .defaultIntent
                  ) else {
                // Fallback: try via CIImage
                let ciImage = try loadCIImage(from: source)
                guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                    throw PixelUtilsError.loadFailed("Failed to create CGImage from data")
                }
                return cgImage
            }
            return cgImage
            
        case .file(let url):
            // Check if URL is remote and throw a friendly error
            if URLUtilities.isRemoteURL(url) {
                throw PixelUtilsError.invalidSource(
                    URLUtilities.remoteURLErrorMessage(example: "let result = try ImageAugmentor.rotate(source: .data(data), degrees: 45)")
                )
            }
            guard let provider = CGDataProvider(url: url as CFURL),
                  let cgImage = CGImage(
                    pngDataProviderSource: provider,
                    decode: nil,
                    shouldInterpolate: true,
                    intent: .defaultIntent
                  ) ?? CGImage(
                    jpegDataProviderSource: provider,
                    decode: nil,
                    shouldInterpolate: true,
                    intent: .defaultIntent
                  ) else {
                let ciImage = try loadCIImage(from: source)
                guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                    throw PixelUtilsError.loadFailed("Failed to load image from file")
                }
                return cgImage
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
    
    private static func loadPlatformImage(from source: ImageSource) throws -> PlatformImage {
        let cgImage = try loadCGImage(from: source)
        #if canImport(UIKit)
        return UIImage(cgImage: cgImage)
        #else
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        #endif
    }
    
    // MARK: - Private Helpers - Transforms
    
    private static func applyRotation(to image: CIImage, degrees: Float) -> CIImage {
        let radians = CGFloat(degrees) * .pi / 180.0
        let transform = CGAffineTransform(rotationAngle: radians)
        return image.transformed(by: transform)
    }
    
    private static func applyHorizontalFlip(to image: CIImage) -> CIImage {
        let transform = CGAffineTransform(scaleX: -1, y: 1)
            .translatedBy(x: -image.extent.width, y: 0)
        return image.transformed(by: transform)
    }
    
    private static func applyVerticalFlip(to image: CIImage) -> CIImage {
        let transform = CGAffineTransform(scaleX: 1, y: -1)
            .translatedBy(x: 0, y: -image.extent.height)
        return image.transformed(by: transform)
    }
    
    private static func applyBrightness(to image: CIImage, value: Float) -> CIImage {
        // CIColorControls brightness range is typically -1 to 1
        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(NSNumber(value: value), forKey: kCIInputBrightnessKey)
        return filter.outputImage ?? image
    }
    
    private static func applyContrast(to image: CIImage, value: Float) -> CIImage {
        // CIColorControls contrast: 0 = no contrast, 1 = normal, 2 = double
        let contrastValue = 1.0 + value  // Map -1..1 to 0..2
        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(NSNumber(value: contrastValue), forKey: kCIInputContrastKey)
        return filter.outputImage ?? image
    }
    
    private static func applySaturation(to image: CIImage, value: Float) -> CIImage {
        // CIColorControls saturation: 0 = grayscale, 1 = normal, 2 = double
        let saturationValue = 1.0 + value  // Map -1..1 to 0..2
        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(NSNumber(value: saturationValue), forKey: kCIInputSaturationKey)
        return filter.outputImage ?? image
    }
    
    private static func applyHueRotation(to image: CIImage, value: Float) -> CIImage {
        // CIHueAdjust uses radians, value is -1 to 1 representing full rotation
        let radians = value * Float.pi * 2
        let filter = CIFilter(name: "CIHueAdjust")!
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(NSNumber(value: radians), forKey: kCIInputAngleKey)
        return filter.outputImage ?? image
    }
    
    private static func applyBlur(to image: CIImage, options: BlurOptions) -> CIImage {
        let filterName = options.type == .gaussian ? "CIGaussianBlur" : "CIBoxBlur"
        let filter = CIFilter(name: filterName)!
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(NSNumber(value: options.radius), forKey: kCIInputRadiusKey)
        return filter.outputImage ?? image
    }
    
    private static func renderToPlatformImage(_ ciImage: CIImage) throws -> PlatformImage {
        let extent = ciImage.extent
        guard let cgImage = ciContext.createCGImage(ciImage, from: extent) else {
            throw PixelUtilsError.processingFailed("Failed to render CIImage to CGImage")
        }
        
        #if canImport(UIKit)
        return UIImage(cgImage: cgImage)
        #else
        return NSImage(cgImage: cgImage, size: NSSize(width: extent.width, height: extent.height))
        #endif
    }
    
    // MARK: - Private Helpers - Random
    
    private static func sampleUniform(range: Float?, using rng: inout SeededRandomNumberGenerator) -> Float {
        guard let range = range, range > 0 else { return 0 }
        return Float.random(in: -range...range, using: &rng)
    }
    
    private static func addNoiseToRegion(
        context: CGContext,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        rng: inout SeededRandomNumberGenerator
    ) {
        guard let data = context.data else { return }
        let bytesPerRow = context.bytesPerRow
        let buffer = data.assumingMemoryBound(to: UInt8.self)
        
        for row in y..<min(y + height, context.height) {
            for col in x..<min(x + width, context.width) {
                let offset = row * bytesPerRow + col * 4
                buffer[offset] = UInt8.random(in: 0...255, using: &rng)     // R
                buffer[offset + 1] = UInt8.random(in: 0...255, using: &rng) // G
                buffer[offset + 2] = UInt8.random(in: 0...255, using: &rng) // B
                // Alpha stays at 255
            }
        }
    }
}

// MARK: - Seeded Random Number Generator

/// A seeded random number generator for reproducible augmentations.
///
/// Uses a linear congruential generator (LCG) for fast, reproducible randomness.
/// The parameters are chosen for good statistical properties:
/// - Multiplier: 6364136223846793005 (from PCG)
/// - Increment: 1442695040888963407 (from PCG)
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        // LCG with parameters from PCG
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
