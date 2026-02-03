//
//  Letterbox.swift
//  SwiftPixelUtils
//
//  YOLO-style letterbox padding with reverse coordinate transform
//

import Foundation
import CoreGraphics
import CoreImage

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Letterbox Options

/// Options for letterbox padding.
///
/// Letterboxing preserves aspect ratio while fitting an image to a target size
/// by adding padding around the edges.
///
/// ## Example
///
/// ```swift
/// let options = LetterboxOptions(
///     targetWidth: 640,
///     targetHeight: 640,
///     fillColor: (114, 114, 114),  // YOLO gray
///     scaleUp: true
/// )
/// ```
public struct LetterboxOptions {
    /// Target width for the letterboxed image.
    public let targetWidth: Int
    
    /// Target height for the letterboxed image.
    public let targetHeight: Int
    
    /// Fill color for padding (RGB values 0-255).
    /// Default is YOLO gray [114, 114, 114].
    public var fillColor: (r: UInt8, g: UInt8, b: UInt8) = (114, 114, 114)
    
    /// Whether to scale up images smaller than target.
    /// Default is true.
    public var scaleUp: Bool = true
    
    /// Whether to center the image or align to top-left.
    /// Default is true (centered).
    public var center: Bool = true
    
    public init(
        targetWidth: Int,
        targetHeight: Int,
        fillColor: (r: UInt8, g: UInt8, b: UInt8) = (114, 114, 114),
        scaleUp: Bool = true,
        center: Bool = true
    ) {
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.fillColor = fillColor
        self.scaleUp = scaleUp
        self.center = center
    }
}

// MARK: - Letterbox Extended Result

/// Extended result of letterbox operation with additional transform information.
public struct LetterboxExtendedResult {
    /// Letterboxed image as base64 PNG.
    public let imageBase64: String
    
    /// The letterboxed CGImage.
    public let cgImage: CGImage
    
    /// Width of the letterboxed image.
    public let width: Int
    
    /// Height of the letterboxed image.
    public let height: Int
    
    /// Original image dimensions.
    public let originalWidth: Int
    public let originalHeight: Int
    
    /// Scale factor applied to the image.
    public let scale: Double
    
    /// X offset (padding on left).
    public let offsetX: Double
    
    /// Y offset (padding on top).
    public let offsetY: Double
    
    /// Fill color used for padding.
    public let fillColor: (r: UInt8, g: UInt8, b: UInt8)
    
    /// Processing time in milliseconds.
    public let processingTimeMs: Double
}

// MARK: - Letterbox

/// YOLO-style letterbox padding operations.
///
/// Letterboxing resizes an image to fit within a target size while maintaining
/// the original aspect ratio, filling any remaining space with a solid color
/// (typically gray for YOLO models).
///
/// ## Overview
///
/// Letterboxing is essential for:
/// - **Object Detection**: YOLO, SSD, Faster R-CNN require consistent input sizes
/// - **Aspect Ratio Preservation**: Prevents object distortion
/// - **Coordinate Mapping**: Provides transform info to map detections back
///
/// ## Example
///
/// ```swift
/// // Apply letterbox for YOLO inference
/// let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
/// let result = try Letterbox.applySync(to: .cgImage(image), options: options)
///
/// // After inference, reverse-transform detection boxes
/// let originalBoxes = Letterbox.reverseTransformBoxes(
///     boxes: detectionBoxes,
///     scale: result.scale,
///     offsetX: result.offsetX,
///     offsetY: result.offsetY
/// )
/// ```
public enum Letterbox {
    
    // MARK: - Apply Letterbox
    
    /// Apply letterbox padding to an image (async).
    ///
    /// - Parameters:
    ///   - source: Image source to letterbox
    ///   - options: Letterbox configuration
    /// - Returns: Letterbox result with transform information
    /// - Throws: `PixelUtilsError` if processing fails
    public static func apply(
        to source: ImageSource,
        options: LetterboxOptions
    ) throws -> LetterboxExtendedResult {
        try applySync(to: source, options: options)
    }
    
    /// Apply letterbox padding to an image (synchronous).
    ///
    /// - Parameters:
    ///   - source: Image source to letterbox
    ///   - options: Letterbox configuration
    /// - Returns: Letterbox result with transform information
    /// - Throws: `PixelUtilsError` if processing fails
    public static func applySync(
        to source: ImageSource,
        options: LetterboxOptions
    ) throws -> LetterboxExtendedResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load source image
        let cgImage = try loadCGImage(from: source)
        let originalWidth = cgImage.width
        let originalHeight = cgImage.height
        
        // Calculate scale to fit while preserving aspect ratio
        let scaleX = Double(options.targetWidth) / Double(originalWidth)
        let scaleY = Double(options.targetHeight) / Double(originalHeight)
        var scale = min(scaleX, scaleY)
        
        // Don't scale up if scaleUp is false
        if !options.scaleUp && scale > 1.0 {
            scale = 1.0
        }
        
        // Calculate scaled dimensions
        let scaledWidth = Int(Double(originalWidth) * scale)
        let scaledHeight = Int(Double(originalHeight) * scale)
        
        // Calculate padding
        let padX: Double
        let padY: Double
        
        if options.center {
            padX = Double(options.targetWidth - scaledWidth) / 2.0
            padY = Double(options.targetHeight - scaledHeight) / 2.0
        } else {
            padX = 0
            padY = 0
        }
        
        // Create letterboxed image
        let letterboxedImage = try createLetterboxedImage(
            source: cgImage,
            targetWidth: options.targetWidth,
            targetHeight: options.targetHeight,
            scaledWidth: scaledWidth,
            scaledHeight: scaledHeight,
            padX: padX,
            padY: padY,
            fillColor: options.fillColor
        )
        
        // Encode to base64
        let base64 = try encodeToBase64PNG(letterboxedImage)
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return LetterboxExtendedResult(
            imageBase64: base64,
            cgImage: letterboxedImage,
            width: options.targetWidth,
            height: options.targetHeight,
            originalWidth: originalWidth,
            originalHeight: originalHeight,
            scale: scale,
            offsetX: padX,
            offsetY: padY,
            fillColor: options.fillColor,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Reverse Transform
    
    /// Reverse transform detection boxes from letterboxed space to original image space.
    ///
    /// - Parameters:
    ///   - boxes: Array of bounding boxes in [x1, y1, x2, y2] format
    ///   - scale: Scale factor from letterbox operation
    ///   - offsetX: X offset (padding) from letterbox operation
    ///   - offsetY: Y offset (padding) from letterbox operation
    /// - Returns: Boxes transformed to original image coordinates
    public static func reverseTransformBoxes(
        boxes: [[Double]],
        scale: Double,
        offsetX: Double,
        offsetY: Double
    ) -> [[Double]] {
        return boxes.map { box in
            guard box.count >= 4 else { return box }
            
            // Remove padding offset and scale back
            let x1 = (box[0] - offsetX) / scale
            let y1 = (box[1] - offsetY) / scale
            let x2 = (box[2] - offsetX) / scale
            let y2 = (box[3] - offsetY) / scale
            
            return [x1, y1, x2, y2]
        }
    }
    
    /// Reverse transform a single box from letterboxed space to original image space.
    ///
    /// - Parameters:
    ///   - box: Bounding box in [x1, y1, x2, y2] format
    ///   - scale: Scale factor from letterbox operation
    ///   - offsetX: X offset (padding) from letterbox operation
    ///   - offsetY: Y offset (padding) from letterbox operation
    /// - Returns: Box transformed to original image coordinates
    public static func reverseTransformBox(
        box: [Double],
        scale: Double,
        offsetX: Double,
        offsetY: Double
    ) -> [Double] {
        reverseTransformBoxes(boxes: [box], scale: scale, offsetX: offsetX, offsetY: offsetY)[0]
    }
    
    /// Reverse transform detections from letterboxed space to original image space.
    ///
    /// - Parameters:
    ///   - detections: Array of Detection objects
    ///   - scale: Scale factor from letterbox operation
    ///   - offsetX: X offset (padding) from letterbox operation
    ///   - offsetY: Y offset (padding) from letterbox operation
    ///   - format: Format of the detection boxes
    /// - Returns: Detections with transformed coordinates
    public static func reverseTransformDetections(
        detections: [Detection],
        scale: Double,
        offsetX: Double,
        offsetY: Double,
        format: BoxFormat = .xyxy
    ) -> [Detection] {
        return detections.map { detection in
            // Convert to xyxy if needed
            var xyxyBox: [Double]
            switch format {
            case .xyxy:
                xyxyBox = detection.box.map { Double($0) }
            case .xywh:
                // xywh to xyxy: [x, y, w, h] -> [x, y, x+w, y+h]
                let box = detection.box
                xyxyBox = [Double(box[0]), Double(box[1]), Double(box[0] + box[2]), Double(box[1] + box[3])]
            case .cxcywh:
                // cxcywh to xyxy: [cx, cy, w, h] -> [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
                let box = detection.box
                let cx = Double(box[0])
                let cy = Double(box[1])
                let w = Double(box[2])
                let h = Double(box[3])
                xyxyBox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            }
            
            // Transform coordinates
            let transformed = reverseTransformBox(
                box: xyxyBox,
                scale: scale,
                offsetX: offsetX,
                offsetY: offsetY
            )
            
            // Convert back to original format
            var finalBox: [Float]
            switch format {
            case .xyxy:
                finalBox = transformed.map { Float($0) }
            case .xywh:
                // xyxy to xywh: [x1, y1, x2, y2] -> [x1, y1, x2-x1, y2-y1]
                finalBox = [Float(transformed[0]), Float(transformed[1]), Float(transformed[2] - transformed[0]), Float(transformed[3] - transformed[1])]
            case .cxcywh:
                // xyxy to cxcywh: [x1, y1, x2, y2] -> [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
                let w = transformed[2] - transformed[0]
                let h = transformed[3] - transformed[1]
                let cx = (transformed[0] + transformed[2]) / 2
                let cy = (transformed[1] + transformed[3]) / 2
                finalBox = [Float(cx), Float(cy), Float(w), Float(h)]
            }
            
            return Detection(
                box: finalBox,
                score: detection.score,
                classIndex: detection.classIndex,
                label: detection.label
            )
        }
    }
    
    // MARK: - Combined Operations
    
    /// Apply letterbox and extract pixel data in one operation.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - letterboxOptions: Letterbox configuration
    ///   - pixelOptions: Pixel extraction options
    /// - Returns: Tuple of pixel data result and letterbox transform info
    /// - Throws: `PixelUtilsError` if processing fails
    public static func applyAndExtract(
        from source: ImageSource,
        letterboxOptions: LetterboxOptions,
        pixelOptions: PixelDataOptions = PixelDataOptions()
    ) throws -> (pixels: PixelDataResult, letterbox: LetterboxExtendedResult) {
        // First apply letterbox
        let letterboxed = try applySync(to: source, options: letterboxOptions)
        
        // Then extract pixels
        var adjustedOptions = pixelOptions
        adjustedOptions.resize = nil // Already resized by letterbox
        
        let pixels = try PixelExtractor.getPixelData(
            source: .cgImage(letterboxed.cgImage),
            options: adjustedOptions
        )
        
        return (pixels: pixels, letterbox: letterboxed)
    }
    
    // MARK: - Private Helpers
    
    private static func loadCGImage(from source: ImageSource) throws -> CGImage {
        switch source {
        case .cgImage(let cgImage):
            return cgImage
            
        case .data(let data):
            guard let provider = CGDataProvider(data: data as CFData),
                  let image = CGImage(jpegDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) ??
                              CGImage(pngDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
                throw PixelUtilsError.loadFailed("Cannot decode image data")
            }
            return image
            
        case .base64(let base64String):
            guard let data = Data(base64Encoded: base64String) else {
                throw PixelUtilsError.loadFailed("Invalid base64 string")
            }
            guard let provider = CGDataProvider(data: data as CFData),
                  let image = CGImage(jpegDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) ??
                              CGImage(pngDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
                throw PixelUtilsError.loadFailed("Cannot decode base64 image")
            }
            return image
            
        case .file(let url):
            // Check if this is a remote URL
            if URLUtilities.isRemoteURL(url) {
                throw PixelUtilsError.loadFailed(
                    URLUtilities.remoteURLErrorMessage(example: "let result = try Letterbox.apply(to: .data(data), options: options)")
                )
            }
            guard let data = try? Data(contentsOf: url) else {
                throw PixelUtilsError.loadFailed("Cannot load file at \(url)")
            }
            guard let provider = CGDataProvider(data: data as CFData),
                  let image = CGImage(jpegDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) ??
                              CGImage(pngDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
                throw PixelUtilsError.loadFailed("Cannot decode image from \(url)")
            }
            return image
            
        #if canImport(UIKit)
        case .uiImage(let uiImage):
            guard let cgImage = uiImage.cgImage else {
                throw PixelUtilsError.loadFailed("Cannot get CGImage from UIImage")
            }
            return cgImage
        #endif
            
        #if canImport(AppKit)
        case .nsImage(let nsImage):
            guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                throw PixelUtilsError.loadFailed("Cannot get CGImage from NSImage")
            }
            return cgImage
        #endif
        }
    }
    
    private static func createLetterboxedImage(
        source: CGImage,
        targetWidth: Int,
        targetHeight: Int,
        scaledWidth: Int,
        scaledHeight: Int,
        padX: Double,
        padY: Double,
        fillColor: (r: UInt8, g: UInt8, b: UInt8)
    ) throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        // Fill with background color
        context.setFillColor(CGColor(
            red: CGFloat(fillColor.r) / 255.0,
            green: CGFloat(fillColor.g) / 255.0,
            blue: CGFloat(fillColor.b) / 255.0,
            alpha: 1.0
        ))
        context.fill(CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        
        // Draw scaled image centered
        // Note: CoreGraphics has origin at bottom-left, so we need to flip Y
        let drawRect = CGRect(
            x: padX,
            y: Double(targetHeight) - padY - Double(scaledHeight),
            width: Double(scaledWidth),
            height: Double(scaledHeight)
        )
        context.draw(source, in: drawRect)
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create letterboxed image")
        }
        
        return result
    }
    
    private static func encodeToBase64PNG(_ cgImage: CGImage) throws -> String {
        #if canImport(UIKit)
        let uiImage = UIImage(cgImage: cgImage)
        guard let data = uiImage.pngData() else {
            throw PixelUtilsError.processingFailed("Failed to encode image to PNG")
        }
        return data.base64EncodedString()
        #elseif canImport(AppKit)
        let bitmap = NSBitmapImageRep(cgImage: cgImage)
        guard let data = bitmap.representation(using: .png, properties: [:]) else {
            throw PixelUtilsError.processingFailed("Failed to encode image to PNG")
        }
        return data.base64EncodedString()
        #endif
    }
}
