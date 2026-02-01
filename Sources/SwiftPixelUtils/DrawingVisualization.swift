//
//  DrawingVisualization.swift
//  SwiftPixelUtils
//
//  Draw bounding boxes, keypoints, masks, and heatmaps for debugging
//

import Foundation
import CoreGraphics
import CoreText

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Drawing Types

/// Options for drawing bounding boxes.
public struct BoxDrawingOptions {
    /// Line width for box borders.
    public var lineWidth: CGFloat = 2.0
    
    /// Font size for labels.
    public var fontSize: CGFloat = 12.0
    
    /// Whether to draw labels.
    public var drawLabels: Bool = true
    
    /// Whether to draw confidence scores.
    public var drawScores: Bool = true
    
    /// Background alpha for label text.
    public var labelBackgroundAlpha: CGFloat = 0.7
    
    /// Default color if not specified per box (RGBA 0-255).
    public var defaultColor: (r: UInt8, g: UInt8, b: UInt8, a: UInt8) = (255, 0, 0, 255)
    
    public init(
        lineWidth: CGFloat = 2.0,
        fontSize: CGFloat = 12.0,
        drawLabels: Bool = true,
        drawScores: Bool = true,
        labelBackgroundAlpha: CGFloat = 0.7,
        defaultColor: (r: UInt8, g: UInt8, b: UInt8, a: UInt8) = (255, 0, 0, 255)
    ) {
        self.lineWidth = lineWidth
        self.fontSize = fontSize
        self.drawLabels = drawLabels
        self.drawScores = drawScores
        self.labelBackgroundAlpha = labelBackgroundAlpha
        self.defaultColor = defaultColor
    }
}

/// A bounding box to draw with optional styling.
public struct DrawableBox {
    /// Box coordinates [x1, y1, x2, y2].
    public let box: [Float]
    
    /// Optional label text.
    public var label: String?
    
    /// Optional confidence score.
    public var score: Float?
    
    /// Optional color override (RGBA 0-255).
    public var color: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)?
    
    public init(
        box: [Float],
        label: String? = nil,
        score: Float? = nil,
        color: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)? = nil
    ) {
        self.box = box
        self.label = label
        self.score = score
        self.color = color
    }
}

/// Options for drawing keypoints.
public struct KeypointDrawingOptions {
    /// Radius of keypoint circles.
    public var pointRadius: CGFloat = 4.0
    
    /// Line width for skeleton connections.
    public var lineWidth: CGFloat = 2.0
    
    /// Whether to draw skeleton connections.
    public var drawSkeleton: Bool = true
    
    /// Skeleton connections as pairs of keypoint indices.
    public var skeleton: [(Int, Int)]?
    
    /// Default color for keypoints (RGBA 0-255).
    public var defaultColor: (r: UInt8, g: UInt8, b: UInt8, a: UInt8) = (0, 255, 0, 255)
    
    /// Minimum confidence threshold for drawing.
    public var confidenceThreshold: Float = 0.5
    
    public init(
        pointRadius: CGFloat = 4.0,
        lineWidth: CGFloat = 2.0,
        drawSkeleton: Bool = true,
        skeleton: [(Int, Int)]? = nil,
        defaultColor: (r: UInt8, g: UInt8, b: UInt8, a: UInt8) = (0, 255, 0, 255),
        confidenceThreshold: Float = 0.5
    ) {
        self.pointRadius = pointRadius
        self.lineWidth = lineWidth
        self.drawSkeleton = drawSkeleton
        self.skeleton = skeleton
        self.defaultColor = defaultColor
        self.confidenceThreshold = confidenceThreshold
    }
    
    /// COCO pose skeleton connections.
    public static let cocoSkeleton: [(Int, Int)] = [
        (0, 1), (0, 2), (1, 3), (2, 4),  // Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  // Arms
        (5, 11), (6, 12), (11, 12),  // Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  // Legs
    ]
}

/// A keypoint to draw.
public struct DrawableKeypoint {
    /// X coordinate.
    public let x: Float
    
    /// Y coordinate.
    public let y: Float
    
    /// Confidence score (0-1).
    public var confidence: Float = 1.0
    
    /// Optional color override.
    public var color: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)?
    
    public init(
        x: Float,
        y: Float,
        confidence: Float = 1.0,
        color: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)? = nil
    ) {
        self.x = x
        self.y = y
        self.confidence = confidence
        self.color = color
    }
}

/// Options for mask overlay.
public struct MaskOverlayOptions {
    /// Alpha for the mask overlay (0-1).
    public var alpha: CGFloat = 0.5
    
    /// Color for the mask (RGB 0-255).
    public var color: (r: UInt8, g: UInt8, b: UInt8) = (0, 128, 255)
    
    /// Width of the mask.
    public let maskWidth: Int
    
    /// Height of the mask.
    public let maskHeight: Int
    
    /// Threshold for binary mask (values above this are colored).
    public var threshold: Float = 0.5
    
    public init(
        alpha: CGFloat = 0.5,
        color: (r: UInt8, g: UInt8, b: UInt8) = (0, 128, 255),
        maskWidth: Int,
        maskHeight: Int,
        threshold: Float = 0.5
    ) {
        self.alpha = alpha
        self.color = color
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight
        self.threshold = threshold
    }
}

/// Options for heatmap overlay.
public struct HeatmapOverlayOptions {
    /// Alpha for the heatmap overlay (0-1).
    public var alpha: CGFloat = 0.6
    
    /// Color scheme for the heatmap.
    public var colorScheme: HeatmapColorScheme = .jet
    
    /// Width of the heatmap.
    public let heatmapWidth: Int
    
    /// Height of the heatmap.
    public let heatmapHeight: Int
    
    public init(
        alpha: CGFloat = 0.6,
        colorScheme: HeatmapColorScheme = .jet,
        heatmapWidth: Int,
        heatmapHeight: Int
    ) {
        self.alpha = alpha
        self.colorScheme = colorScheme
        self.heatmapWidth = heatmapWidth
        self.heatmapHeight = heatmapHeight
    }
}

/// Color schemes for heatmaps.
public enum HeatmapColorScheme {
    case jet
    case hot
    case viridis
    case grayscale
}

/// Result of drawing operation.
public struct DrawingResult {
    /// Image as base64 PNG.
    public let imageBase64: String
    
    /// The CGImage.
    public let cgImage: CGImage
    
    /// Width of the image.
    public let width: Int
    
    /// Height of the image.
    public let height: Int
    
    /// Processing time in milliseconds.
    public let processingTimeMs: Double
}

// MARK: - Drawing Operations

/// Drawing and visualization utilities for ML debugging.
///
/// Provides visualization for:
/// - Bounding boxes with labels and scores
/// - Keypoint poses with skeleton connections
/// - Segmentation mask overlays
/// - Attention/activation heatmaps
///
/// ## Example
///
/// ```swift
/// // Draw detection boxes
/// let boxes = detections.map { det in
///     DrawableBox(box: det.box, label: det.label, score: det.score)
/// }
///
/// let result = try Drawing.drawBoxes(
///     on: .cgImage(image),
///     boxes: boxes,
///     options: BoxDrawingOptions(lineWidth: 2, drawLabels: true)
/// )
///
/// // Use result.imageBase64 for display
/// ```
public enum Drawing {
    
    // MARK: - Draw Boxes
    
    /// Draw bounding boxes on an image.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - boxes: Boxes to draw
    ///   - options: Drawing options
    /// - Returns: Drawing result
    /// - Throws: `PixelUtilsError` if drawing fails
    public static func drawBoxes(
        on source: ImageSource,
        boxes: [DrawableBox],
        options: BoxDrawingOptions = BoxDrawingOptions()
    ) throws -> DrawingResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let cgImage = try loadCGImage(from: source)
        let width = cgImage.width
        let height = cgImage.height
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw boxes
        for box in boxes {
            guard box.box.count >= 4 else { continue }
            
            let color = box.color ?? options.defaultColor
            let cgColor = CGColor(
                red: CGFloat(color.r) / 255.0,
                green: CGFloat(color.g) / 255.0,
                blue: CGFloat(color.b) / 255.0,
                alpha: CGFloat(color.a) / 255.0
            )
            
            // Convert coordinates (CoreGraphics has origin at bottom-left)
            let x1 = CGFloat(box.box[0])
            let y1 = CGFloat(height) - CGFloat(box.box[3])  // Flip Y
            let x2 = CGFloat(box.box[2])
            let y2 = CGFloat(height) - CGFloat(box.box[1])  // Flip Y
            
            let rect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
            
            // Draw box
            context.setStrokeColor(cgColor)
            context.setLineWidth(options.lineWidth)
            context.stroke(rect)
            
            // Draw label if enabled
            if options.drawLabels, let label = box.label {
                var labelText = label
                if options.drawScores, let score = box.score {
                    labelText += String(format: " %.2f", score)
                }
                
                // Draw label background and text using CoreText
                let fontSize = options.fontSize
                let font = CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil)
                
                let attributes: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                ]
                
                let attributedString = NSAttributedString(string: labelText, attributes: attributes)
                let line = CTLineCreateWithAttributedString(attributedString)
                let textBounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
                
                let padding: CGFloat = 4
                let labelX = x1
                let labelY = y1 - textBounds.height - padding * 2  // Position above box
                let bgWidth = textBounds.width + padding * 2
                let bgHeight = textBounds.height + padding * 2
                
                // Draw label background
                let bgColor = CGColor(
                    red: CGFloat(color.r) / 255.0,
                    green: CGFloat(color.g) / 255.0,
                    blue: CGFloat(color.b) / 255.0,
                    alpha: options.labelBackgroundAlpha
                )
                context.setFillColor(bgColor)
                let bgRect = CGRect(x: labelX, y: labelY, width: bgWidth, height: bgHeight)
                context.fill(bgRect)
                
                // Draw text
                context.saveGState()
                context.textMatrix = .identity
                let textX = labelX + padding
                let textY = labelY + padding + textBounds.height - (textBounds.height - fontSize) / 2
                context.textPosition = CGPoint(x: textX, y: textY)
                CTLineDraw(line, context)
                context.restoreGState()
            }
        }
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create result image")
        }
        
        let base64 = try encodeToBase64PNG(result)
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return DrawingResult(
            imageBase64: base64,
            cgImage: result,
            width: width,
            height: height,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Draw Keypoints
    
    /// Draw keypoints on an image.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - keypoints: Keypoints to draw
    ///   - options: Drawing options
    /// - Returns: Drawing result
    /// - Throws: `PixelUtilsError` if drawing fails
    public static func drawKeypoints(
        on source: ImageSource,
        keypoints: [DrawableKeypoint],
        options: KeypointDrawingOptions = KeypointDrawingOptions()
    ) throws -> DrawingResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let cgImage = try loadCGImage(from: source)
        let width = cgImage.width
        let height = cgImage.height
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw skeleton connections first (so points are on top)
        if options.drawSkeleton, let skeleton = options.skeleton {
            let defaultCGColor = CGColor(
                red: CGFloat(options.defaultColor.r) / 255.0,
                green: CGFloat(options.defaultColor.g) / 255.0,
                blue: CGFloat(options.defaultColor.b) / 255.0,
                alpha: CGFloat(options.defaultColor.a) / 255.0
            )
            
            context.setStrokeColor(defaultCGColor)
            context.setLineWidth(options.lineWidth)
            
            for (i, j) in skeleton {
                if i < keypoints.count && j < keypoints.count {
                    let p1 = keypoints[i]
                    let p2 = keypoints[j]
                    
                    if p1.confidence >= options.confidenceThreshold &&
                       p2.confidence >= options.confidenceThreshold {
                        context.move(to: CGPoint(x: CGFloat(p1.x), y: CGFloat(height) - CGFloat(p1.y)))
                        context.addLine(to: CGPoint(x: CGFloat(p2.x), y: CGFloat(height) - CGFloat(p2.y)))
                        context.strokePath()
                    }
                }
            }
        }
        
        // Draw keypoints
        for kp in keypoints {
            if kp.confidence < options.confidenceThreshold { continue }
            
            let color = kp.color ?? options.defaultColor
            let cgColor = CGColor(
                red: CGFloat(color.r) / 255.0,
                green: CGFloat(color.g) / 255.0,
                blue: CGFloat(color.b) / 255.0,
                alpha: CGFloat(color.a) / 255.0
            )
            
            let x = CGFloat(kp.x)
            let y = CGFloat(height) - CGFloat(kp.y)  // Flip Y
            
            context.setFillColor(cgColor)
            let rect = CGRect(
                x: x - options.pointRadius,
                y: y - options.pointRadius,
                width: options.pointRadius * 2,
                height: options.pointRadius * 2
            )
            context.fillEllipse(in: rect)
        }
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create result image")
        }
        
        let base64 = try encodeToBase64PNG(result)
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return DrawingResult(
            imageBase64: base64,
            cgImage: result,
            width: width,
            height: height,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Overlay Mask
    
    /// Overlay a segmentation mask on an image.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - mask: Mask values (0-1 range)
    ///   - options: Overlay options
    /// - Returns: Drawing result
    /// - Throws: `PixelUtilsError` if overlay fails
    public static func overlayMask(
        on source: ImageSource,
        mask: [Float],
        options: MaskOverlayOptions
    ) throws -> DrawingResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let cgImage = try loadCGImage(from: source)
        let width = cgImage.width
        let height = cgImage.height
        
        // Validate mask size
        guard mask.count == options.maskWidth * options.maskHeight else {
            throw PixelUtilsError.invalidOptions(
                "Mask size \(mask.count) doesn't match \(options.maskWidth)x\(options.maskHeight)"
            )
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Create mask overlay
        let scaleX = Float(width) / Float(options.maskWidth)
        let scaleY = Float(height) / Float(options.maskHeight)
        
        context.setFillColor(CGColor(
            red: CGFloat(options.color.r) / 255.0,
            green: CGFloat(options.color.g) / 255.0,
            blue: CGFloat(options.color.b) / 255.0,
            alpha: options.alpha
        ))
        
        for my in 0..<options.maskHeight {
            for mx in 0..<options.maskWidth {
                let maskValue = mask[my * options.maskWidth + mx]
                if maskValue >= options.threshold {
                    let x = CGFloat(Float(mx) * scaleX)
                    let y = CGFloat(Float(height) - Float(my + 1) * scaleY)  // Flip Y
                    let w = CGFloat(scaleX)
                    let h = CGFloat(scaleY)
                    context.fill(CGRect(x: x, y: y, width: w, height: h))
                }
            }
        }
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create result image")
        }
        
        let base64 = try encodeToBase64PNG(result)
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return DrawingResult(
            imageBase64: base64,
            cgImage: result,
            width: width,
            height: height,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Overlay Heatmap
    
    /// Overlay a heatmap on an image.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - heatmap: Heatmap values (0-1 range)
    ///   - options: Overlay options
    /// - Returns: Drawing result
    /// - Throws: `PixelUtilsError` if overlay fails
    public static func overlayHeatmap(
        on source: ImageSource,
        heatmap: [Float],
        options: HeatmapOverlayOptions
    ) throws -> DrawingResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let cgImage = try loadCGImage(from: source)
        let width = cgImage.width
        let height = cgImage.height
        
        // Validate heatmap size
        guard heatmap.count == options.heatmapWidth * options.heatmapHeight else {
            throw PixelUtilsError.invalidOptions(
                "Heatmap size \(heatmap.count) doesn't match \(options.heatmapWidth)x\(options.heatmapHeight)"
            )
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Create heatmap overlay
        let scaleX = Float(width) / Float(options.heatmapWidth)
        let scaleY = Float(height) / Float(options.heatmapHeight)
        
        for hy in 0..<options.heatmapHeight {
            for hx in 0..<options.heatmapWidth {
                let value = max(0, min(1, heatmap[hy * options.heatmapWidth + hx]))
                let color = colorForValue(value, scheme: options.colorScheme)
                
                context.setFillColor(CGColor(
                    red: CGFloat(color.r) / 255.0,
                    green: CGFloat(color.g) / 255.0,
                    blue: CGFloat(color.b) / 255.0,
                    alpha: options.alpha * CGFloat(value)
                ))
                
                let x = CGFloat(Float(hx) * scaleX)
                let y = CGFloat(Float(height) - Float(hy + 1) * scaleY)  // Flip Y
                let w = CGFloat(scaleX)
                let h = CGFloat(scaleY)
                context.fill(CGRect(x: x, y: y, width: w, height: h))
            }
        }
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create result image")
        }
        
        let base64 = try encodeToBase64PNG(result)
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return DrawingResult(
            imageBase64: base64,
            cgImage: result,
            width: width,
            height: height,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Internal Helpers
    
    static func loadCGImage(from source: ImageSource) throws -> CGImage {
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
            
        case .url(let url), .file(let url):
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
    
    static func encodeToBase64PNG(_ cgImage: CGImage) throws -> String {
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
    
    private static func colorForValue(_ value: Float, scheme: HeatmapColorScheme) -> (r: UInt8, g: UInt8, b: UInt8) {
        switch scheme {
        case .jet:
            return jetColor(value)
        case .hot:
            return hotColor(value)
        case .viridis:
            return viridisColor(value)
        case .grayscale:
            let gray = UInt8(value * 255)
            return (gray, gray, gray)
        }
    }
    
    private static func jetColor(_ value: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        // Jet colormap: blue -> cyan -> green -> yellow -> red
        let v = max(0, min(1, value))
        
        var r: Float = 0
        var g: Float = 0
        var b: Float = 0
        
        if v < 0.125 {
            b = 0.5 + v * 4
        } else if v < 0.375 {
            b = 1
            g = (v - 0.125) * 4
        } else if v < 0.625 {
            g = 1
            b = 1 - (v - 0.375) * 4
        } else if v < 0.875 {
            g = 1 - (v - 0.625) * 4
            r = (v - 0.625) * 4
        } else {
            r = 1
        }
        
        return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
    }
    
    private static func hotColor(_ value: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        // Hot colormap: black -> red -> yellow -> white
        let v = max(0, min(1, value))
        
        var r: Float = 0
        var g: Float = 0
        var b: Float = 0
        
        if v < 0.33 {
            r = v / 0.33
        } else if v < 0.66 {
            r = 1
            g = (v - 0.33) / 0.33
        } else {
            r = 1
            g = 1
            b = (v - 0.66) / 0.34
        }
        
        return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
    }
    
    private static func viridisColor(_ value: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        // Simplified viridis approximation
        let v = max(0, min(1, value))
        
        let r = 0.267004 + v * (0.993248 - 0.267004)
        let g = 0.004874 + v * (0.906157 - 0.004874)
        let b = 0.329415 + v * (0.143936 - 0.329415) * (1 - v) + v * 0.143936
        
        return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
    }
}
