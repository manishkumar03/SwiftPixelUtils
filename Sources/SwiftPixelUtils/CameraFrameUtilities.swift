//
//  CameraFrameUtilities.swift
//  SwiftPixelUtils
//
//  High-performance camera frame processing for vision-camera integration
//

import Foundation
import CoreGraphics

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Camera Frame Types

/// Pixel format for camera frames.
public enum CameraPixelFormat: String {
    case yuv420 = "yuv420"
    case nv12 = "nv12"
    case nv21 = "nv21"
    case bgra = "bgra"
    case rgba = "rgba"
}

/// Camera frame source.
public struct CameraFrameSource {
    /// Frame data as base64-encoded string.
    public let dataBase64: String
    
    /// Width of the frame.
    public let width: Int
    
    /// Height of the frame.
    public let height: Int
    
    /// Pixel format.
    public let pixelFormat: CameraPixelFormat
    
    /// Bytes per row (stride) if different from width * bytesPerPixel.
    public var bytesPerRow: Int?
    
    public init(
        dataBase64: String,
        width: Int,
        height: Int,
        pixelFormat: CameraPixelFormat,
        bytesPerRow: Int? = nil
    ) {
        self.dataBase64 = dataBase64
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
        self.bytesPerRow = bytesPerRow
    }
}

/// Options for camera frame processing.
public struct CameraFrameOptions {
    /// Whether to normalize output to [0, 1].
    public var normalize: Bool = true
    
    /// Custom mean for normalization.
    public var mean: [Float]?
    
    /// Custom std for normalization.
    public var std: [Float]?
    
    /// Target resize dimensions (nil for no resize).
    public var resize: (width: Int, height: Int)?
    
    /// Output color format.
    public var colorFormat: ColorFormat = .rgb
    
    /// Output data layout.
    public var dataLayout: DataLayout = .hwc
    
    /// Whether to apply rotation correction.
    public var rotationDegrees: Int = 0
    
    public init(
        normalize: Bool = true,
        mean: [Float]? = nil,
        std: [Float]? = nil,
        resize: (width: Int, height: Int)? = nil,
        colorFormat: ColorFormat = .rgb,
        dataLayout: DataLayout = .hwc,
        rotationDegrees: Int = 0
    ) {
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.resize = resize
        self.colorFormat = colorFormat
        self.dataLayout = dataLayout
        self.rotationDegrees = rotationDegrees
    }
}

/// YUV conversion options.
public struct YUVConversionOptions {
    /// Pixel format of the source.
    public let pixelFormat: CameraPixelFormat
    
    /// Width of the frame.
    public let width: Int
    
    /// Height of the frame.
    public let height: Int
    
    /// Y plane data as base64.
    public let yPlaneBase64: String
    
    /// UV plane data as base64 (for NV12/NV21).
    public var uvPlaneBase64: String?
    
    /// U plane data as base64 (for YUV420).
    public var uPlaneBase64: String?
    
    /// V plane data as base64 (for YUV420).
    public var vPlaneBase64: String?
    
    public init(
        pixelFormat: CameraPixelFormat,
        width: Int,
        height: Int,
        yPlaneBase64: String,
        uvPlaneBase64: String? = nil,
        uPlaneBase64: String? = nil,
        vPlaneBase64: String? = nil
    ) {
        self.pixelFormat = pixelFormat
        self.width = width
        self.height = height
        self.yPlaneBase64 = yPlaneBase64
        self.uvPlaneBase64 = uvPlaneBase64
        self.uPlaneBase64 = uPlaneBase64
        self.vPlaneBase64 = vPlaneBase64
    }
}

/// Result of camera frame processing.
public struct CameraFrameResult {
    /// Processed pixel data.
    public let pixelData: PixelDataResult
    
    /// Processing time in milliseconds.
    public let processingTimeMs: Double
    
    /// Original frame dimensions.
    public let originalWidth: Int
    public let originalHeight: Int
}

// MARK: - Camera Frame Utilities

/// High-performance camera frame processing utilities.
///
/// Optimized for real-time processing of camera frames from vision-camera
/// or AVFoundation capture sessions.
///
/// ## Supported Formats
///
/// - **YUV420**: Planar YUV with separate U and V planes
/// - **NV12**: Semi-planar YUV with interleaved UV plane
/// - **NV21**: Semi-planar YUV with interleaved VU plane
/// - **BGRA**: 4-channel BGRA (common on iOS)
/// - **RGBA**: 4-channel RGBA
///
/// ## Example
///
/// ```swift
/// // Process camera frame for ML inference
/// let options = CameraFrameOptions(
///     normalize: true,
///     resize: (width: 224, height: 224),
///     colorFormat: .rgb,
///     dataLayout: .nchw
/// )
///
/// let result = try CameraFrameUtilities.processCameraFrame(
///     source: frameSource,
///     options: options
/// )
///
/// // Use result.pixelData.data for inference
/// ```
public enum CameraFrameUtilities {
    
    // MARK: - Process Camera Frame
    
    /// Process a camera frame for ML inference.
    ///
    /// - Parameters:
    ///   - source: Camera frame source
    ///   - options: Processing options
    /// - Returns: Camera frame result
    /// - Throws: `PixelUtilsError` if processing fails
    public static func processCameraFrame(
        source: CameraFrameSource,
        options: CameraFrameOptions = CameraFrameOptions()
    ) throws -> CameraFrameResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Decode base64 data
        guard let frameData = Data(base64Encoded: source.dataBase64) else {
            throw PixelUtilsError.invalidOptions("Invalid base64 frame data")
        }
        
        // Convert to RGB based on pixel format
        var rgbData: [UInt8]
        
        switch source.pixelFormat {
        case .yuv420, .nv12, .nv21:
            // Need separate plane handling
            rgbData = try convertYUVFrameToRGB(
                data: frameData,
                width: source.width,
                height: source.height,
                format: source.pixelFormat
            )
            
        case .bgra:
            rgbData = convertBGRAToRGB(
                data: [UInt8](frameData),
                width: source.width,
                height: source.height
            )
            
        case .rgba:
            rgbData = convertRGBAToRGB(
                data: [UInt8](frameData),
                width: source.width,
                height: source.height
            )
        }
        
        // Create CGImage from RGB data
        let cgImage = try createCGImage(
            from: rgbData,
            width: source.width,
            height: source.height
        )
        
        // Build pixel extraction options
        var pixelOptions = PixelDataOptions(
            colorFormat: options.colorFormat,
            dataLayout: options.dataLayout
        )
        
        if let resize = options.resize {
            pixelOptions.resize = ResizeOptions(width: resize.width, height: resize.height)
        }
        
        if options.normalize {
            if let mean = options.mean, let std = options.std {
                pixelOptions.normalization = Normalization(
                    preset: .custom,
                    mean: mean,
                    std: std
                )
            } else {
                pixelOptions.normalization = .scale
            }
        } else {
            pixelOptions.normalization = .raw
        }
        
        let result = try extractPixelsSync(from: cgImage, options: pixelOptions)
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return CameraFrameResult(
            pixelData: result,
            processingTimeMs: processingTime,
            originalWidth: source.width,
            originalHeight: source.height
        )
    }
    
    // MARK: - YUV Conversion
    
    /// Convert YUV data to RGB.
    ///
    /// - Parameters:
    ///   - options: YUV conversion options
    /// - Returns: RGB data as [UInt8]
    /// - Throws: `PixelUtilsError` if conversion fails
    public static func convertYUVToRGB(
        options: YUVConversionOptions
    ) throws -> [UInt8] {
        // Decode Y plane
        guard let yData = Data(base64Encoded: options.yPlaneBase64) else {
            throw PixelUtilsError.invalidOptions("Invalid Y plane base64 data")
        }
        
        var rgbData: [UInt8]
        
        switch options.pixelFormat {
        case .nv12:
            guard let uvBase64 = options.uvPlaneBase64,
                  let uvData = Data(base64Encoded: uvBase64) else {
                throw PixelUtilsError.invalidOptions("NV12 requires UV plane data")
            }
            rgbData = convertNV12ToRGB(
                yPlane: [UInt8](yData),
                uvPlane: [UInt8](uvData),
                width: options.width,
                height: options.height,
                uvFirst: true
            )
            
        case .nv21:
            guard let uvBase64 = options.uvPlaneBase64,
                  let uvData = Data(base64Encoded: uvBase64) else {
                throw PixelUtilsError.invalidOptions("NV21 requires UV plane data")
            }
            rgbData = convertNV12ToRGB(
                yPlane: [UInt8](yData),
                uvPlane: [UInt8](uvData),
                width: options.width,
                height: options.height,
                uvFirst: false  // VU order
            )
            
        case .yuv420:
            guard let uBase64 = options.uPlaneBase64,
                  let vBase64 = options.vPlaneBase64,
                  let uData = Data(base64Encoded: uBase64),
                  let vData = Data(base64Encoded: vBase64) else {
                throw PixelUtilsError.invalidOptions("YUV420 requires U and V plane data")
            }
            rgbData = convertYUV420ToRGB(
                yPlane: [UInt8](yData),
                uPlane: [UInt8](uData),
                vPlane: [UInt8](vData),
                width: options.width,
                height: options.height
            )
            
        default:
            throw PixelUtilsError.invalidOptions(
                "Unsupported pixel format for YUV conversion: \(options.pixelFormat)"
            )
        }
        
        return rgbData
    }
    
    // MARK: - Private Helpers
    
    private static func convertYUVFrameToRGB(
        data: Data,
        width: Int,
        height: Int,
        format: CameraPixelFormat
    ) throws -> [UInt8] {
        let bytes = [UInt8](data)
        let ySize = width * height
        let uvSize = (width / 2) * (height / 2)
        
        switch format {
        case .yuv420:
            // Planar: Y plane, then U plane, then V plane
            let uOffset = ySize
            let vOffset = ySize + uvSize
            
            guard bytes.count >= ySize + uvSize * 2 else {
                throw PixelUtilsError.invalidOptions(
                    "Insufficient data for YUV420: expected \(ySize + uvSize * 2), got \(bytes.count)"
                )
            }
            
            let yPlane = Array(bytes[0..<ySize])
            let uPlane = Array(bytes[uOffset..<(uOffset + uvSize)])
            let vPlane = Array(bytes[vOffset..<(vOffset + uvSize)])
            
            return convertYUV420ToRGB(yPlane: yPlane, uPlane: uPlane, vPlane: vPlane, width: width, height: height)
            
        case .nv12, .nv21:
            // Semi-planar: Y plane, then interleaved UV or VU
            let uvOffset = ySize
            
            guard bytes.count >= ySize + uvSize else {
                throw PixelUtilsError.invalidOptions(
                    "Insufficient data for NV12/NV21: expected \(ySize + uvSize), got \(bytes.count)"
                )
            }
            
            let yPlane = Array(bytes[0..<ySize])
            let uvPlane = Array(bytes[uvOffset...])
            
            return convertNV12ToRGB(yPlane: yPlane, uvPlane: uvPlane, width: width, height: height, uvFirst: format == .nv12)
            
        default:
            throw PixelUtilsError.invalidOptions(
                "Unsupported format for YUV conversion: \(format)"
            )
        }
    }
    
    private static func convertYUV420ToRGB(
        yPlane: [UInt8],
        uPlane: [UInt8],
        vPlane: [UInt8],
        width: Int,
        height: Int
    ) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        
        for y in 0..<height {
            for x in 0..<width {
                let yIdx = y * width + x
                let uvIdx = (y / 2) * (width / 2) + (x / 2)
                
                let yValue = Float(yPlane[yIdx])
                let uValue = Float(uPlane[uvIdx]) - 128
                let vValue = Float(vPlane[uvIdx]) - 128
                
                // YUV to RGB conversion (BT.601)
                let r = yValue + 1.402 * vValue
                let g = yValue - 0.344136 * uValue - 0.714136 * vValue
                let b = yValue + 1.772 * uValue
                
                let rgbIdx = yIdx * 3
                rgb[rgbIdx] = UInt8(clamping: Int(r.rounded()))
                rgb[rgbIdx + 1] = UInt8(clamping: Int(g.rounded()))
                rgb[rgbIdx + 2] = UInt8(clamping: Int(b.rounded()))
            }
        }
        
        return rgb
    }
    
    private static func convertNV12ToRGB(
        yPlane: [UInt8],
        uvPlane: [UInt8],
        width: Int,
        height: Int,
        uvFirst: Bool
    ) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        
        for y in 0..<height {
            for x in 0..<width {
                let yIdx = y * width + x
                let uvIdx = (y / 2) * width + (x / 2) * 2
                
                let yValue = Float(yPlane[yIdx])
                let uValue: Float
                let vValue: Float
                
                if uvFirst {
                    // NV12: UV order
                    uValue = Float(uvPlane[uvIdx]) - 128
                    vValue = Float(uvPlane[uvIdx + 1]) - 128
                } else {
                    // NV21: VU order
                    vValue = Float(uvPlane[uvIdx]) - 128
                    uValue = Float(uvPlane[uvIdx + 1]) - 128
                }
                
                // YUV to RGB conversion (BT.601)
                let r = yValue + 1.402 * vValue
                let g = yValue - 0.344136 * uValue - 0.714136 * vValue
                let b = yValue + 1.772 * uValue
                
                let rgbIdx = yIdx * 3
                rgb[rgbIdx] = UInt8(clamping: Int(r.rounded()))
                rgb[rgbIdx + 1] = UInt8(clamping: Int(g.rounded()))
                rgb[rgbIdx + 2] = UInt8(clamping: Int(b.rounded()))
            }
        }
        
        return rgb
    }
    
    private static func convertBGRAToRGB(
        data: [UInt8],
        width: Int,
        height: Int
    ) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        
        for i in 0..<(width * height) {
            let bgraIdx = i * 4
            let rgbIdx = i * 3
            
            rgb[rgbIdx] = data[bgraIdx + 2]     // R from B position
            rgb[rgbIdx + 1] = data[bgraIdx + 1] // G
            rgb[rgbIdx + 2] = data[bgraIdx]     // B from R position
        }
        
        return rgb
    }
    
    private static func convertRGBAToRGB(
        data: [UInt8],
        width: Int,
        height: Int
    ) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        
        for i in 0..<(width * height) {
            let rgbaIdx = i * 4
            let rgbIdx = i * 3
            
            rgb[rgbIdx] = data[rgbaIdx]
            rgb[rgbIdx + 1] = data[rgbaIdx + 1]
            rgb[rgbIdx + 2] = data[rgbaIdx + 2]
        }
        
        return rgb
    }
    
    private static func createCGImage(
        from rgbData: [UInt8],
        width: Int,
        height: Int
    ) throws -> CGImage {
        // Convert RGB to RGBA for CGImage
        var rgbaData = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let rgbIdx = i * 3
            let rgbaIdx = i * 4
            rgbaData[rgbaIdx] = rgbData[rgbIdx]
            rgbaData[rgbaIdx + 1] = rgbData[rgbIdx + 1]
            rgbaData[rgbaIdx + 2] = rgbData[rgbIdx + 2]
            rgbaData[rgbaIdx + 3] = 255
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(data: Data(rgbaData) as CFData) else {
            throw PixelUtilsError.processingFailed("Failed to create data provider")
        }
        
        guard let cgImage = CGImage(
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
            throw PixelUtilsError.processingFailed("Failed to create CGImage")
        }
        
        return cgImage
    }
    
    private static func extractPixelsSync(
        from cgImage: CGImage,
        options: PixelDataOptions
    ) throws -> PixelDataResult {
        // Use a simple synchronous extraction
        let width = cgImage.width
        let height = cgImage.height
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create context")
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to float and normalize
        let channels = options.colorFormat.channelCount
        var floatData = [Float](repeating: 0, count: width * height * channels)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIdx = (y * width + x) * 4
                let outIdx = (y * width + x) * channels
                
                switch options.colorFormat {
                case .rgb:
                    floatData[outIdx] = Float(pixelData[pixelIdx]) / 255.0
                    floatData[outIdx + 1] = Float(pixelData[pixelIdx + 1]) / 255.0
                    floatData[outIdx + 2] = Float(pixelData[pixelIdx + 2]) / 255.0
                case .grayscale:
                    let r = Float(pixelData[pixelIdx])
                    let g = Float(pixelData[pixelIdx + 1])
                    let b = Float(pixelData[pixelIdx + 2])
                    floatData[outIdx] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                default:
                    floatData[outIdx] = Float(pixelData[pixelIdx]) / 255.0
                    if channels > 1 { floatData[outIdx + 1] = Float(pixelData[pixelIdx + 1]) / 255.0 }
                    if channels > 2 { floatData[outIdx + 2] = Float(pixelData[pixelIdx + 2]) / 255.0 }
                    if channels > 3 { floatData[outIdx + 3] = Float(pixelData[pixelIdx + 3]) / 255.0 }
                }
            }
        }
        
        // Apply normalization if not scale
        if options.normalization.preset != .scale && options.normalization.preset != .raw {
            if let mean = options.normalization.mean, let std = options.normalization.std {
                for y in 0..<height {
                    for x in 0..<width {
                        for c in 0..<channels {
                            let idx = (y * width + x) * channels + c
                            floatData[idx] = (floatData[idx] - mean[c]) / std[c]
                        }
                    }
                }
            }
        }
        
        return PixelDataResult(
            data: floatData,
            width: width,
            height: height,
            channels: channels,
            colorFormat: options.colorFormat,
            dataLayout: options.dataLayout,
            shape: [height, width, channels],
            processingTimeMs: 0
        )
    }
}
