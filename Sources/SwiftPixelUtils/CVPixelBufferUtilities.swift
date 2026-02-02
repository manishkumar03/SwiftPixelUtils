import Foundation
import CoreVideo
import Accelerate
import CoreGraphics

// MARK: - CVPixelBuffer Utilities

/// Utilities for converting CVPixelBuffer to tensor-ready data
public enum CVPixelBufferUtilities {
    
    /// Supported pixel formats for buffer creation
    public enum PixelFormat {
        case bgra8
        case rgba8
        
        var cvFormat: OSType {
            switch self {
            case .bgra8: return kCVPixelFormatType_32BGRA
            case .rgba8: return kCVPixelFormatType_32RGBA
            }
        }
    }
    
    /// Result of CVPixelBuffer conversion
    public struct ConversionResult {
        /// Normalized float data ready for model input
        public let data: [Float]
        /// Original pixel buffer width
        public let originalWidth: Int
        /// Original pixel buffer height
        public let originalHeight: Int
        /// Output tensor width
        public let tensorWidth: Int
        /// Output tensor height
        public let tensorHeight: Int
        /// Number of channels
        public let channels: Int
        
        public init(data: [Float], originalWidth: Int, originalHeight: Int, tensorWidth: Int, tensorHeight: Int, channels: Int) {
            self.data = data
            self.originalWidth = originalWidth
            self.originalHeight = originalHeight
            self.tensorWidth = tensorWidth
            self.tensorHeight = tensorHeight
            self.channels = channels
        }
    }
    
    /// Channel ordering for output tensor data
    public enum ChannelOrder {
        case rgb
        case bgr
    }
    
    // MARK: - Pixel Buffer Creation
    
    /// Creates a CVPixelBuffer from a CGImage, resized to the specified dimensions.
    ///
    /// This is useful for preparing images for CoreML model input.
    ///
    /// ## Usage
    ///
    /// ```swift
    /// let pixelBuffer = try CVPixelBufferUtilities.createPixelBuffer(
    ///     from: cgImage,
    ///     width: 518,
    ///     height: 518,
    ///     pixelFormat: .bgra8
    /// )
    ///
    /// let input = try MLDictionaryFeatureProvider(dictionary: [
    ///     "image": MLFeatureValue(pixelBuffer: pixelBuffer)
    /// ])
    /// ```
    ///
    /// - Parameters:
    ///   - cgImage: Source image to convert
    ///   - width: Target width for the pixel buffer
    ///   - height: Target height for the pixel buffer
    ///   - pixelFormat: Pixel format for the buffer (default .bgra8)
    /// - Returns: CVPixelBuffer containing the resized image
    /// - Throws: ``PixelUtilsError`` if buffer creation fails
    public static func createPixelBuffer(
        from cgImage: CGImage,
        width: Int,
        height: Int,
        pixelFormat: PixelFormat = .bgra8
    ) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormat.cvFormat,
            attrs as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw PixelUtilsError.processingFailed("Failed to create pixel buffer (status: \(status))")
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let bitmapInfo: UInt32
        switch pixelFormat {
        case .bgra8:
            bitmapInfo = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        case .rgba8:
            bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context for pixel buffer")
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
    
    /// Creates a CVPixelBuffer from a CGImage using the image's original dimensions.
    ///
    /// - Parameters:
    ///   - cgImage: Source image to convert
    ///   - pixelFormat: Pixel format for the buffer (default .bgra8)
    /// - Returns: CVPixelBuffer containing the image at original size
    /// - Throws: ``PixelUtilsError`` if buffer creation fails
    public static func createPixelBuffer(
        from cgImage: CGImage,
        pixelFormat: PixelFormat = .bgra8
    ) throws -> CVPixelBuffer {
        return try createPixelBuffer(
            from: cgImage,
            width: cgImage.width,
            height: cgImage.height,
            pixelFormat: pixelFormat
        )
    }
    
    /// Converts CVPixelBuffer to normalized Float array for model input.
    /// - Parameters:
    ///   - pixelBuffer: Input pixel buffer from camera or image
    ///   - targetSize: Target size for the tensor (width, height)
    ///   - normalization: Normalization to apply
    ///   - channelOrder: Desired channel order (default .rgb)
    ///   - includeAlpha: Whether to include alpha channel (default false)
    /// - Returns: ConversionResult with normalized data
    /// - Throws: PixelUtilsError if conversion fails
    public static func toTensorData(
        _ pixelBuffer: CVPixelBuffer,
        targetSize: (width: Int, height: Int)? = nil,
        normalization: Normalization = Normalization(preset: .scale),
        channelOrder: ChannelOrder = .rgb,
        includeAlpha: Bool = false
    ) throws -> ConversionResult {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        // Extract RGBA data based on pixel format
        var rgbaData: [UInt8]
        
        switch pixelFormat {
        case kCVPixelFormatType_32BGRA:
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                throw PixelUtilsError.processingFailed("Cannot access pixel buffer")
            }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            rgbaData = extractBGRA(baseAddress: baseAddress, width: width, height: height, bytesPerRow: bytesPerRow)
            
        case kCVPixelFormatType_32RGBA:
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                throw PixelUtilsError.processingFailed("Cannot access pixel buffer")
            }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            rgbaData = extractRGBA(baseAddress: baseAddress, width: width, height: height, bytesPerRow: bytesPerRow)
            
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            rgbaData = try extractYUV420BiPlanar(pixelBuffer: pixelBuffer)
            
        default:
            throw PixelUtilsError.processingFailed("Pixel format \(getPixelFormatDescription(pixelFormat)) not supported")
        }
        
        // Resize if needed
        let targetWidth = targetSize?.width ?? width
        let targetHeight = targetSize?.height ?? height
        
        if targetWidth != width || targetHeight != height {
            rgbaData = resizeRGBA(data: rgbaData, sourceWidth: width, sourceHeight: height, targetWidth: targetWidth, targetHeight: targetHeight)
        }
        
        // Convert to float and arrange channels
        let channels = includeAlpha ? 4 : 3
        var floatData = [Float](repeating: 0, count: targetWidth * targetHeight * channels)
        
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let srcIdx = (y * targetWidth + x) * 4
                let dstIdx = (y * targetWidth + x) * channels
                
                let r = Float(rgbaData[srcIdx]) / 255.0
                let g = Float(rgbaData[srcIdx + 1]) / 255.0
                let b = Float(rgbaData[srcIdx + 2]) / 255.0
                let a = Float(rgbaData[srcIdx + 3]) / 255.0
                
                switch channelOrder {
                case .rgb:
                    floatData[dstIdx] = r
                    floatData[dstIdx + 1] = g
                    floatData[dstIdx + 2] = b
                    if includeAlpha { floatData[dstIdx + 3] = a }
                case .bgr:
                    floatData[dstIdx] = b
                    floatData[dstIdx + 1] = g
                    floatData[dstIdx + 2] = r
                    if includeAlpha { floatData[dstIdx + 3] = a }
                }
            }
        }
        
        // Apply normalization
        floatData = applyNormalization(floatData, normalization: normalization, channels: channels)
        
        return ConversionResult(
            data: floatData,
            originalWidth: width,
            originalHeight: height,
            tensorWidth: targetWidth,
            tensorHeight: targetHeight,
            channels: channels
        )
    }
    
    /// Gets a human-readable description of a pixel format.
    /// - Parameter format: CVPixelFormatType
    /// - Returns: String description
    public static func getPixelFormatDescription(_ format: OSType) -> String {
        switch format {
        case kCVPixelFormatType_32BGRA: return "32BGRA"
        case kCVPixelFormatType_32RGBA: return "32RGBA"
        case kCVPixelFormatType_32ARGB: return "32ARGB"
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: return "420YpCbCr8BiPlanarVideoRange"
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: return "420YpCbCr8BiPlanarFullRange"
        default: return "Unknown(\(format))"
        }
    }
    
    // MARK: Private Helpers
    
    private static func extractBGRA(baseAddress: UnsafeMutableRawPointer, width: Int, height: Int, bytesPerRow: Int) -> [UInt8] {
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = y * bytesPerRow + x * 4
                let dstIdx = (y * width + x) * 4
                rgba[dstIdx] = ptr[srcIdx + 2]     // R
                rgba[dstIdx + 1] = ptr[srcIdx + 1] // G
                rgba[dstIdx + 2] = ptr[srcIdx]     // B
                rgba[dstIdx + 3] = ptr[srcIdx + 3] // A
            }
        }
        
        return rgba
    }
    
    private static func extractRGBA(baseAddress: UnsafeMutableRawPointer, width: Int, height: Int, bytesPerRow: Int) -> [UInt8] {
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = y * bytesPerRow + x * 4
                let dstIdx = (y * width + x) * 4
                rgba[dstIdx] = ptr[srcIdx]
                rgba[dstIdx + 1] = ptr[srcIdx + 1]
                rgba[dstIdx + 2] = ptr[srcIdx + 2]
                rgba[dstIdx + 3] = ptr[srcIdx + 3]
            }
        }
        
        return rgba
    }
    
    private static func extractYUV420BiPlanar(pixelBuffer: CVPixelBuffer) throws -> [UInt8] {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2 else {
            throw PixelUtilsError.processingFailed("Expected bi-planar format")
        }
        
        guard let yPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
              let uvPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else {
            throw PixelUtilsError.processingFailed("Cannot access YUV planes")
        }
        
        let yPtr = yPlane.assumingMemoryBound(to: UInt8.self)
        let uvPtr = uvPlane.assumingMemoryBound(to: UInt8.self)
        
        let yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let uvBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
        
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let yIdx = y * yBytesPerRow + x
                let uvIdx = (y / 2) * uvBytesPerRow + (x / 2) * 2
                
                let yValue = Float(yPtr[yIdx])
                let uValue = Float(uvPtr[uvIdx]) - 128
                let vValue = Float(uvPtr[uvIdx + 1]) - 128
                
                // YUV to RGB conversion (BT.601)
                var r = yValue + 1.402 * vValue
                var g = yValue - 0.344 * uValue - 0.714 * vValue
                var b = yValue + 1.772 * uValue
                
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                let idx = (y * width + x) * 4
                rgba[idx] = UInt8(r)
                rgba[idx + 1] = UInt8(g)
                rgba[idx + 2] = UInt8(b)
                rgba[idx + 3] = 255
            }
        }
        
        return rgba
    }
    
    private static func resizeRGBA(data: [UInt8], sourceWidth: Int, sourceHeight: Int, targetWidth: Int, targetHeight: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)
        
        let scaleX = Float(sourceWidth) / Float(targetWidth)
        let scaleY = Float(sourceHeight) / Float(targetHeight)
        
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)
                let srcIdx = (srcY * sourceWidth + srcX) * 4
                let dstIdx = (y * targetWidth + x) * 4
                
                result[dstIdx] = data[srcIdx]
                result[dstIdx + 1] = data[srcIdx + 1]
                result[dstIdx + 2] = data[srcIdx + 2]
                result[dstIdx + 3] = data[srcIdx + 3]
            }
        }
        
        return result
    }
    
    private static func applyNormalization(_ data: [Float], normalization: Normalization, channels: Int) -> [Float] {
        var result = data
        
        switch normalization.preset {
        case .raw:
            // No normalization, but data is already 0-1, so multiply back to 0-255
            for i in 0..<result.count {
                result[i] = result[i] * 255.0
            }
        case .scale:
            break // Already in 0-1 range
        case .tensorflow:
            // Map [0, 1] to [-1, 1]
            for i in 0..<result.count {
                result[i] = result[i] * 2.0 - 1.0
            }
        case .imagenet:
            let mean = normalization.mean ?? [0.485, 0.456, 0.406]
            let std = normalization.std ?? [0.229, 0.224, 0.225]
            let pixelCount = result.count / channels
            for i in 0..<pixelCount {
                for c in 0..<min(channels, mean.count) {
                    let idx = i * channels + c
                    result[idx] = (result[idx] - mean[c]) / std[c]
                }
            }
        case .custom:
            if let mean = normalization.mean, let std = normalization.std {
                let pixelCount = result.count / channels
                for i in 0..<pixelCount {
                    for c in 0..<min(channels, mean.count) {
                        let idx = i * channels + c
                        let m = c < mean.count ? mean[c] : 0
                        let s = c < std.count ? std[c] : 1
                        result[idx] = (result[idx] - m) / s
                    }
                }
            }
        }
        
        return result
    }
}
