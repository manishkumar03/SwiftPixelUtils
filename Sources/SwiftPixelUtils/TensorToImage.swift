//
//  TensorToImage.swift
//  SwiftPixelUtils
//
//  Convert processed tensors back to images
//

import Foundation
import CoreGraphics

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Tensor to Image Options

/// Options for converting tensor data back to an image.
public struct TensorToImageOptions {
    /// Number of channels (1 for grayscale, 3 for RGB, 4 for RGBA).
    public var channels: Int = 3
    
    /// Data layout of the input tensor.
    public var dataLayout: DataLayout = .hwc
    
    /// Whether to denormalize the data.
    public var denormalize: Bool = true
    
    /// Mean values used for normalization (to reverse).
    public var mean: [Float] = [0.485, 0.456, 0.406]
    
    /// Standard deviation values used for normalization (to reverse).
    public var std: [Float] = [0.229, 0.224, 0.225]
    
    /// Output format (png or jpeg).
    public var outputFormat: ImageOutputFormat = .png
    
    /// JPEG quality (0-100) if outputFormat is jpeg.
    public var jpegQuality: Int = 90
    
    public init(
        channels: Int = 3,
        dataLayout: DataLayout = .hwc,
        denormalize: Bool = true,
        mean: [Float] = [0.485, 0.456, 0.406],
        std: [Float] = [0.229, 0.224, 0.225],
        outputFormat: ImageOutputFormat = .png,
        jpegQuality: Int = 90
    ) {
        self.channels = channels
        self.dataLayout = dataLayout
        self.denormalize = denormalize
        self.mean = mean
        self.std = std
        self.outputFormat = outputFormat
        self.jpegQuality = jpegQuality
    }
}

/// Output format for tensor-to-image conversion.
public enum ImageOutputFormat: String {
    case png
    case jpeg
}

// MARK: - Tensor to Image Result

/// Result of tensor to image conversion.
public struct TensorToImageResult {
    /// Image as base64 encoded string.
    public let imageBase64: String
    
    /// The CGImage.
    public let cgImage: CGImage
    
    /// Width of the output image.
    public let width: Int
    
    /// Height of the output image.
    public let height: Int
    
    /// Processing time in milliseconds.
    public let processingTimeMs: Double
}

// MARK: - Tensor to Image

/// Convert processed tensor data back to viewable images.
///
/// This is useful for:
/// - Debugging preprocessing pipelines
/// - Visualizing augmented images
/// - Saving intermediate results
///
/// ## Example
///
/// ```swift
/// // Convert tensor back to image
/// let result = try TensorToImage.convert(
///     data: tensorData,
///     width: 224,
///     height: 224,
///     options: TensorToImageOptions(
///         denormalize: true,
///         mean: [0.485, 0.456, 0.406],
///         std: [0.229, 0.224, 0.225]
///     )
/// )
/// print("Image: \(result.imageBase64.prefix(50))...")
/// ```
public enum TensorToImage {
    
    // MARK: - Convert
    
    /// Convert tensor data to an image (async).
    ///
    /// - Parameters:
    ///   - data: Float array of tensor data
    ///   - width: Width of the image
    ///   - height: Height of the image
    ///   - options: Conversion options
    /// - Returns: Tensor to image result
    /// - Throws: `PixelUtilsError` if conversion fails
    public static func convert(
        data: [Float],
        width: Int,
        height: Int,
        options: TensorToImageOptions = TensorToImageOptions()
    ) async throws -> TensorToImageResult {
        try convertSync(data: data, width: width, height: height, options: options)
    }
    
    /// Convert tensor data to an image (synchronous).
    ///
    /// - Parameters:
    ///   - data: Float array of tensor data
    ///   - width: Width of the image
    ///   - height: Height of the image
    ///   - options: Conversion options
    /// - Returns: Tensor to image result
    /// - Throws: `PixelUtilsError` if conversion fails
    public static func convertSync(
        data: [Float],
        width: Int,
        height: Int,
        options: TensorToImageOptions = TensorToImageOptions()
    ) throws -> TensorToImageResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let channels = options.channels
        let expectedSize = width * height * channels
        guard data.count >= expectedSize else {
            throw PixelUtilsError.invalidOptions(
                "Data size \(data.count) is less than expected \(expectedSize) for \(width)x\(height)x\(channels)"
            )
        }
        
        // Denormalize if needed
        var processedData = data
        if options.denormalize {
            processedData = try denormalizeData(
                data,
                width: width,
                height: height,
                channels: channels,
                layout: options.dataLayout,
                mean: options.mean,
                std: options.std
            )
        }
        
        // Convert to HWC if needed
        var hwcData = processedData
        if options.dataLayout == .chw || options.dataLayout == .nchw {
            hwcData = convertCHWToHWC(processedData, width: width, height: height, channels: channels)
        }
        
        // Convert to 8-bit RGBA
        let rgbaData = convertToRGBA(hwcData, width: width, height: height, channels: channels)
        
        // Create CGImage
        let cgImage = try createCGImage(from: rgbaData, width: width, height: height)
        
        // Encode to base64
        let base64 = try encodeToBase64(cgImage, format: options.outputFormat, quality: options.jpegQuality)
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return TensorToImageResult(
            imageBase64: base64,
            cgImage: cgImage,
            width: width,
            height: height,
            processingTimeMs: processingTime
        )
    }
    
    /// Convert a PixelDataResult back to an image.
    ///
    /// - Parameters:
    ///   - result: The pixel data result to convert
    ///   - normalization: The normalization that was applied
    /// - Returns: Tensor to image result
    /// - Throws: `PixelUtilsError` if conversion fails
    public static func convert(
        result: PixelDataResult,
        normalization: Normalization = .imagenet
    ) async throws -> TensorToImageResult {
        var options = TensorToImageOptions()
        options.channels = result.channels
        options.dataLayout = result.dataLayout
        
        // Set denormalization parameters based on normalization used
        switch normalization.preset {
        case .scale:
            options.denormalize = false // Just scale by 255
            
        case .imagenet:
            options.denormalize = true
            options.mean = [0.485, 0.456, 0.406]
            options.std = [0.229, 0.224, 0.225]
            
        case .tensorflow:
            options.denormalize = true
            // TensorFlow uses [-1, 1], so mean=0.5, std=0.5 to get back to [0, 1]
            options.mean = [0.5, 0.5, 0.5]
            options.std = [0.5, 0.5, 0.5]
            
        case .raw:
            options.denormalize = false
            
        case .custom:
            options.denormalize = true
            if let mean = normalization.mean {
                options.mean = mean
            }
            if let std = normalization.std {
                options.std = std
            }
        }
        
        return try await convert(
            data: result.data,
            width: result.width,
            height: result.height,
            options: options
        )
    }
    
    // MARK: - Private Helpers
    
    private static func denormalizeData(
        _ data: [Float],
        width: Int,
        height: Int,
        channels: Int,
        layout: DataLayout,
        mean: [Float],
        std: [Float]
    ) throws -> [Float] {
        guard mean.count >= channels && std.count >= channels else {
            throw PixelUtilsError.invalidOptions(
                "Mean and std arrays must have at least \(channels) values"
            )
        }
        
        var result = [Float](repeating: 0, count: data.count)
        
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<channels {
                    let idx: Int
                    switch layout {
                    case .hwc:
                        idx = (y * width + x) * channels + c
                    case .chw:
                        idx = c * height * width + y * width + x
                    case .nhwc:
                        idx = (y * width + x) * channels + c
                    case .nchw:
                        idx = c * height * width + y * width + x
                    }
                    
                    // Reverse: output = (input * std) + mean
                    // Then scale to [0, 255]
                    let normalized = data[idx]
                    let denormalized = (normalized * std[c]) + mean[c]
                    result[idx] = denormalized * 255.0
                }
            }
        }
        
        return result
    }
    
    private static func convertCHWToHWC(_ data: [Float], width: Int, height: Int, channels: Int) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        
        for c in 0..<channels {
            for y in 0..<height {
                for x in 0..<width {
                    let chwIdx = c * height * width + y * width + x
                    let hwcIdx = (y * width + x) * channels + c
                    result[hwcIdx] = data[chwIdx]
                }
            }
        }
        
        return result
    }
    
    private static func convertToRGBA(_ data: [Float], width: Int, height: Int, channels: Int) -> [UInt8] {
        var rgbaData = [UInt8](repeating: 255, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = (y * width + x) * channels
                let dstIdx = (y * width + x) * 4
                
                if channels == 1 {
                    // Grayscale
                    let gray = UInt8(clamping: Int(data[srcIdx].rounded()))
                    rgbaData[dstIdx] = gray
                    rgbaData[dstIdx + 1] = gray
                    rgbaData[dstIdx + 2] = gray
                    rgbaData[dstIdx + 3] = 255
                } else if channels == 3 {
                    // RGB
                    rgbaData[dstIdx] = UInt8(clamping: Int(data[srcIdx].rounded()))
                    rgbaData[dstIdx + 1] = UInt8(clamping: Int(data[srcIdx + 1].rounded()))
                    rgbaData[dstIdx + 2] = UInt8(clamping: Int(data[srcIdx + 2].rounded()))
                    rgbaData[dstIdx + 3] = 255
                } else if channels == 4 {
                    // RGBA
                    rgbaData[dstIdx] = UInt8(clamping: Int(data[srcIdx].rounded()))
                    rgbaData[dstIdx + 1] = UInt8(clamping: Int(data[srcIdx + 1].rounded()))
                    rgbaData[dstIdx + 2] = UInt8(clamping: Int(data[srcIdx + 2].rounded()))
                    rgbaData[dstIdx + 3] = UInt8(clamping: Int(data[srcIdx + 3].rounded()))
                }
            }
        }
        
        return rgbaData
    }
    
    private static func createCGImage(from rgbaData: [UInt8], width: Int, height: Int) throws -> CGImage {
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
    
    private static func encodeToBase64(_ cgImage: CGImage, format: ImageOutputFormat, quality: Int) throws -> String {
        #if canImport(UIKit)
        let uiImage = UIImage(cgImage: cgImage)
        let data: Data?
        switch format {
        case .png:
            data = uiImage.pngData()
        case .jpeg:
            data = uiImage.jpegData(compressionQuality: CGFloat(quality) / 100.0)
        }
        guard let imageData = data else {
            throw PixelUtilsError.processingFailed("Failed to encode image")
        }
        return imageData.base64EncodedString()
        
        #elseif canImport(AppKit)
        let bitmap = NSBitmapImageRep(cgImage: cgImage)
        let data: Data?
        switch format {
        case .png:
            data = bitmap.representation(using: .png, properties: [:])
        case .jpeg:
            data = bitmap.representation(using: .jpeg, properties: [.compressionFactor: CGFloat(quality) / 100.0])
        }
        guard let imageData = data else {
            throw PixelUtilsError.processingFailed("Failed to encode image")
        }
        return imageData.base64EncodedString()
        #endif
    }
}
