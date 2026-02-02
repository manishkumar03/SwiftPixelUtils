//
//  PixelExtractor.swift
//  SwiftPixelUtils
//
//  Core pixel data extraction functionality
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

/// A high-performance image preprocessing engine for machine learning inference pipelines.
///
/// `PixelExtractor` provides a comprehensive set of tools for loading images from various sources,
/// applying transformations, and extracting normalized pixel data in formats suitable for ML models.
///
/// ## Overview
///
/// The extraction pipeline processes images through several stages:
///
/// 1. **Loading**: Images are loaded from URLs, files, Data, base64 strings, or platform-native types
/// 2. **ROI Cropping**: Optional region-of-interest extraction
/// 3. **Resizing**: Multiple strategies (cover, contain, stretch, letterbox)
/// 4. **Pixel Extraction**: RGBA data extraction via CoreGraphics
/// 5. **Color Conversion**: Transform to target color space (RGB, BGR, HSV, LAB, etc.)
/// 6. **Normalization**: Scale values using presets (ImageNet, TensorFlow) or custom mean/std
/// 7. **Layout Transformation**: Rearrange to target memory layout (HWC, CHW, NCHW)
///
/// ## Color Space Mathematics
///
/// The library implements standard color space conversion formulas:
///
/// ### Grayscale (ITU-R BT.601 Luma)
/// ```
/// Y = 0.299R + 0.587G + 0.114B
/// ```
/// These coefficients are derived from human perception studies showing that the eye
/// is most sensitive to green, less to red, and least to blue.
///
/// ### YUV (ITU-R BT.601)
/// ```
/// Y =  0.299R + 0.587G + 0.114B
/// U = -0.147R - 0.289G + 0.436B
/// V =  0.615R - 0.515G - 0.100B
/// ```
///
/// ### YCbCr (ITU-R BT.601 with digital offsets)
/// ```
/// Y  = 0.299R + 0.587G + 0.114B
/// Cb = 128 - 0.169R - 0.331G + 0.500B
/// Cr = 128 + 0.500R - 0.419G - 0.081B
/// ```
///
/// ## Topics
///
/// ### Pixel Extraction
/// - ``getPixelData(source:options:)``
/// - ``batchGetPixelData(sources:options:concurrency:)``
///
/// ### Color Formats
/// - ``ColorFormat``
///
/// ### Normalization
/// - ``Normalization``
/// - ``NormalizationPreset``
public enum PixelExtractor {
    
    // MARK: - Main API
    
    /// Extract pixel data from a single image source for ML inference.
    ///
    /// This is the primary entry point for image preprocessing. The method handles the complete
    /// pipeline from loading the raw image to producing normalized, layout-transformed pixel data.
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Basic usage with YOLO preprocessing
    /// let result = try await PixelExtractor.getPixelData(
    ///     source: .url(imageURL),
    ///     options: ModelPresets.yolov8.options
    /// )
    ///
    /// // Access the preprocessed tensor data
    /// let tensorData = result.data  // [Float] in NCHW layout
    /// print(result.shape)           // [1, 3, 640, 640]
    /// ```
    ///
    /// ## Processing Pipeline
    ///
    /// 1. Load image from source (async for URLs)
    /// 2. Apply ROI cropping if specified
    /// 3. Resize using the specified strategy
    /// 4. Extract RGBA pixels via CoreGraphics bitmap context
    /// 5. Convert to target color format
    /// 6. Apply normalization (mean subtraction, std division)
    /// 7. Transform to target memory layout
    ///
    /// - Parameters:
    ///   - source: The image source (URL, file, Data, base64, CGImage, UIImage, or NSImage)
    ///   - options: Processing configuration including color format, resize, normalization, and layout
    /// - Returns: A ``PixelDataResult`` containing the preprocessed pixel array and metadata
    /// - Throws: ``PixelUtilsError`` if loading or processing fails
    public static func getPixelData(
        source: ImageSource,
        options: PixelDataOptions = PixelDataOptions()
    ) async throws -> PixelDataResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load CGImage from source, optionally normalizing orientation
        let cgImage = try await loadCGImage(from: source, normalizeOrientation: options.normalizeOrientation)
        
        // Track original size for letterbox info
        let originalSize = CGSize(width: cgImage.width, height: cgImage.height)
        
        // Apply ROI if specified
        var processedImage = cgImage
        if let roi = options.roi {
            processedImage = try applyROI(to: processedImage, roi: roi)
        }
        
        // Apply resize if specified, capture letterbox info
        var letterboxInfo: LetterboxInfo? = nil
        if let resize = options.resize {
            let resizeResult = try applyResize(to: processedImage, options: resize, originalSize: originalSize)
            processedImage = resizeResult.image
            letterboxInfo = resizeResult.letterboxInfo
        }
        
        let width = processedImage.width
        let height = processedImage.height
        
        // Extract RGBA pixels
        let rgbaData = try extractRGBAPixels(from: processedImage)
        
        // Convert to target color format
        let colorData = try convertColorFormat(
            rgbaData,
            width: width,
            height: height,
            targetFormat: options.colorFormat
        )
        
        let channels = options.colorFormat.channelCount
        
        // Apply normalization
        let normalizedData = try applyNormalization(
            colorData,
            normalization: options.normalization,
            channels: channels
        )
        
        // Apply data layout transformation
        let layoutData = try applyDataLayout(
            normalizedData,
            width: width,
            height: height,
            channels: channels,
            layout: options.dataLayout
        )
        
        // Generate uint8Data if requested or if using raw normalization
        let uint8Data: [UInt8]?
        if options.outputFormat == .uint8Array || options.normalization == .raw {
            // Apply layout to raw color data and convert to UInt8
            // colorData is in [0, 1] range, so multiply by 255 to get [0, 255]
            let rawLayoutData = try applyDataLayout(
                colorData,
                width: width,
                height: height,
                channels: channels,
                layout: options.dataLayout
            )
            uint8Data = rawLayoutData.map { UInt8(min(255, max(0, $0 * 255.0))) }
        } else {
            uint8Data = nil
        }
        
        // Generate int32Data if requested
        let int32Data: [Int32]?
        if options.outputFormat == .int32Array {
            // Convert normalized float data to Int32
            // For raw normalization, values are 0-255; otherwise scale appropriately
            if options.normalization == .raw {
                int32Data = layoutData.map { Int32($0) }
            } else {
                // For normalized data in [0,1] or [-1,1], scale to Int32 range
                // Using a reasonable scale factor for ML inference
                int32Data = layoutData.map { Int32($0 * 255.0) }
            }
        } else {
            int32Data = nil
        }
        
        // Generate float16Data if requested
        let float16Data: [UInt16]?
        if options.outputFormat == .float16Array {
            float16Data = convertToFloat16(layoutData)
        } else {
            float16Data = nil
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        let shape = calculateShape(
            width: width,
            height: height,
            channels: channels,
            layout: options.dataLayout
        )
        
        return PixelDataResult(
            data: layoutData,
            uint8Data: uint8Data,
            int32Data: int32Data,
            float16Data: float16Data,
            width: width,
            height: height,
            channels: channels,
            colorFormat: options.colorFormat,
            dataLayout: options.dataLayout,
            shape: shape,
            processingTimeMs: processingTime,
            letterboxInfo: letterboxInfo
        )
    }
    
    // MARK: - Simplified Framework-Specific API
    
    /// Get preprocessed image data ready for ML model inference.
    ///
    /// This is the simplest way to get data ready for your ML model. Just specify the
    /// framework and image size, and the method handles all configuration automatically.
    ///
    /// ## Example
    ///
    /// ```swift
    /// // For TensorFlow Lite quantized model (e.g., MobileNet)
    /// let input = try await PixelExtractor.getModelInput(
    ///     source: .uiImage(image),
    ///     framework: .tfliteQuantized,
    ///     width: 224,
    ///     height: 224
    /// )
    /// // input.data contains UInt8 bytes in NHWC layout, ready for TFLite
    ///
    /// // For PyTorch model
    /// let input = try await PixelExtractor.getModelInput(
    ///     source: .uiImage(image),
    ///     framework: .pytorch,
    ///     width: 224,
    ///     height: 224
    /// )
    /// // input.data contains Float32 bytes in NCHW layout with ImageNet normalization
    /// ```
    ///
    /// - Parameters:
    ///   - source: Image source (URL, file, UIImage, etc.)
    ///   - framework: Target ML framework that determines all preprocessing settings
    ///   - width: Target width for the model input
    ///   - height: Target height for the model input
    ///   - resizeStrategy: How to resize the image (default: .cover for classification, use .letterbox for detection)
    /// - Returns: ``ModelInputResult`` containing raw bytes and metadata
    /// - Throws: ``PixelUtilsError`` if preprocessing fails
    public static func getModelInput(
        source: ImageSource,
        framework: MLFramework,
        width: Int,
        height: Int,
        resizeStrategy: ResizeStrategy = .cover
    ) async throws -> ModelInputResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Get base options from framework and add resize
        var options = framework.options
        options.resize = ResizeOptions(width: width, height: height, strategy: resizeStrategy)
        
        // Extract pixel data
        let result = try await getPixelData(source: source, options: options)
        
        // Convert to raw Data based on output format
        let outputData: Data
        let dataType: String
        
        switch framework {
        case .tfliteQuantized, .openCV, .onnxQuantizedUInt8:
            // UInt8 output
            guard let uint8Data = result.uint8Data else {
                throw PixelUtilsError.processingFailed("Failed to generate UInt8 data for \(framework)")
            }
            outputData = Data(uint8Data)
            dataType = "UInt8"
            
        case .execuTorchQuantized, .onnxQuantizedInt8:
            // Int8 output (convert from raw 0-255 to -128 to 127)
            guard let uint8Data = result.uint8Data else {
                throw PixelUtilsError.processingFailed("Failed to generate data for \(framework)")
            }
            let int8Data = uint8Data.map { Int8(bitPattern: $0 &- 128) }
            outputData = int8Data.withUnsafeBufferPointer { Data(buffer: $0) }
            dataType = "Int8"
            
        case .onnxFloat16:
            // Float16 output
            guard let float16Data = result.float16Data else {
                throw PixelUtilsError.processingFailed("Failed to generate Float16 data for \(framework)")
            }
            outputData = float16Data.withUnsafeBufferPointer { ptr in
                Data(buffer: ptr)
            }
            dataType = "Float16"
            
        default:
            // Float32 output
            outputData = result.data.withUnsafeBytes { Data($0) }
            dataType = "Float32"
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return ModelInputResult(
            data: outputData,
            shape: result.shape,
            width: result.width,
            height: result.height,
            channels: result.channels,
            dataType: dataType,
            processingTimeMs: processingTime
        )
    }
    
    /// Process multiple images with concurrency control
    /// - Parameters:
    ///   - sources: Array of image sources
    ///   - options: Processing options
    ///   - concurrency: Maximum concurrent operations
    /// - Returns: Array of pixel data results
    public static func batchGetPixelData(
        sources: [ImageSource],
        options: PixelDataOptions = PixelDataOptions(),
        concurrency: Int = 4
    ) async throws -> [PixelDataResult] {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        var results: [PixelDataResult] = []
        results.reserveCapacity(sources.count)
        
        // Process in batches based on concurrency
        for batch in sources.chunked(into: concurrency) {
            let batchResults = try await withThrowingTaskGroup(
                of: PixelDataResult.self
            ) { group in
                for source in batch {
                    group.addTask {
                        try await getPixelData(source: source, options: options)
                    }
                }
                
                var collected: [PixelDataResult] = []
                for try await result in group {
                    collected.append(result)
                }
                return collected
            }
            results.append(contentsOf: batchResults)
        }
        
        return results
    }
    
    // MARK: - Image Loading
    
    private static func loadCGImage(from source: ImageSource, normalizeOrientation: Bool = false) async throws -> CGImage {
        switch source {
        case .url(let url):
            return try await loadFromURL(url)
            
        case .file(let url):
            return try loadFromFile(url)
            
        case .data(let data):
            return try loadFromData(data)
            
        case .base64(let base64String):
            return try loadFromBase64(base64String)
            
        case .cgImage(let cgImage):
            return cgImage
            
        #if canImport(UIKit)
        case .uiImage(let uiImage):
            // Handle orientation normalization if requested
            if normalizeOrientation && uiImage.imageOrientation != .up {
                let normalizedImage = normalizeImageOrientation(uiImage)
                guard let cgImage = normalizedImage.cgImage else {
                    throw PixelUtilsError.loadFailed("Cannot get CGImage from normalized UIImage")
                }
                return cgImage
            }
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
    
    #if canImport(UIKit)
    /// Normalizes UIImage orientation by redrawing to .up orientation.
    /// This fixes EXIF rotation issues that cause silent flips/rotations.
    private static func normalizeImageOrientation(_ image: UIImage) -> UIImage {
        guard image.imageOrientation != .up else { return image }
        
        let size = image.size
        UIGraphicsBeginImageContextWithOptions(size, false, image.scale)
        defer { UIGraphicsEndImageContext() }
        
        image.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? image
    }
    #endif
    
    private static func loadFromURL(_ url: URL) async throws -> CGImage {
        let (data, _) = try await URLSession.shared.data(from: url)
        return try loadFromData(data)
    }
    
    private static func loadFromFile(_ url: URL) throws -> CGImage {
        guard let provider = CGDataProvider(url: url as CFURL) else {
            throw PixelUtilsError.loadFailed("Cannot create data provider from file URL")
        }
        
        let pathExtension = url.pathExtension.lowercased()
        
        if pathExtension == "png" {
            guard let image = CGImage(pngDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
                throw PixelUtilsError.loadFailed("Cannot decode PNG image")
            }
            return image
        } else if ["jpg", "jpeg"].contains(pathExtension) {
            guard let image = CGImage(jpegDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
                throw PixelUtilsError.loadFailed("Cannot decode JPEG image")
            }
            return image
        } else {
            // Try generic image loading
            let data = try Data(contentsOf: url)
            return try loadFromData(data)
        }
    }
    
    private static func loadFromData(_ data: Data) throws -> CGImage {
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw PixelUtilsError.loadFailed("Cannot create data provider")
        }
        
        guard let image = CGImage(jpegDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) ??
                          CGImage(pngDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
            throw PixelUtilsError.loadFailed("Cannot decode image data")
        }
        
        return image
    }
    
    private static func loadFromBase64(_ base64String: String) throws -> CGImage {
        // Remove data URI prefix if present
        let cleanBase64 = base64String
            .replacingOccurrences(of: "data:image/png;base64,", with: "")
            .replacingOccurrences(of: "data:image/jpeg;base64,", with: "")
            .replacingOccurrences(of: "data:image/jpg;base64,", with: "")
        
        guard let data = Data(base64Encoded: cleanBase64) else {
            throw PixelUtilsError.loadFailed("Invalid base64 string")
        }
        
        return try loadFromData(data)
    }
    
    // MARK: - Image Processing
    
    /// Internal result from resize operations that includes optional letterbox info
    private struct ResizeResult {
        let image: CGImage
        let letterboxInfo: LetterboxInfo?
    }
    
    private static func applyROI(to image: CGImage, roi: ROI) throws -> CGImage {
        let rect = CGRect(x: roi.x, y: roi.y, width: roi.width, height: roi.height)
        
        // Validate ROI bounds
        if rect.maxX > CGFloat(image.width) || rect.maxY > CGFloat(image.height) ||
           rect.minX < 0 || rect.minY < 0 {
            throw PixelUtilsError.invalidROI("ROI is out of image bounds")
        }
        
        guard let cropped = image.cropping(to: rect) else {
            throw PixelUtilsError.processingFailed("Failed to crop image")
        }
        
        return cropped
    }
    
    private static func applyResize(to image: CGImage, options: ResizeOptions, originalSize: CGSize) throws -> ResizeResult {
        let targetSize = CGSize(width: options.width, height: options.height)
        
        switch options.strategy {
        case .cover:
            let resized = try resizeCover(image: image, targetSize: targetSize)
            return ResizeResult(image: resized, letterboxInfo: nil)
        case .contain:
            let resized = try resizeContain(image: image, targetSize: targetSize, padColor: options.padColor)
            return ResizeResult(image: resized, letterboxInfo: nil)
        case .stretch:
            let resized = try resizeStretch(image: image, targetSize: targetSize)
            return ResizeResult(image: resized, letterboxInfo: nil)
        case .letterbox:
            return try resizeLetterboxWithInfo(image: image, targetSize: targetSize, fillColor: options.letterboxColor, originalSize: originalSize)
        }
    }
    
    private static func resizeCover(image: CGImage, targetSize: CGSize) throws -> CGImage {
        let sourceSize = CGSize(width: image.width, height: image.height)
        let scale = max(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
        let scaledSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        
        let x = (scaledSize.width - targetSize.width) / 2
        let y = (scaledSize.height - targetSize.height) / 2
        
        let context = try createContext(size: targetSize)
        context.draw(image, in: CGRect(x: -x, y: -y, width: scaledSize.width, height: scaledSize.height))
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create resized image")
        }
        return result
    }
    
    private static func resizeContain(image: CGImage, targetSize: CGSize, padColor: [Float]?) throws -> CGImage {
        let sourceSize = CGSize(width: image.width, height: image.height)
        let scale = min(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
        let scaledSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        
        let x = (targetSize.width - scaledSize.width) / 2
        let y = (targetSize.height - scaledSize.height) / 2
        
        let context = try createContext(size: targetSize)
        
        // Fill with pad color if specified
        if let padColor = padColor, padColor.count >= 3 {
            context.setFillColor(red: CGFloat(padColor[0]) / 255.0,
                               green: CGFloat(padColor[1]) / 255.0,
                               blue: CGFloat(padColor[2]) / 255.0,
                               alpha: 1.0)
            context.fill(CGRect(origin: .zero, size: targetSize))
        }
        
        context.draw(image, in: CGRect(x: x, y: y, width: scaledSize.width, height: scaledSize.height))
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create resized image")
        }
        return result
    }
    
    private static func resizeStretch(image: CGImage, targetSize: CGSize) throws -> CGImage {
        let context = try createContext(size: targetSize)
        context.draw(image, in: CGRect(origin: .zero, size: targetSize))
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create resized image")
        }
        return result
    }
    
    private static func resizeLetterbox(image: CGImage, targetSize: CGSize, fillColor: [Float]?) throws -> CGImage {
        let sourceSize = CGSize(width: image.width, height: image.height)
        let scale = min(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
        let scaledSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        
        let x = (targetSize.width - scaledSize.width) / 2
        let y = (targetSize.height - scaledSize.height) / 2
        
        let context = try createContext(size: targetSize)
        
        // Fill with letterbox color (default YOLO gray)
        let color = fillColor ?? [114, 114, 114]
        context.setFillColor(red: CGFloat(color[0]) / 255.0,
                           green: CGFloat(color[1]) / 255.0,
                           blue: CGFloat(color[2]) / 255.0,
                           alpha: 1.0)
        context.fill(CGRect(origin: .zero, size: targetSize))
        
        context.draw(image, in: CGRect(x: x, y: y, width: scaledSize.width, height: scaledSize.height))
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create letterboxed image")
        }
        return result
    }
    
    /// Letterbox resize that also returns transform info for reverse coordinate mapping
    private static func resizeLetterboxWithInfo(image: CGImage, targetSize: CGSize, fillColor: [Float]?, originalSize: CGSize) throws -> ResizeResult {
        let sourceSize = CGSize(width: image.width, height: image.height)
        let scale = min(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
        let scaledSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        
        let x = (targetSize.width - scaledSize.width) / 2
        let y = (targetSize.height - scaledSize.height) / 2
        
        let context = try createContext(size: targetSize)
        
        // Fill with letterbox color (default YOLO gray)
        let color = fillColor ?? [114, 114, 114]
        context.setFillColor(red: CGFloat(color[0]) / 255.0,
                           green: CGFloat(color[1]) / 255.0,
                           blue: CGFloat(color[2]) / 255.0,
                           alpha: 1.0)
        context.fill(CGRect(origin: .zero, size: targetSize))
        
        context.draw(image, in: CGRect(x: x, y: y, width: scaledSize.width, height: scaledSize.height))
        
        guard let result = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create letterboxed image")
        }
        
        let letterboxInfo = LetterboxInfo(
            scale: Float(scale),
            offset: CGPoint(x: x, y: y),
            originalSize: originalSize,
            letterboxedSize: targetSize
        )
        
        return ResizeResult(image: result, letterboxInfo: letterboxInfo)
    }
    
    private static func createContext(size: CGSize) throws -> CGContext {
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: Int(size.width) * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        return context
    }
    
    // MARK: - Pixel Extraction
    
    private static func extractRGBAPixels(from image: CGImage) throws -> [UInt8] {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create context for pixel extraction")
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return pixelData
    }
    
    // MARK: - Color Conversion
    
    private static func convertColorFormat(
        _ rgbaData: [UInt8],
        width: Int,
        height: Int,
        targetFormat: ColorFormat
    ) throws -> [Float] {
        let pixelCount = width * height
        
        switch targetFormat {
        case .rgb:
            return convertRGBAtoRGB(rgbaData, pixelCount: pixelCount)
        case .rgba:
            return rgbaData.map { Float($0) / 255.0 }
        case .bgr:
            return convertRGBAtoBGR(rgbaData, pixelCount: pixelCount)
        case .bgra:
            return convertRGBAtoBGRA(rgbaData, pixelCount: pixelCount)
        case .grayscale:
            return convertRGBAtoGrayscale(rgbaData, pixelCount: pixelCount)
        case .hsv:
            return convertRGBAtoHSV(rgbaData, pixelCount: pixelCount)
        case .hsl:
            return convertRGBAtoHSL(rgbaData, pixelCount: pixelCount)
        case .lab:
            return convertRGBAtoLAB(rgbaData, pixelCount: pixelCount)
        case .yuv:
            return convertRGBAtoYUV(rgbaData, pixelCount: pixelCount)
        case .ycbcr:
            return convertRGBAtoYCbCr(rgbaData, pixelCount: pixelCount)
        }
    }
    
    private static func convertRGBAtoRGB(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var rgb = [Float](repeating: 0, count: pixelCount * 3)
        
        for i in 0..<pixelCount {
            let rgbaIndex = i * 4
            let rgbIndex = i * 3
            rgb[rgbIndex] = Float(rgba[rgbaIndex]) / 255.0
            rgb[rgbIndex + 1] = Float(rgba[rgbaIndex + 1]) / 255.0
            rgb[rgbIndex + 2] = Float(rgba[rgbaIndex + 2]) / 255.0
        }
        
        return rgb
    }
    
    private static func convertRGBAtoBGR(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var bgr = [Float](repeating: 0, count: pixelCount * 3)
        
        for i in 0..<pixelCount {
            let rgbaIndex = i * 4
            let bgrIndex = i * 3
            bgr[bgrIndex] = Float(rgba[rgbaIndex + 2]) / 255.0
            bgr[bgrIndex + 1] = Float(rgba[rgbaIndex + 1]) / 255.0
            bgr[bgrIndex + 2] = Float(rgba[rgbaIndex]) / 255.0
        }
        
        return bgr
    }
    
    private static func convertRGBAtoBGRA(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var bgra = [Float](repeating: 0, count: pixelCount * 4)
        
        for i in 0..<pixelCount {
            let index = i * 4
            bgra[index] = Float(rgba[index + 2]) / 255.0
            bgra[index + 1] = Float(rgba[index + 1]) / 255.0
            bgra[index + 2] = Float(rgba[index]) / 255.0
            bgra[index + 3] = Float(rgba[index + 3]) / 255.0
        }
        
        return bgra
    }
    
    /// Converts RGBA pixel data to grayscale using the ITU-R BT.601 luma coefficients.
    ///
    /// ## Mathematical Basis
    ///
    /// This conversion uses the **luminosity method** (also called luma), which weights RGB
    /// channels according to human perception of brightness:
    ///
    /// ```
    /// Y = 0.299R + 0.587G + 0.114B
    /// ```
    ///
    /// ### Coefficient Origins
    ///
    /// These coefficients come from the **ITU-R Recommendation BT.601** standard for
    /// standard-definition television (SDTV). They are derived from:
    ///
    /// - **0.299 (Red)**: Red contributes ~30% to perceived brightness
    /// - **0.587 (Green)**: Green contributes ~59% to perceived brightness (highest sensitivity)
    /// - **0.114 (Blue)**: Blue contributes ~11% to perceived brightness (lowest sensitivity)
    ///
    /// The human eye has more green-sensitive cone cells than red or blue, hence green's
    /// dominant weighting. These coefficients sum to 1.0, preserving overall brightness.
    ///
    /// ### Alternative Standards
    ///
    /// - **BT.709** (HDTV): `Y = 0.2126R + 0.7152G + 0.0722B`
    /// - **BT.2020** (UHDTV): `Y = 0.2627R + 0.6780G + 0.0593B`
    ///
    /// BT.601 is used here for broad compatibility with most ML models and image processing.
    ///
    /// - Parameters:
    ///   - rgba: Raw RGBA pixel data as UInt8 values (0-255)
    ///   - pixelCount: Total number of pixels (width × height)
    /// - Returns: Grayscale values normalized to [0, 1] range
    private static func convertRGBAtoGrayscale(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var gray = [Float](repeating: 0, count: pixelCount)
        
        for i in 0..<pixelCount {
            let index = i * 4
            let r = Float(rgba[index])
            let g = Float(rgba[index + 1])
            let b = Float(rgba[index + 2])
            // ITU-R BT.601 luma coefficients for perceptual grayscale
            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        
        return gray
    }
    
    /// Converts RGBA pixel data to HSV (Hue, Saturation, Value) color space.
    ///
    /// ## HSV Color Model
    ///
    /// HSV is a cylindrical color model that represents colors in terms more intuitive
    /// to human perception than RGB:
    ///
    /// - **Hue (H)**: The color type, represented as an angle (0-360°)
    ///   - 0°/360° = Red
    ///   - 60° = Yellow
    ///   - 120° = Green
    ///   - 180° = Cyan
    ///   - 240° = Blue
    ///   - 300° = Magenta
    ///
    /// - **Saturation (S)**: Color purity/intensity (0-1)
    ///   - 0 = Gray (no color)
    ///   - 1 = Pure, vivid color
    ///
    /// - **Value (V)**: Brightness (0-1)
    ///   - 0 = Black
    ///   - 1 = Maximum brightness for that hue/saturation
    ///
    /// ## Conversion Formulas
    ///
    /// Given RGB values normalized to [0, 1]:
    ///
    /// ```
    /// V = max(R, G, B)
    /// S = (V == 0) ? 0 : (V - min(R, G, B)) / V
    ///
    /// H = 0                                  if delta == 0
    ///   = 60° × ((G - B) / delta mod 6)      if V == R
    ///   = 60° × ((B - R) / delta + 2)        if V == G
    ///   = 60° × ((R - G) / delta + 4)        if V == B
    /// ```
    ///
    /// ## Output Format
    ///
    /// All three channels are normalized to [0, 1]:
    /// - H: Divided by 360 to map angle to [0, 1]
    /// - S: Already in [0, 1]
    /// - V: Already in [0, 1]
    ///
    /// - Parameters:
    ///   - rgba: Raw RGBA pixel data as UInt8 values (0-255)
    ///   - pixelCount: Total number of pixels (width × height)
    /// - Returns: HSV values with all channels normalized to [0, 1]
    private static func convertRGBAtoHSV(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var hsv = [Float](repeating: 0, count: pixelCount * 3)
        
        for i in 0..<pixelCount {
            let rgbaIndex = i * 4
            let hsvIndex = i * 3
            
            let r = Float(rgba[rgbaIndex]) / 255.0
            let g = Float(rgba[rgbaIndex + 1]) / 255.0
            let b = Float(rgba[rgbaIndex + 2]) / 255.0
            
            let maxVal = max(r, g, b)  // V component
            let minVal = min(r, g, b)
            let delta = maxVal - minVal  // Chroma
            
            // Hue calculation - determines which 60° sector of the color wheel
            var h: Float = 0
            if delta > 0 {
                if maxVal == r {
                    // Red is max: hue is between yellow and magenta
                    h = 60 * ((g - b) / delta).truncatingRemainder(dividingBy: 6)
                } else if maxVal == g {
                    // Green is max: hue is between cyan and yellow
                    h = 60 * ((b - r) / delta + 2)
                } else {
                    // Blue is max: hue is between magenta and cyan
                    h = 60 * ((r - g) / delta + 4)
                }
            }
            if h < 0 { h += 360 }  // Ensure positive hue angle
            
            // Saturation: ratio of chroma to value (how "pure" the color is)
            let s = maxVal == 0 ? 0 : delta / maxVal
            
            // Value: the maximum RGB component (brightness)
            let v = maxVal
            
            // Normalize all to [0, 1] for ML compatibility
            hsv[hsvIndex] = h / 360.0
            hsv[hsvIndex + 1] = s
            hsv[hsvIndex + 2] = v
        }
        
        return hsv
    }
    
    /// Converts RGBA pixel data to HSL (Hue, Saturation, Lightness) color space.
    ///
    /// ## HSL vs HSV
    ///
    /// HSL and HSV are both cylindrical representations, but differ in how they model brightness:
    ///
    /// | Property | HSV | HSL |
    /// |----------|-----|-----|
    /// | Brightness | Value (V) | Lightness (L) |
    /// | White | High V, low S | L = 1 |
    /// | Black | V = 0 | L = 0 |
    /// | Pure color | V = 1, S = 1 | L = 0.5, S = 1 |
    ///
    /// HSL is often preferred in design tools because L = 0.5 represents "pure" colors,
    /// while in HSV, pure colors have varying perceived brightness.
    ///
    /// ## Conversion Formulas
    ///
    /// Given RGB values normalized to [0, 1]:
    ///
    /// ```
    /// L = (max(R,G,B) + min(R,G,B)) / 2
    ///
    /// S = 0                          if delta == 0 (gray)
    ///   = delta / (1 - |2L - 1|)     otherwise
    ///
    /// H = (same as HSV calculation)
    /// ```
    ///
    /// ### Saturation Formula Explanation
    ///
    /// The divisor `(1 - |2L - 1|)` creates a "double cone" shape:
    /// - At L = 0.5, divisor = 1 (maximum possible saturation)
    /// - At L = 0 or L = 1, divisor approaches 0 (colors become gray)
    ///
    /// This ensures saturation represents the "colorfulness" relative to what's
    /// possible at that lightness level.
    ///
    /// - Parameters:
    ///   - rgba: Raw RGBA pixel data as UInt8 values (0-255)
    ///   - pixelCount: Total number of pixels (width × height)
    /// - Returns: HSL values with all channels normalized to [0, 1]
    private static func convertRGBAtoHSL(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var hsl = [Float](repeating: 0, count: pixelCount * 3)
        
        for i in 0..<pixelCount {
            let rgbaIndex = i * 4
            let hslIndex = i * 3
            
            let r = Float(rgba[rgbaIndex]) / 255.0
            let g = Float(rgba[rgbaIndex + 1]) / 255.0
            let b = Float(rgba[rgbaIndex + 2]) / 255.0
            
            let maxVal = max(r, g, b)
            let minVal = min(r, g, b)
            let delta = maxVal - minVal  // Chroma
            
            // Lightness: average of max and min (midpoint of color range)
            let l = (maxVal + minVal) / 2
            
            // Saturation: chroma relative to lightness
            // The formula creates a double-cone where S=0 at L=0 and L=1
            let s = delta == 0 ? 0 : delta / (1 - abs(2 * l - 1))
            
            // Hue calculation (identical to HSV)
            var h: Float = 0
            if delta > 0 {
                if maxVal == r {
                    h = 60 * ((g - b) / delta).truncatingRemainder(dividingBy: 6)
                } else if maxVal == g {
                    h = 60 * ((b - r) / delta + 2)
                } else {
                    h = 60 * ((r - g) / delta + 4)
                }
            }
            if h < 0 { h += 360 }
            
            hsl[hslIndex] = h / 360.0
            hsl[hslIndex + 1] = s
            hsl[hslIndex + 2] = l
        }
        
        return hsl
    }
    
    /// Converts RGBA pixel data to a simplified LAB-like color space.
    ///
    /// ## Important Note
    ///
    /// This is a **simplified approximation** of the CIE LAB color space, not a full
    /// implementation. True LAB conversion requires an intermediate XYZ transform and
    /// uses non-linear functions. This simplified version is computationally efficient
    /// and provides perceptually-informed color separation suitable for many ML tasks.
    ///
    /// ## True CIE LAB Color Space
    ///
    /// The full CIE LAB (L*a*b*) is defined as:
    ///
    /// ```
    /// RGB → XYZ → LAB
    ///
    /// XYZ Conversion (sRGB D65 illuminant):
    /// X = 0.4124564R + 0.3575761G + 0.1804375B
    /// Y = 0.2126729R + 0.7151522G + 0.0721750B
    /// Z = 0.0193339R + 0.1191920G + 0.9503041B
    ///
    /// LAB Conversion (D65 reference white: Xn=95.047, Yn=100, Zn=108.883):
    /// L* = 116 × f(Y/Yn) - 16
    /// a* = 500 × (f(X/Xn) - f(Y/Yn))
    /// b* = 200 × (f(Y/Yn) - f(Z/Zn))
    ///
    /// where f(t) = t^(1/3)           if t > (6/29)³
    ///            = (1/3)(29/6)²t + 4/29  otherwise
    /// ```
    ///
    /// ## This Implementation
    ///
    /// Uses a computationally efficient approximation:
    ///
    /// ```
    /// L = 0.299R + 0.587G + 0.114B           (BT.601 luminance)
    /// A = 0.5 × (R - L)                       (red-green opponent)
    /// B = 0.5 × (B - L)                       (blue-yellow opponent)
    /// ```
    ///
    /// This preserves the key property of LAB: separating luminance from chrominance
    /// in a perceptually-meaningful way, while being 10-100× faster than true LAB.
    ///
    /// ## When to Use True LAB
    ///
    /// Use a full CIE LAB implementation (via Core Image or custom) when:
    /// - Color accuracy is critical (printing, color matching)
    /// - Computing perceptual color differences (Delta E)
    /// - Working with color science applications
    ///
    /// This simplified version is suitable for:
    /// - ML preprocessing where approximate color separation suffices
    /// - Real-time applications requiring speed
    /// - Feature extraction where exact colorimetry isn't needed
    ///
    /// - Parameters:
    ///   - rgba: Raw RGBA pixel data as UInt8 values (0-255)
    ///   - pixelCount: Total number of pixels (width × height)
    /// - Returns: Simplified LAB-like values normalized to [0, 1] range
    private static func convertRGBAtoLAB(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var lab = [Float](repeating: 0, count: pixelCount * 3)
        
        for i in 0..<pixelCount {
            let rgbaIndex = i * 4
            let labIndex = i * 3
            
            let r = Float(rgba[rgbaIndex]) / 255.0
            let g = Float(rgba[rgbaIndex + 1]) / 255.0
            let b = Float(rgba[rgbaIndex + 2]) / 255.0
            
            // Luminance using BT.601 coefficients
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b
            
            // Simplified opponent color channels
            // A-like channel: red vs luminance (approximates red-green)
            // B-like channel: blue vs luminance (approximates blue-yellow)
            lab[labIndex] = luminance
            lab[labIndex + 1] = 0.5 * (r - luminance)  // Shifted to ~[0, 0.5]
            lab[labIndex + 2] = 0.5 * (b - luminance)  // Shifted to ~[0, 0.5]
        }
        
        return lab
    }
    
    /// Converts RGBA pixel data to YUV color space using ITU-R BT.601 coefficients.
    ///
    /// ## YUV Color Model
    ///
    /// YUV separates image luminance (Y) from chrominance (U, V), which was historically
    /// important for analog television transmission compatibility with black-and-white TVs.
    /// Today, it's widely used in video compression (MPEG, H.264) and image processing.
    ///
    /// - **Y (Luma)**: Brightness information, identical to grayscale
    /// - **U (Cb)**: Blue-difference chroma component (B - Y, scaled)
    /// - **V (Cr)**: Red-difference chroma component (R - Y, scaled)
    ///
    /// ## ITU-R BT.601 Conversion Matrix
    ///
    /// The coefficients implement the standard BT.601 transform:
    ///
    /// ```
    /// | Y |   |  0.299     0.587     0.114   |   | R |
    /// | U | = | -0.14713  -0.28886   0.436   | × | G |
    /// | V |   |  0.615    -0.51499  -0.10001 |   | B |
    /// ```
    ///
    /// ### Coefficient Derivation
    ///
    /// The U and V coefficients are derived from the luma coefficients:
    ///
    /// ```
    /// Y = 0.299R + 0.587G + 0.114B
    ///
    /// U = 0.492 × (B - Y)
    ///   = 0.492 × (B - 0.299R - 0.587G - 0.114B)
    ///   = -0.14713R - 0.28886G + 0.436B
    ///
    /// V = 0.877 × (R - Y)
    ///   = 0.877 × (R - 0.299R - 0.587G - 0.114B)
    ///   = 0.615R - 0.51499G - 0.10001B
    /// ```
    ///
    /// The scaling factors (0.492, 0.877) normalize U and V to prevent overflow
    /// while maximizing dynamic range for analog transmission.
    ///
    /// ## Output Range
    ///
    /// - **Y**: [0, 1]
    /// - **U**: approximately [-0.436, 0.436]
    /// - **V**: approximately [-0.615, 0.615]
    ///
    /// Note: U and V can be negative. For ML models expecting [0, 1], you may need
    /// additional normalization.
    ///
    /// - Parameters:
    ///   - rgba: Raw RGBA pixel data as UInt8 values (0-255)
    ///   - pixelCount: Total number of pixels (width × height)
    /// - Returns: YUV values (Y in [0,1], U and V may be negative)
    private static func convertRGBAtoYUV(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var yuv = [Float](repeating: 0, count: pixelCount * 3)
        
        for i in 0..<pixelCount {
            let rgbaIndex = i * 4
            let yuvIndex = i * 3
            
            let r = Float(rgba[rgbaIndex]) / 255.0
            let g = Float(rgba[rgbaIndex + 1]) / 255.0
            let b = Float(rgba[rgbaIndex + 2]) / 255.0
            
            // Y: Luma (same as grayscale, BT.601)
            yuv[yuvIndex] = 0.299 * r + 0.587 * g + 0.114 * b
            // U: Scaled blue-difference (Cb-like)
            yuv[yuvIndex + 1] = -0.14713 * r - 0.28886 * g + 0.436 * b
            // V: Scaled red-difference (Cr-like)
            yuv[yuvIndex + 2] = 0.615 * r - 0.51499 * g - 0.10001 * b
        }
        
        return yuv
    }
    
    /// Converts RGBA pixel data to YCbCr color space (ITU-R BT.601 digital).
    ///
    /// ## YCbCr vs YUV
    ///
    /// YCbCr is the **digital** version of YUV, with values offset and scaled for
    /// 8-bit digital representation. The key differences:
    ///
    /// | Property | YUV | YCbCr |
    /// |----------|-----|-------|
    /// | Domain | Analog (-0.5 to 0.5) | Digital (0-255) |
    /// | Chroma offset | None (centered at 0) | 128 (centered at mid-gray) |
    /// | Usage | Analog TV | JPEG, MPEG, H.264 |
    ///
    /// ## ITU-R BT.601 Digital Conversion
    ///
    /// The coefficients implement the standard "studio swing" or "limited range":
    ///
    /// ```
    /// Y  =  0.299R + 0.587G + 0.114B
    /// Cb = 128 - 0.168736R - 0.331264G + 0.500000B
    /// Cr = 128 + 0.500000R - 0.418688G - 0.081312B
    /// ```
    ///
    /// ### Coefficient Derivation
    ///
    /// The Cb/Cr coefficients come from normalizing the U/V components:
    ///
    /// ```
    /// Cb = 128 + (1/2) × (B - Y) / (1 - 0.114)
    ///    = 128 + (B - Y) / 1.772
    ///
    /// Cr = 128 + (1/2) × (R - Y) / (1 - 0.299)
    ///    = 128 + (R - Y) / 1.402
    /// ```
    ///
    /// The divisors (1.772, 1.402) normalize chroma so that:
    /// - Pure blue (0, 0, 255) → Cb = 255
    /// - Pure red (255, 0, 0) → Cr = 255
    ///
    /// ### The "128" Offset
    ///
    /// The offset of 128 shifts chroma from signed to unsigned:
    /// - Cb, Cr = 128 represents "no color" (gray)
    /// - Values < 128 are negative chroma
    /// - Values > 128 are positive chroma
    ///
    /// This allows storage in unsigned 8-bit values (0-255).
    ///
    /// ## Output Range
    ///
    /// All channels normalized to [0, 1]:
    /// - **Y**: [0, 1] (black to white)
    /// - **Cb**: [0, 1] (blue contribution, 0.5 = neutral)
    /// - **Cr**: [0, 1] (red contribution, 0.5 = neutral)
    ///
    /// ## Common Usage
    ///
    /// YCbCr is the standard color space for:
    /// - JPEG compression
    /// - MPEG video codecs
    /// - H.264/H.265 video
    /// - Most camera sensors (via ISP)
    ///
    /// - Parameters:
    ///   - rgba: Raw RGBA pixel data as UInt8 values (0-255)
    ///   - pixelCount: Total number of pixels (width × height)
    /// - Returns: YCbCr values normalized to [0, 1]
    private static func convertRGBAtoYCbCr(_ rgba: [UInt8], pixelCount: Int) -> [Float] {
        var ycbcr = [Float](repeating: 0, count: pixelCount * 3)
        
        for i in 0..<pixelCount {
            let rgbaIndex = i * 4
            let ycbcrIndex = i * 3
            
            // Work in 0-255 range for coefficient accuracy
            let r = Float(rgba[rgbaIndex])
            let g = Float(rgba[rgbaIndex + 1])
            let b = Float(rgba[rgbaIndex + 2])
            
            // Y: Luma (BT.601 coefficients)
            ycbcr[ycbcrIndex] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            // Cb: Blue-difference chroma with 128 offset
            ycbcr[ycbcrIndex + 1] = (128 - 0.168736 * r - 0.331264 * g + 0.5 * b) / 255.0
            // Cr: Red-difference chroma with 128 offset
            ycbcr[ycbcrIndex + 2] = (128 + 0.5 * r - 0.418688 * g - 0.081312 * b) / 255.0
        }
        
        return ycbcr
    }
    
    // MARK: - Normalization
    
    /// Applies normalization to pixel data using the specified preset or custom values.
    ///
    /// ## Normalization Presets
    ///
    /// ### Scale (Default)
    /// Maps raw 0-255 values to [0, 1]:
    /// ```
    /// normalized = value / 255.0
    /// ```
    /// Used by most simple models and for general image processing.
    ///
    /// ### ImageNet
    /// Applies z-score normalization with ImageNet dataset statistics:
    /// ```
    /// normalized = (value - mean) / std
    ///
    /// Mean: [0.485, 0.456, 0.406] (R, G, B)
    /// Std:  [0.229, 0.224, 0.225] (R, G, B)
    /// ```
    ///
    /// #### Why These Specific Values?
    ///
    /// The ImageNet statistics were computed from ~1.2 million images in the
    /// ILSVRC 2012 dataset. They represent the average pixel values and standard
    /// deviations across all training images:
    ///
    /// - **Mean [0.485, 0.456, 0.406]**: Images tend to be slightly red-biased
    ///   (outdoor scenes, skin tones). Red > Green > Blue.
    ///
    /// - **Std [0.229, 0.224, 0.225]**: Relatively uniform variance across channels,
    ///   with red slightly higher (more variation in red channel).
    ///
    /// #### Mathematical Justification
    ///
    /// Z-score normalization (standardization) transforms data to have zero mean
    /// and unit variance:
    /// ```
    /// z = (x - μ) / σ
    /// ```
    ///
    /// This is crucial for neural networks because:
    /// 1. **Faster convergence**: Gradients flow better through normalized activations
    /// 2. **Numerical stability**: Prevents exploding/vanishing gradients
    /// 3. **Weight initialization**: Most init schemes assume zero-mean inputs
    ///
    /// ### TensorFlow
    /// Maps [0, 1] to [-1, 1]:
    /// ```
    /// normalized = value * 2.0 - 1.0
    /// ```
    /// Used by MobileNet, EfficientNet, and many TensorFlow models.
    ///
    /// ### Raw
    /// Returns values in original [0, 255] range.
    /// Used when models expect unnormalized integer inputs.
    ///
    /// - Parameters:
    ///   - data: Pixel data already scaled to [0, 1]
    ///   - normalization: Normalization configuration
    ///   - channels: Number of color channels
    /// - Returns: Normalized pixel data
    /// - Throws: ``PixelUtilsError/invalidOptions(_:)`` if custom normalization missing mean/std
    private static func applyNormalization(
        _ data: [Float],
        normalization: Normalization,
        channels: Int
    ) throws -> [Float] {
        switch normalization.preset {
        case .scale:
            return data // Already normalized to [0, 1]
            
        case .imagenet:
            let mean = normalization.mean ?? [0.485, 0.456, 0.406]
            let std = normalization.std ?? [0.229, 0.224, 0.225]
            return normalizeWithMeanStd(data, mean: mean, std: std, channels: channels)
            
        case .tensorflow:
            return data.map { $0 * 2.0 - 1.0 } // [0, 1] -> [-1, 1]
            
        case .raw:
            return data.map { $0 * 255.0 } // [0, 1] -> [0, 255]
            
        case .custom:
            guard let mean = normalization.mean, let std = normalization.std else {
                throw PixelUtilsError.invalidOptions("Custom normalization requires mean and std")
            }
            return normalizeWithMeanStd(data, mean: mean, std: std, channels: channels)
        }
    }
    
    /// Applies per-channel mean subtraction and standard deviation division (z-score normalization).
    ///
    /// ## Z-Score Normalization
    ///
    /// For each pixel and channel:
    /// ```
    /// output[c] = (input[c] - mean[c]) / std[c]
    /// ```
    ///
    /// ## Why Per-Channel?
    ///
    /// Different color channels have different statistical properties:
    /// - Natural images tend to have more red (sky, skin, earth tones)
    /// - Standard deviation varies by channel due to color distributions
    ///
    /// Per-channel normalization ensures each channel contributes equally to the loss
    /// function during training, preventing color bias in learned features.
    ///
    /// ## Common Mean/Std Values
    ///
    /// | Dataset/Model | Mean (R, G, B) | Std (R, G, B) |
    /// |--------------|----------------|---------------|
    /// | ImageNet | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
    /// | CLIP | [0.481, 0.458, 0.408] | [0.269, 0.261, 0.276] |
    /// | CIFAR-10 | [0.491, 0.482, 0.447] | [0.247, 0.243, 0.262] |
    ///
    /// ## Channel Fallback
    ///
    /// If mean/std arrays have fewer elements than channels (e.g., grayscale input
    /// with RGB statistics), the last value is repeated. This allows ImageNet stats
    /// to work with single-channel inputs.
    ///
    /// - Parameters:
    ///   - data: Input pixel data in [0, 1] range
    ///   - mean: Per-channel mean values to subtract
    ///   - std: Per-channel standard deviation values to divide by
    ///   - channels: Number of color channels in the data
    /// - Returns: Z-score normalized data (approximately [-2, 2] for most pixels)
    private static func normalizeWithMeanStd(_ data: [Float], mean: [Float], std: [Float], channels: Int) -> [Float] {
        var normalized = [Float](repeating: 0, count: data.count)
        let pixelCount = data.count / channels
        
        for i in 0..<pixelCount {
            for c in 0..<channels {
                let index = i * channels + c
                let channelMean = c < mean.count ? mean[c] : mean.last ?? 0
                let channelStd = c < std.count ? std[c] : std.last ?? 1
                normalized[index] = (data[index] - channelMean) / channelStd
            }
        }
        
        return normalized
    }
    
    // MARK: - Data Layout
    
    /// Transforms pixel data between different memory layout formats.
    ///
    /// ## Memory Layout Formats
    ///
    /// ### HWC (Height × Width × Channels)
    /// ```
    /// Memory: [R, G, B, R, G, B, ...] (interleaved)
    /// Index:  pixel * channels + channel
    /// ```
    /// - **Used by**: TensorFlow, CoreML, most image libraries
    /// - **Advantage**: Cache-friendly for per-pixel operations
    /// - **Shape**: [Height, Width, Channels]
    ///
    /// ### CHW (Channels × Height × Width)
    /// ```
    /// Memory: [R, R, R, ..., G, G, G, ..., B, B, B, ...] (planar)
    /// Index:  channel * (height * width) + row * width + col
    /// ```
    /// - **Used by**: PyTorch, ONNX, most research frameworks
    /// - **Advantage**: SIMD-friendly for channel-wise operations, better for convolutions
    /// - **Shape**: [Channels, Height, Width]
    ///
    /// ### NHWC (Batch × Height × Width × Channels)
    /// - HWC with batch dimension prepended
    /// - **Used by**: TensorFlow batch inference
    /// - **Shape**: [1, Height, Width, Channels]
    ///
    /// ### NCHW (Batch × Channels × Height × Width)
    /// - CHW with batch dimension prepended
    /// - **Used by**: PyTorch batch inference, ONNX Runtime
    /// - **Shape**: [1, Channels, Height, Width]
    ///
    /// ## Why CHW for Deep Learning?
    ///
    /// CHW layout is preferred by many frameworks because:
    ///
    /// 1. **Convolution efficiency**: Filters slide over spatial dimensions, so having
    ///    channels contiguous allows SIMD vectorization of channel operations.
    ///
    /// 2. **GPU memory coalescing**: When threads process adjacent pixels, CHW ensures
    ///    memory accesses are coalesced (sequential addresses).
    ///
    /// 3. **Batch normalization**: Channel-wise stats are computed over spatial dims,
    ///    making CHW more efficient for reading entire channels.
    ///
    /// - Parameters:
    ///   - data: Input pixel data in HWC format
    ///   - width: Image width in pixels
    ///   - height: Image height in pixels
    ///   - channels: Number of color channels
    ///   - layout: Target memory layout format
    /// - Returns: Pixel data rearranged to the target layout
    private static func applyDataLayout(
        _ data: [Float],
        width: Int,
        height: Int,
        channels: Int,
        layout: DataLayout
    ) throws -> [Float] {
        switch layout {
        case .hwc:
            return data // Already in HWC format
            
        case .chw:
            return convertHWCtoCHW(data, width: width, height: height, channels: channels)
            
        case .nhwc:
            // Add batch dimension (N=1)
            return data
            
        case .nchw:
            // Add batch dimension (N=1) and convert to CHW
            return convertHWCtoCHW(data, width: width, height: height, channels: channels)
        }
    }
    
    /// Converts pixel data from HWC (interleaved) to CHW (planar) memory layout.
    ///
    /// ## Index Transformation
    ///
    /// This function remaps pixel data from interleaved to planar format:
    ///
    /// ```
    /// HWC Index: row * width * channels + col * channels + channel
    ///          = (row * width + col) * channels + channel
    ///
    /// CHW Index: channel * (height * width) + row * width + col
    ///          = channel * pixelCount + row * width + col
    /// ```
    ///
    /// ## Example
    ///
    /// For a 2×2 RGB image:
    /// ```
    /// HWC: [R00, G00, B00, R01, G01, B01, R10, G10, B10, R11, G11, B11]
    ///       pixel(0,0)     pixel(0,1)     pixel(1,0)     pixel(1,1)
    ///
    /// CHW: [R00, R01, R10, R11, G00, G01, G10, G11, B00, B01, B10, B11]
    ///       ---- Red ----  ---- Green --  ---- Blue ---
    /// ```
    ///
    /// ## Memory Access Patterns
    ///
    /// The triple nested loop (channels → rows → cols) ensures sequential writes
    /// to the output array, which is cache-friendly. Reads from the input are
    /// strided but this is typically less impactful than strided writes.
    ///
    /// - Parameters:
    ///   - data: Input pixel data in HWC format [H × W × C elements]
    ///   - width: Image width in pixels
    ///   - height: Image height in pixels
    ///   - channels: Number of color channels
    /// - Returns: Pixel data in CHW format [C × H × W elements]
    private static func convertHWCtoCHW(_ data: [Float], width: Int, height: Int, channels: Int) -> [Float] {
        var chw = [Float](repeating: 0, count: data.count)
        let pixelCount = width * height
        
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let hwcIndex = h * width * channels + w * channels + c
                    let chwIndex = c * pixelCount + h * width + w
                    chw[chwIndex] = data[hwcIndex]
                }
            }
        }
        
        return chw
    }
    
    // MARK: - Helper Functions
    
    private static func calculateShape(width: Int, height: Int, channels: Int, layout: DataLayout) -> [Int] {
        switch layout {
        case .hwc:
            return [height, width, channels]
        case .chw:
            return [channels, height, width]
        case .nhwc:
            return [1, height, width, channels]
        case .nchw:
            return [1, channels, height, width]
        }
    }
    
    /// Convert Float32 array to Float16 (stored as UInt16 bit patterns).
    /// Uses Accelerate framework for efficient SIMD conversion on Apple Silicon.
    private static func convertToFloat16(_ floatData: [Float]) -> [UInt16] {
        var float16Data = [UInt16](repeating: 0, count: floatData.count)
        
        // Use Accelerate for fast SIMD conversion
        floatData.withUnsafeBufferPointer { srcBuffer in
            float16Data.withUnsafeMutableBufferPointer { dstBuffer in
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcBuffer.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(floatData.count),
                    rowBytes: floatData.count * MemoryLayout<Float>.size
                )
                var dst = vImage_Buffer(
                    data: dstBuffer.baseAddress!,
                    height: 1,
                    width: vImagePixelCount(floatData.count),
                    rowBytes: floatData.count * MemoryLayout<UInt16>.size
                )
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
            }
        }
        
        return float16Data
    }
}

// MARK: - Array Extension

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}
