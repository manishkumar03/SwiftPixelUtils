//
//  MultiCropOperations.swift
//  SwiftPixelUtils
//
//  Multi-crop operations for test-time augmentation
//

import Foundation
import CoreGraphics

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Types

/// Position of a crop in the original image.
public struct CropPosition {
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int
    public let isFlipped: Bool
    
    public init(x: Int, y: Int, width: Int, height: Int, isFlipped: Bool = false) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.isFlipped = isFlipped
    }
}

/// Result of a multi-crop operation.
public struct MultiCropResult {
    /// Array of pixel data results for each crop.
    public let crops: [PixelDataResult]
    
    /// Positions of each crop in the original image.
    public let positions: [CropPosition]
    
    /// Processing time in milliseconds.
    public let processingTimeMs: Double
}

/// Options for five/ten crop operations.
public struct CropOptions {
    /// Width of each crop.
    public let width: Int
    
    /// Height of each crop.
    public let height: Int
    
    /// Normalization to apply.
    public var normalization: Normalization = .imagenet
    
    /// Data layout for output.
    public var dataLayout: DataLayout = .hwc
    
    /// Color format for output.
    public var colorFormat: ColorFormat = .rgb
    
    public init(
        width: Int,
        height: Int,
        normalization: Normalization = .imagenet,
        dataLayout: DataLayout = .hwc,
        colorFormat: ColorFormat = .rgb
    ) {
        self.width = width
        self.height = height
        self.normalization = normalization
        self.dataLayout = dataLayout
        self.colorFormat = colorFormat
    }
}

/// Options for grid extraction.
public struct GridOptions {
    /// Number of columns in the grid.
    public let columns: Int
    
    /// Number of rows in the grid.
    public let rows: Int
    
    /// Overlap between adjacent patches (0.0 to 1.0).
    public var overlap: Float = 0.0
    
    /// Normalization to apply.
    public var normalization: Normalization = .imagenet
    
    /// Data layout for output.
    public var dataLayout: DataLayout = .hwc
    
    /// Color format for output.
    public var colorFormat: ColorFormat = .rgb
    
    public init(
        columns: Int,
        rows: Int,
        overlap: Float = 0.0,
        normalization: Normalization = .imagenet,
        dataLayout: DataLayout = .hwc,
        colorFormat: ColorFormat = .rgb
    ) {
        self.columns = columns
        self.rows = rows
        self.overlap = overlap
        self.normalization = normalization
        self.dataLayout = dataLayout
        self.colorFormat = colorFormat
    }
}

/// Grid patch information.
public struct GridPatch {
    /// Pixel data for this patch.
    public let pixelData: PixelDataResult
    
    /// Row index in the grid.
    public let row: Int
    
    /// Column index in the grid.
    public let column: Int
    
    /// Position in original image.
    public let position: CropPosition
}

/// Result of grid extraction.
public struct GridExtractionResult {
    /// All patches from the grid.
    public let patches: [GridPatch]
    
    /// Number of rows.
    public let rows: Int
    
    /// Number of columns.
    public let columns: Int
    
    /// Processing time in milliseconds.
    public let processingTimeMs: Double
}

/// Options for random crop.
public struct RandomCropOptions {
    /// Width of each crop.
    public let width: Int
    
    /// Height of each crop.
    public let height: Int
    
    /// Number of random crops to generate.
    public var count: Int = 1
    
    /// Random seed for reproducibility (nil for random).
    public var seed: UInt64? = nil
    
    /// Normalization to apply.
    public var normalization: Normalization = .imagenet
    
    /// Data layout for output.
    public var dataLayout: DataLayout = .hwc
    
    /// Color format for output.
    public var colorFormat: ColorFormat = .rgb
    
    public init(
        width: Int,
        height: Int,
        count: Int = 1,
        seed: UInt64? = nil,
        normalization: Normalization = .imagenet,
        dataLayout: DataLayout = .hwc,
        colorFormat: ColorFormat = .rgb
    ) {
        self.width = width
        self.height = height
        self.count = count
        self.seed = seed
        self.normalization = normalization
        self.dataLayout = dataLayout
        self.colorFormat = colorFormat
    }
}

/// Result of a random crop.
public struct RandomCrop {
    /// Pixel data for this crop.
    public let pixelData: PixelDataResult
    
    /// Position in original image.
    public let position: CropPosition
    
    /// Seed used for this crop (if any).
    public let seed: UInt64?
}

/// Result of random crop operation.
public struct RandomCropResult {
    /// All random crops.
    public let crops: [RandomCrop]
    
    /// Processing time in milliseconds.
    public let processingTimeMs: Double
}

// MARK: - Multi-Crop Operations

/// Multi-crop operations for test-time augmentation.
///
/// Provides standard cropping strategies used in deep learning:
/// - **Five Crop**: 4 corners + center
/// - **Ten Crop**: Five crop + horizontal flips
/// - **Grid Extraction**: Divide image into grid of patches
/// - **Random Crop**: Random crops with optional seed for reproducibility
///
/// ## Example
///
/// ```swift
/// // Five crop for test-time augmentation
/// let result = try await MultiCropOperations.fiveCrop(
///     from: .cgImage(image),
///     options: CropOptions(width: 224, height: 224)
/// )
///
/// // Average predictions across crops
/// var avgPredictions = [Float](repeating: 0, count: numClasses)
/// for crop in result.crops {
///     let pred = model.predict(crop.data)
///     for i in 0..<numClasses {
///         avgPredictions[i] += pred[i] / Float(result.crops.count)
///     }
/// }
/// ```
public enum MultiCropOperations {
    
    // MARK: - Five Crop
    
    /// Extract five crops: four corners and center.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - options: Crop configuration
    /// - Returns: Multi-crop result with 5 crops
    /// - Throws: `PixelUtilsError` if processing fails
    public static func fiveCrop(
        from source: ImageSource,
        options: CropOptions
    ) async throws -> MultiCropResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load source image
        let cgImage = try loadCGImage(from: source)
        let imageWidth = cgImage.width
        let imageHeight = cgImage.height
        
        guard options.width <= imageWidth && options.height <= imageHeight else {
            throw PixelUtilsError.invalidOptions(
                "Crop size (\(options.width)x\(options.height)) exceeds image size (\(imageWidth)x\(imageHeight))"
            )
        }
        
        // Calculate positions: TL, TR, BL, BR, Center
        let positions: [(Int, Int)] = [
            (0, 0),  // Top-left
            (imageWidth - options.width, 0),  // Top-right
            (0, imageHeight - options.height),  // Bottom-left
            (imageWidth - options.width, imageHeight - options.height),  // Bottom-right
            ((imageWidth - options.width) / 2, (imageHeight - options.height) / 2)  // Center
        ]
        
        let pixelOptions = PixelDataOptions(
            colorFormat: options.colorFormat,
            normalization: options.normalization,
            dataLayout: options.dataLayout
        )
        
        var results: [PixelDataResult] = []
        var cropPositions: [CropPosition] = []
        
        for (x, y) in positions {
            let cropped = try cropImage(cgImage, x: x, y: y, width: options.width, height: options.height)
            let result = try await PixelExtractor.getPixelData(source: .cgImage(cropped), options: pixelOptions)
            results.append(result)
            cropPositions.append(CropPosition(
                x: x, y: y,
                width: options.width, height: options.height,
                isFlipped: false
            ))
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return MultiCropResult(
            crops: results,
            positions: cropPositions,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Ten Crop
    
    /// Extract ten crops: five crop + horizontal flips.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - options: Crop configuration
    /// - Returns: Multi-crop result with 10 crops
    /// - Throws: `PixelUtilsError` if processing fails
    public static func tenCrop(
        from source: ImageSource,
        options: CropOptions
    ) async throws -> MultiCropResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load source image
        let cgImage = try loadCGImage(from: source)
        let imageWidth = cgImage.width
        let imageHeight = cgImage.height
        
        guard options.width <= imageWidth && options.height <= imageHeight else {
            throw PixelUtilsError.invalidOptions(
                "Crop size (\(options.width)x\(options.height)) exceeds image size (\(imageWidth)x\(imageHeight))"
            )
        }
        
        // Calculate positions: TL, TR, BL, BR, Center
        let positions: [(Int, Int)] = [
            (0, 0),
            (imageWidth - options.width, 0),
            (0, imageHeight - options.height),
            (imageWidth - options.width, imageHeight - options.height),
            ((imageWidth - options.width) / 2, (imageHeight - options.height) / 2)
        ]
        
        let pixelOptions = PixelDataOptions(
            colorFormat: options.colorFormat,
            normalization: options.normalization,
            dataLayout: options.dataLayout
        )
        
        var results: [PixelDataResult] = []
        var cropPositions: [CropPosition] = []
        
        // Original crops
        for (x, y) in positions {
            let cropped = try cropImage(cgImage, x: x, y: y, width: options.width, height: options.height)
            let result = try await PixelExtractor.getPixelData(source: .cgImage(cropped), options: pixelOptions)
            results.append(result)
            cropPositions.append(CropPosition(
                x: x, y: y,
                width: options.width, height: options.height,
                isFlipped: false
            ))
        }
        
        // Flipped crops
        for (x, y) in positions {
            let cropped = try cropImage(cgImage, x: x, y: y, width: options.width, height: options.height)
            let flipped = try flipImageHorizontally(cropped)
            let result = try await PixelExtractor.getPixelData(source: .cgImage(flipped), options: pixelOptions)
            results.append(result)
            cropPositions.append(CropPosition(
                x: x, y: y,
                width: options.width, height: options.height,
                isFlipped: true
            ))
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return MultiCropResult(
            crops: results,
            positions: cropPositions,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Grid Extraction
    
    /// Extract patches from image in a grid pattern.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - options: Grid configuration
    /// - Returns: Grid extraction result
    /// - Throws: `PixelUtilsError` if processing fails
    public static func extractGrid(
        from source: ImageSource,
        options: GridOptions
    ) async throws -> GridExtractionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load source image
        let cgImage = try loadCGImage(from: source)
        let imageWidth = cgImage.width
        let imageHeight = cgImage.height
        
        // Calculate patch size with overlap
        let overlapFactor = 1.0 - Double(options.overlap)
        let patchWidth = Int(Double(imageWidth) / (Double(options.columns - 1) * overlapFactor + 1))
        let patchHeight = Int(Double(imageHeight) / (Double(options.rows - 1) * overlapFactor + 1))
        
        let stepX = Int(Double(patchWidth) * overlapFactor)
        let stepY = Int(Double(patchHeight) * overlapFactor)
        
        let pixelOptions = PixelDataOptions(
            colorFormat: options.colorFormat,
            normalization: options.normalization,
            dataLayout: options.dataLayout
        )
        
        var patches: [GridPatch] = []
        
        for row in 0..<options.rows {
            for col in 0..<options.columns {
                let x = min(col * stepX, imageWidth - patchWidth)
                let y = min(row * stepY, imageHeight - patchHeight)
                
                let actualWidth = min(patchWidth, imageWidth - x)
                let actualHeight = min(patchHeight, imageHeight - y)
                
                let cropped = try cropImage(cgImage, x: x, y: y, width: actualWidth, height: actualHeight)
                let result = try await PixelExtractor.getPixelData(source: .cgImage(cropped), options: pixelOptions)
                
                patches.append(GridPatch(
                    pixelData: result,
                    row: row,
                    column: col,
                    position: CropPosition(x: x, y: y, width: actualWidth, height: actualHeight)
                ))
            }
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return GridExtractionResult(
            patches: patches,
            rows: options.rows,
            columns: options.columns,
            processingTimeMs: processingTime
        )
    }
    
    // MARK: - Random Crop
    
    /// Extract random crops from an image.
    ///
    /// - Parameters:
    ///   - source: Image source
    ///   - options: Random crop configuration
    /// - Returns: Random crop result
    /// - Throws: `PixelUtilsError` if processing fails
    public static func randomCrop(
        from source: ImageSource,
        options: RandomCropOptions
    ) async throws -> RandomCropResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Load source image
        let cgImage = try loadCGImage(from: source)
        let imageWidth = cgImage.width
        let imageHeight = cgImage.height
        
        guard options.width <= imageWidth && options.height <= imageHeight else {
            throw PixelUtilsError.invalidOptions(
                "Crop size (\(options.width)x\(options.height)) exceeds image size (\(imageWidth)x\(imageHeight))"
            )
        }
        
        // Initialize random generator
        var generator: RandomNumberGenerator
        if let seed = options.seed {
            generator = SeededRandomNumberGenerator(seed: seed)
        } else {
            generator = SystemRandomNumberGenerator()
        }
        
        let maxX = imageWidth - options.width
        let maxY = imageHeight - options.height
        
        let pixelOptions = PixelDataOptions(
            colorFormat: options.colorFormat,
            normalization: options.normalization,
            dataLayout: options.dataLayout
        )
        
        var crops: [RandomCrop] = []
        
        for _ in 0..<options.count {
            let x = Int.random(in: 0...maxX, using: &generator)
            let y = Int.random(in: 0...maxY, using: &generator)
            
            let cropped = try cropImage(cgImage, x: x, y: y, width: options.width, height: options.height)
            let result = try await PixelExtractor.getPixelData(source: .cgImage(cropped), options: pixelOptions)
            
            crops.append(RandomCrop(
                pixelData: result,
                position: CropPosition(x: x, y: y, width: options.width, height: options.height),
                seed: options.seed
            ))
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return RandomCropResult(
            crops: crops,
            processingTimeMs: processingTime
        )
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
    
    private static func cropImage(_ image: CGImage, x: Int, y: Int, width: Int, height: Int) throws -> CGImage {
        let rect = CGRect(x: x, y: y, width: width, height: height)
        guard let cropped = image.cropping(to: rect) else {
            throw PixelUtilsError.processingFailed("Failed to crop image at (\(x), \(y), \(width), \(height))")
        }
        return cropped
    }
    
    private static func flipImageHorizontally(_ image: CGImage) throws -> CGImage {
        let width = image.width
        let height = image.height
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
        
        // Flip horizontally
        context.translateBy(x: CGFloat(width), y: 0)
        context.scaleBy(x: -1, y: 1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let flipped = context.makeImage() else {
            throw PixelUtilsError.processingFailed("Failed to create flipped image")
        }
        
        return flipped
    }
}
