//
//  SegmentationOutput.swift
//  SwiftPixelUtils
//
//  High-level API for processing semantic segmentation model outputs
//

import Foundation
import CoreGraphics

// MARK: - Segmentation Result Types

/// Result from segmentation output processing
public struct SegmentationResult {
    /// Class mask with class index at each pixel position
    /// Shape: [height, width] flattened to 1D array
    public let classMask: [Int]
    
    /// Confidence map with maximum probability at each pixel
    /// Shape: [height, width] flattened to 1D array
    public let confidenceMap: [Float]
    
    /// Width of the segmentation mask
    public let width: Int
    
    /// Height of the segmentation mask
    public let height: Int
    
    /// Number of classes in the model
    public let numClasses: Int
    
    /// Processing time in milliseconds
    public let processingTimeMs: Double
    
    /// Class statistics: class index -> pixel count
    public let classPixelCounts: [Int: Int]
    
    /// Labels for each class (if provided)
    public let labels: [String]?
    
    public init(
        classMask: [Int],
        confidenceMap: [Float],
        width: Int,
        height: Int,
        numClasses: Int,
        processingTimeMs: Double,
        classPixelCounts: [Int: Int],
        labels: [String]? = nil
    ) {
        self.classMask = classMask
        self.confidenceMap = confidenceMap
        self.width = width
        self.height = height
        self.numClasses = numClasses
        self.processingTimeMs = processingTimeMs
        self.classPixelCounts = classPixelCounts
        self.labels = labels
    }
    
    /// Get the class index at a specific position
    public func classAt(x: Int, y: Int) -> Int {
        guard x >= 0 && x < width && y >= 0 && y < height else { return 0 }
        return classMask[y * width + x]
    }
    
    /// Get the confidence at a specific position
    public func confidenceAt(x: Int, y: Int) -> Float {
        guard x >= 0 && x < width && y >= 0 && y < height else { return 0 }
        return confidenceMap[y * width + x]
    }
    
    /// Get classes present in the segmentation (excluding background at index 0)
    public var presentClasses: [Int] {
        classPixelCounts.keys.filter { $0 > 0 }.sorted()
    }
    
    /// Get a summary of detected classes with labels and pixel percentages
    public var classSummary: [(classIndex: Int, label: String, pixelCount: Int, percentage: Float)] {
        let totalPixels = width * height
        return classPixelCounts
            .filter { $0.key > 0 && $0.value > 0 } // Exclude background
            .sorted { $0.value > $1.value } // Sort by pixel count
            .map { (classIndex, pixelCount) in
                let label: String
                if let labelArray = labels, classIndex >= 0, classIndex < labelArray.count {
                    label = labelArray[classIndex]
                } else {
                    label = "class_\(classIndex)"
                }
                let percentage = Float(pixelCount) / Float(totalPixels) * 100
                return (classIndex, label, pixelCount, percentage)
            }
    }
    
    /// Get binary mask for a specific class
    /// - Parameter classIndex: The class index to extract
    /// - Returns: Binary mask where 1 = class present, 0 = not present
    public func binaryMask(forClass classIndex: Int) -> [Float] {
        classMask.map { $0 == classIndex ? 1.0 : 0.0 }
    }
    
    /// Convert to colored RGB mask for visualization
    /// - Parameter palette: Color palette to use (defaults to SegmentationColorPalette.voc)
    /// - Returns: RGB data [R, G, B, R, G, B, ...] for each pixel
    public func toColoredMask(
        palette: SegmentationColorPalette = .voc
    ) -> [UInt8] {
        var rgbData = [UInt8](repeating: 0, count: width * height * 3)
        
        for i in 0..<classMask.count {
            let classIndex = classMask[i]
            let color = palette.color(forClassIndex: classIndex)
            rgbData[i * 3] = color.r
            rgbData[i * 3 + 1] = color.g
            rgbData[i * 3 + 2] = color.b
        }
        
        return rgbData
    }
    
    /// Convert to colored RGBA mask for visualization
    /// - Parameters:
    ///   - palette: Color palette to use
    ///   - alpha: Alpha value for all pixels (0-255)
    /// - Returns: RGBA data [R, G, B, A, R, G, B, A, ...] for each pixel
    public func toColoredMaskRGBA(
        palette: SegmentationColorPalette = .voc,
        alpha: UInt8 = 255
    ) -> [UInt8] {
        var rgbaData = [UInt8](repeating: 0, count: width * height * 4)
        
        for i in 0..<classMask.count {
            let classIndex = classMask[i]
            let color = palette.color(forClassIndex: classIndex)
            rgbaData[i * 4] = color.r
            rgbaData[i * 4 + 1] = color.g
            rgbaData[i * 4 + 2] = color.b
            rgbaData[i * 4 + 3] = alpha
        }
        
        return rgbaData
    }
    
    /// Create a CGImage from the colored segmentation mask
    /// - Parameter palette: Color palette to use
    /// - Returns: CGImage with colored segmentation
    public func toColoredCGImage(palette: SegmentationColorPalette = .voc) -> CGImage? {
        let rgbaData = toColoredMaskRGBA(palette: palette)
        
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
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            return nil
        }
        
        return cgImage
    }
}

// MARK: - Color Palette

/// Color palette for segmentation visualization
public struct SegmentationColorPalette {
    /// Colors for each class index (r, g, b)
    public let colors: [(r: UInt8, g: UInt8, b: UInt8)]
    
    /// Name of the palette
    public let name: String
    
    public init(name: String, colors: [(r: UInt8, g: UInt8, b: UInt8)]) {
        self.name = name
        self.colors = colors
    }
    
    /// Get color for a class index (cycles through palette if index exceeds count)
    public func color(forClassIndex index: Int) -> (r: UInt8, g: UInt8, b: UInt8) {
        guard !colors.isEmpty else { return (128, 128, 128) }
        return colors[index % colors.count]
    }
    
    // MARK: - Predefined Palettes
    
    /// Pascal VOC color palette (21 classes)
    /// Classic segmentation colors used in DeepLab and many other models
    public static let voc = SegmentationColorPalette(
        name: "PASCAL VOC",
        colors: [
            (0, 0, 0),         // 0: background (black)
            (128, 0, 0),       // 1: aeroplane (maroon)
            (0, 128, 0),       // 2: bicycle (green)
            (128, 128, 0),     // 3: bird (olive)
            (0, 0, 128),       // 4: boat (navy)
            (128, 0, 128),     // 5: bottle (purple)
            (0, 128, 128),     // 6: bus (teal)
            (128, 128, 128),   // 7: car (gray)
            (64, 0, 0),        // 8: cat (dark red)
            (192, 0, 0),       // 9: chair (red)
            (64, 128, 0),      // 10: cow (dark green)
            (192, 128, 0),     // 11: diningtable (orange)
            (64, 0, 128),      // 12: dog (dark purple)
            (192, 0, 128),     // 13: horse (pink)
            (64, 128, 128),    // 14: motorbike (dark teal)
            (192, 128, 128),   // 15: person (light pink)
            (0, 64, 0),        // 16: pottedplant (dark green)
            (128, 64, 0),      // 17: sheep (brown)
            (0, 192, 0),       // 18: sofa (lime)
            (128, 192, 0),     // 19: train (yellow-green)
            (0, 64, 128)       // 20: tvmonitor (dark blue)
        ]
    )
    
    /// ADE20K color palette (150 classes)
    /// More diverse palette for scene parsing
    public static let ade20k = SegmentationColorPalette(
        name: "ADE20K",
        colors: generateADE20KColors()
    )
    
    /// Cityscapes color palette (19 classes)
    /// Used for urban scene segmentation
    public static let cityscapes = SegmentationColorPalette(
        name: "Cityscapes",
        colors: [
            (128, 64, 128),    // 0: road
            (244, 35, 232),    // 1: sidewalk
            (70, 70, 70),      // 2: building
            (102, 102, 156),   // 3: wall
            (190, 153, 153),   // 4: fence
            (153, 153, 153),   // 5: pole
            (250, 170, 30),    // 6: traffic light
            (220, 220, 0),     // 7: traffic sign
            (107, 142, 35),    // 8: vegetation
            (152, 251, 152),   // 9: terrain
            (70, 130, 180),    // 10: sky
            (220, 20, 60),     // 11: person
            (255, 0, 0),       // 12: rider
            (0, 0, 142),       // 13: car
            (0, 0, 70),        // 14: truck
            (0, 60, 100),      // 15: bus
            (0, 80, 100),      // 16: train
            (0, 0, 230),       // 17: motorcycle
            (119, 11, 32)      // 18: bicycle
        ]
    )
    
    /// Generate a colorful palette with the specified number of colors
    public static func rainbow(numClasses: Int) -> SegmentationColorPalette {
        var colors: [(r: UInt8, g: UInt8, b: UInt8)] = [(0, 0, 0)] // Background black
        
        for i in 1..<numClasses {
            let hue = Float(i - 1) / Float(numClasses - 1)
            let rgb = hsvToRgb(h: hue, s: 0.8, v: 0.9)
            colors.append(rgb)
        }
        
        return SegmentationColorPalette(name: "Rainbow \(numClasses)", colors: colors)
    }
    
    /// Generate distinctive colors for ADE20K
    private static func generateADE20KColors() -> [(r: UInt8, g: UInt8, b: UInt8)] {
        var colors: [(r: UInt8, g: UInt8, b: UInt8)] = []
        
        // First few ADE20K classes with standard colors
        let baseColors: [(r: UInt8, g: UInt8, b: UInt8)] = [
            (0, 0, 0),         // 0: background
            (120, 120, 120),   // 1: wall
            (180, 120, 120),   // 2: building
            (6, 230, 230),     // 3: sky
            (80, 50, 50),      // 4: floor
            (4, 200, 3),       // 5: tree
            (120, 120, 80),    // 6: ceiling
            (140, 140, 140),   // 7: road
            (204, 5, 255),     // 8: bed
            (230, 230, 230),   // 9: windowpane
            (4, 250, 7),       // 10: grass
            (224, 5, 255),     // 11: cabinet
            (235, 255, 7),     // 12: sidewalk
            (150, 5, 61),      // 13: person
            (120, 120, 70)     // 14: earth
        ]
        
        colors.append(contentsOf: baseColors)
        
        // Generate remaining colors using HSV with varying saturation and value
        for i in baseColors.count..<150 {
            let hue = Float((i * 47) % 360) / 360.0
            let sat = 0.5 + Float((i * 13) % 50) / 100.0
            let val = 0.6 + Float((i * 29) % 40) / 100.0
            colors.append(hsvToRgb(h: hue, s: sat, v: val))
        }
        
        return colors
    }
    
    /// Convert HSV to RGB
    private static func hsvToRgb(h: Float, s: Float, v: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        let c = v * s
        let x = c * (1 - abs(fmod(h * 6, 2) - 1))
        let m = v - c
        
        var r: Float = 0, g: Float = 0, b: Float = 0
        
        if h < 1/6 {
            r = c; g = x; b = 0
        } else if h < 2/6 {
            r = x; g = c; b = 0
        } else if h < 3/6 {
            r = 0; g = c; b = x
        } else if h < 4/6 {
            r = 0; g = x; b = c
        } else if h < 5/6 {
            r = x; g = 0; b = c
        } else {
            r = c; g = 0; b = x
        }
        
        return (
            UInt8((r + m) * 255),
            UInt8((g + m) * 255),
            UInt8((b + m) * 255)
        )
    }
}

// MARK: - SegmentationOutput Main API

/// High-level utilities for processing semantic segmentation model outputs.
///
/// This class provides a simplified API for processing segmentation model outputs:
/// 1. Parsing output format (logits or probabilities)
/// 2. Computing argmax to get class predictions
/// 3. Generating confidence maps
/// 4. Creating colored visualizations
///
/// ## Supported Formats
///
/// | Format | Output Shape | Description |
/// |--------|--------------|-------------|
/// | `.logits` | [1, H, W, C] | Raw class logits (NHWC) |
/// | `.logitsNCHW` | [1, C, H, W] | Raw class logits (NCHW) |
/// | `.probabilities` | [1, H, W, C] | Softmax probabilities |
/// | `.probabilitiesNCHW` | [1, C, H, W] | Softmax probabilities (NCHW) |
///
/// ## Usage
///
/// ```swift
/// // Process DeepLabV3 output
/// let result = try SegmentationOutput.process(
///     outputData: modelOutput,
///     format: .logits(height: 257, width: 257, numClasses: 21),
///     labels: .voc
/// )
///
/// // Get detected classes
/// for (classIndex, label, pixelCount, percentage) in result.classSummary {
///     print("\(label): \(String(format: "%.1f%%", percentage))")
/// }
///
/// // Visualize with colored mask
/// let coloredImage = result.toColoredCGImage(palette: .voc)
/// ```
public enum SegmentationOutput {
    
    // MARK: - Output Format Types
    
    /// Segmentation model output format specification
    public enum OutputFormat {
        /// Raw class logits in NHWC format: [1, height, width, numClasses]
        /// Used by DeepLabV3 and many TFLite models
        case logits(height: Int, width: Int, numClasses: Int)
        
        /// Raw class logits in NCHW format: [1, numClasses, height, width]
        /// Used by many PyTorch models
        case logitsNCHW(height: Int, width: Int, numClasses: Int)
        
        /// Softmax probabilities in NHWC format: [1, height, width, numClasses]
        case probabilities(height: Int, width: Int, numClasses: Int)
        
        /// Softmax probabilities in NCHW format: [1, numClasses, height, width]
        case probabilitiesNCHW(height: Int, width: Int, numClasses: Int)
        
        /// Pre-computed argmax output: [1, height, width]
        /// Some models output class indices directly
        case argmax(height: Int, width: Int, numClasses: Int)
        
        var height: Int {
            switch self {
            case .logits(let h, _, _), .logitsNCHW(let h, _, _),
                 .probabilities(let h, _, _), .probabilitiesNCHW(let h, _, _),
                 .argmax(let h, _, _):
                return h
            }
        }
        
        var width: Int {
            switch self {
            case .logits(_, let w, _), .logitsNCHW(_, let w, _),
                 .probabilities(_, let w, _), .probabilitiesNCHW(_, let w, _),
                 .argmax(_, let w, _):
                return w
            }
        }
        
        var numClasses: Int {
            switch self {
            case .logits(_, _, let n), .logitsNCHW(_, _, let n),
                 .probabilities(_, _, let n), .probabilitiesNCHW(_, _, let n),
                 .argmax(_, _, let n):
                return n
            }
        }
        
        var expectedElementCount: Int {
            switch self {
            case .argmax(let h, let w, _):
                return h * w
            default:
                return height * width * numClasses
            }
        }
    }
    
    /// Label source for mapping class indices to names
    public enum LabelSource {
        /// Pascal VOC labels (21 classes with background)
        case voc
        /// ADE20K labels (150 classes)
        case ade20k
        /// Cityscapes labels (19 classes)
        case cityscapes
        /// COCO-Stuff labels (171 classes)
        case cocoStuff
        /// Custom labels array
        case custom([String])
        /// No labels (returns indices as labels)
        case none
    }
    
    // MARK: - Main Processing Methods
    
    /// Process segmentation model output with automatic parsing and visualization.
    ///
    /// This is the main entry point for processing segmentation outputs. It handles:
    /// - Format-specific parsing (NHWC vs NCHW, logits vs probabilities)
    /// - Argmax computation to get class predictions
    /// - Confidence map generation
    /// - Class statistics computation
    ///
    /// - Parameters:
    ///   - outputData: Raw output data from the model
    ///   - format: Output format specification
    ///   - labels: Label source for mapping indices to names
    /// - Returns: SegmentationResult with processed mask and statistics
    /// - Throws: PixelUtilsError if processing fails
    public static func process(
        outputData: Data,
        format: OutputFormat,
        labels: LabelSource = .voc
    ) throws -> SegmentationResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Convert Data to Float array
        let floats = outputData.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
        
        return try process(
            floatOutput: floats,
            format: format,
            labels: labels,
            startTime: startTime
        )
    }
    
    /// Process segmentation model output from a Float array
    public static func process(
        floatOutput: [Float],
        format: OutputFormat,
        labels: LabelSource = .voc
    ) throws -> SegmentationResult {
        try process(
            floatOutput: floatOutput,
            format: format,
            labels: labels,
            startTime: CFAbsoluteTimeGetCurrent()
        )
    }
    
    // MARK: - Private Implementation
    
    private static func process(
        floatOutput: [Float],
        format: OutputFormat,
        labels: LabelSource,
        startTime: CFAbsoluteTime
    ) throws -> SegmentationResult {
        let height = format.height
        let width = format.width
        let numClasses = format.numClasses
        let pixelCount = height * width
        
        // Validate output size
        guard floatOutput.count >= format.expectedElementCount else {
            throw PixelUtilsError.invalidOptions(
                "Output size \(floatOutput.count) doesn't match expected \(format.expectedElementCount) " +
                "for format \(height)x\(width)x\(numClasses)"
            )
        }
        
        // Initialize result arrays
        var classMask = [Int](repeating: 0, count: pixelCount)
        var confidenceMap = [Float](repeating: 0, count: pixelCount)
        var classPixelCounts = [Int: Int]()
        
        // Process based on format
        switch format {
        case .argmax:
            // Output is already class indices
            for i in 0..<pixelCount {
                let classIndex = Int(floatOutput[i])
                classMask[i] = classIndex
                confidenceMap[i] = 1.0 // No confidence info in argmax
                classPixelCounts[classIndex, default: 0] += 1
            }
            
        case .logits, .probabilities:
            // NHWC format: [1, H, W, C]
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = y * width + x
                    let baseIndex = pixelIndex * numClasses
                    
                    // Find argmax class
                    var maxValue: Float = -Float.infinity
                    var maxClass = 0
                    
                    for c in 0..<numClasses {
                        let value = floatOutput[baseIndex + c]
                        if value > maxValue {
                            maxValue = value
                            maxClass = c
                        }
                    }
                    
                    classMask[pixelIndex] = maxClass
                    
                    // For logits, convert to confidence using softmax
                    if case .logits = format {
                        // Compute softmax for this pixel
                        var expSum: Float = 0
                        for c in 0..<numClasses {
                            expSum += exp(floatOutput[baseIndex + c] - maxValue)
                        }
                        confidenceMap[pixelIndex] = 1.0 / expSum // exp(0) / sum
                    } else {
                        confidenceMap[pixelIndex] = maxValue
                    }
                    
                    classPixelCounts[maxClass, default: 0] += 1
                }
            }
            
        case .logitsNCHW, .probabilitiesNCHW:
            // NCHW format: [1, C, H, W]
            let channelStride = height * width
            
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = y * width + x
                    
                    // Find argmax class
                    var maxValue: Float = -Float.infinity
                    var maxClass = 0
                    
                    for c in 0..<numClasses {
                        let value = floatOutput[c * channelStride + pixelIndex]
                        if value > maxValue {
                            maxValue = value
                            maxClass = c
                        }
                    }
                    
                    classMask[pixelIndex] = maxClass
                    
                    // For logits, convert to confidence using softmax
                    if case .logitsNCHW = format {
                        var expSum: Float = 0
                        for c in 0..<numClasses {
                            expSum += exp(floatOutput[c * channelStride + pixelIndex] - maxValue)
                        }
                        confidenceMap[pixelIndex] = 1.0 / expSum
                    } else {
                        confidenceMap[pixelIndex] = maxValue
                    }
                    
                    classPixelCounts[maxClass, default: 0] += 1
                }
            }
        }
        
        // Resolve labels
        let labelArray = resolveLabels(labels, numClasses: numClasses)
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return SegmentationResult(
            classMask: classMask,
            confidenceMap: confidenceMap,
            width: width,
            height: height,
            numClasses: numClasses,
            processingTimeMs: processingTime,
            classPixelCounts: classPixelCounts,
            labels: labelArray
        )
    }
    
    private static func resolveLabels(_ source: LabelSource, numClasses: Int) -> [String]? {
        switch source {
        case .voc:
            return LabelDatabase.getAllLabels(for: .voc)
        case .ade20k:
            return LabelDatabase.getAllLabels(for: .ade20k)
        case .cityscapes:
            // Return Cityscapes labels
            return [
                "road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "rider", "car", "truck", "bus", "train",
                "motorcycle", "bicycle"
            ]
        case .cocoStuff:
            // Return first 171 COCO-Stuff labels
            return nil // Would need to add COCO-Stuff labels
        case .custom(let labels):
            return labels
        case .none:
            return nil
        }
    }
}

// MARK: - Drawing Extension

extension Drawing {
    
    /// Overlay a colored segmentation mask on an image
    ///
    /// - Parameters:
    ///   - source: Image source to overlay on
    ///   - segmentation: Segmentation result from SegmentationOutput.process()
    ///   - palette: Color palette to use
    ///   - alpha: Overlay alpha (0-1)
    ///   - excludeBackground: Whether to exclude background (class 0) from overlay
    /// - Returns: Drawing result with overlaid mask
    /// - Throws: PixelUtilsError if drawing fails
    public static func overlaySegmentation(
        on source: ImageSource,
        segmentation: SegmentationResult,
        palette: SegmentationColorPalette = .voc,
        alpha: CGFloat = 0.5,
        excludeBackground: Bool = true
    ) throws -> DrawingResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let cgImage = try loadCGImage(from: source)
        let imageWidth = cgImage.width
        let imageHeight = cgImage.height
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: imageWidth,
            height: imageHeight,
            bitsPerComponent: 8,
            bytesPerRow: imageWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PixelUtilsError.processingFailed("Failed to create graphics context")
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
        
        // Calculate scale factors
        let scaleX = Float(imageWidth) / Float(segmentation.width)
        let scaleY = Float(imageHeight) / Float(segmentation.height)
        
        // Draw segmentation overlay
        for y in 0..<segmentation.height {
            for x in 0..<segmentation.width {
                let classIndex = segmentation.classAt(x: x, y: y)
                
                // Skip background if requested
                if excludeBackground && classIndex == 0 { continue }
                
                let color = palette.color(forClassIndex: classIndex)
                
                context.setFillColor(CGColor(
                    red: CGFloat(color.r) / 255.0,
                    green: CGFloat(color.g) / 255.0,
                    blue: CGFloat(color.b) / 255.0,
                    alpha: alpha
                ))
                
                // Map to image coordinates (flip Y for CoreGraphics)
                let imgX = CGFloat(Float(x) * scaleX)
                let imgY = CGFloat(Float(imageHeight) - Float(y + 1) * scaleY)
                let rectWidth = CGFloat(scaleX)
                let rectHeight = CGFloat(scaleY)
                
                context.fill(CGRect(x: imgX, y: imgY, width: rectWidth, height: rectHeight))
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
            width: imageWidth,
            height: imageHeight,
            processingTimeMs: processingTime
        )
    }
}
