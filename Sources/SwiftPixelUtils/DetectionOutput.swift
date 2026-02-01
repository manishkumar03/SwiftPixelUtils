//
//  DetectionOutput.swift
//  SwiftPixelUtils
//
//  High-level API for processing object detection model outputs
//

import Foundation
import CoreGraphics

/// A single object detection with bounding box, class, and confidence
public struct ObjectDetection: Equatable {
    /// Class index from the model
    public let classIndex: Int
    /// Human-readable label
    public let label: String
    /// Confidence score (0-1)
    public let confidence: Float
    /// Bounding box in normalized coordinates (0-1)
    public let boundingBox: CGRect
    /// Bounding box in pixel coordinates (if image size was provided)
    public let pixelBoundingBox: CGRect?
    
    public init(
        classIndex: Int,
        label: String,
        confidence: Float,
        boundingBox: CGRect,
        pixelBoundingBox: CGRect? = nil
    ) {
        self.classIndex = classIndex
        self.label = label
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.pixelBoundingBox = pixelBoundingBox
    }
    
    /// Convert this detection to a DrawableBox for visualization
    /// - Parameters:
    ///   - imageSize: The size of the image to draw on (uses pixelBoundingBox if available, otherwise scales boundingBox)
    ///   - color: Optional color override. If nil, uses default color palette based on classIndex
    /// - Returns: A DrawableBox ready for Drawing.drawBoxes()
    public func toDrawableBox(
        imageSize: CGSize? = nil,
        color: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)? = nil
    ) -> DrawableBox {
        let pixelRect: CGRect
        
        if let pbb = pixelBoundingBox {
            // Use pre-calculated pixel coordinates
            pixelRect = pbb
        } else if let imgSize = imageSize {
            // Scale normalized coordinates to image size
            pixelRect = CGRect(
                x: boundingBox.minX * imgSize.width,
                y: boundingBox.minY * imgSize.height,
                width: boundingBox.width * imgSize.width,
                height: boundingBox.height * imgSize.height
            )
        } else {
            // Use normalized coordinates as-is (0-1 range)
            pixelRect = boundingBox
        }
        
        // Convert to [x1, y1, x2, y2] format
        let boxCoords: [Float] = [
            Float(pixelRect.minX),
            Float(pixelRect.minY),
            Float(pixelRect.maxX),
            Float(pixelRect.maxY)
        ]
        
        // Use provided color or get from default palette
        let boxColor = color ?? DetectionColorPalette.color(forClassIndex: classIndex)
        
        return DrawableBox(
            box: boxCoords,
            label: label,
            score: confidence,
            color: boxColor
        )
    }
}

/// Result from detection output processing
public struct DetectionResult {
    /// All detections after NMS, sorted by confidence
    public let detections: [ObjectDetection]
    /// Processing time in milliseconds
    public let processingTimeMs: Double
    /// Number of raw detections before filtering
    public let rawDetectionCount: Int
    /// Number of detections after confidence filtering (before NMS)
    public let postConfidenceFilterCount: Int
    
    public init(
        detections: [ObjectDetection],
        processingTimeMs: Double,
        rawDetectionCount: Int,
        postConfidenceFilterCount: Int
    ) {
        self.detections = detections
        self.processingTimeMs = processingTimeMs
        self.rawDetectionCount = rawDetectionCount
        self.postConfidenceFilterCount = postConfidenceFilterCount
    }
    
    /// Convenience accessor for count of final detections
    public var count: Int { detections.count }
    
    /// Filter detections by class
    public func filter(byClass classIndex: Int) -> [ObjectDetection] {
        detections.filter { $0.classIndex == classIndex }
    }
    
    /// Filter detections by label
    public func filter(byLabel label: String) -> [ObjectDetection] {
        detections.filter { $0.label.lowercased() == label.lowercased() }
    }
    
    /// Convert all detections to DrawableBox array for visualization
    ///
    /// This is a convenience method for drawing detection results on an image.
    ///
    /// ## Example
    /// ```swift
    /// let result = try DetectionOutput.process(...)
    /// let boxes = result.toDrawableBoxes(imageSize: image.size)
    /// let drawn = try Drawing.drawBoxes(on: .uiImage(image), boxes: boxes)
    /// ```
    ///
    /// - Parameter imageSize: The size of the image to draw on. Required if pixelBoundingBox is nil.
    /// - Returns: Array of DrawableBox ready for Drawing.drawBoxes()
    public func toDrawableBoxes(imageSize: CGSize? = nil) -> [DrawableBox] {
        detections.map { $0.toDrawableBox(imageSize: imageSize) }
    }
}

/// Color palette for detection visualization
///
/// Provides a consistent set of visually distinct colors for drawing bounding boxes.
/// Colors are selected to be easily distinguishable and work well on various backgrounds.
public enum DetectionColorPalette {
    /// Default color palette with 20 distinct colors
    public static let colors: [(r: UInt8, g: UInt8, b: UInt8, a: UInt8)] = [
        (255, 0, 0, 255),       // Red
        (0, 255, 0, 255),       // Green
        (0, 0, 255, 255),       // Blue
        (255, 255, 0, 255),     // Yellow
        (255, 0, 255, 255),     // Magenta
        (0, 255, 255, 255),     // Cyan
        (255, 128, 0, 255),     // Orange
        (128, 0, 255, 255),     // Purple
        (0, 255, 128, 255),     // Spring Green
        (255, 128, 128, 255),   // Light Red
        (128, 255, 128, 255),   // Light Green
        (128, 128, 255, 255),   // Light Blue
        (255, 200, 0, 255),     // Gold
        (255, 0, 128, 255),     // Pink
        (0, 128, 255, 255),     // Sky Blue
        (128, 255, 0, 255),     // Lime
        (255, 128, 255, 255),   // Light Magenta
        (128, 255, 255, 255),   // Light Cyan
        (200, 100, 50, 255),    // Brown
        (100, 200, 150, 255)    // Teal
    ]
    
    /// Get color for a class index (cycles through palette)
    public static func color(forClassIndex index: Int) -> (r: UInt8, g: UInt8, b: UInt8, a: UInt8) {
        colors[index % colors.count]
    }
}

/// High-level utilities for processing object detection model outputs.
///
/// This class provides a simplified API for the common pattern of:
/// 1. Parsing model output format (YOLO, SSD, etc.)
/// 2. Applying confidence threshold
/// 3. Running Non-Maximum Suppression (NMS)
/// 4. Scaling boxes to image coordinates
/// 5. Mapping to human-readable labels
///
/// ## Supported Formats
///
/// | Format | Output Shape | Description |
/// |--------|--------------|-------------|
/// | `.yolov5` / `.yolov8` | [1, N, 85] or [1, 84, N] | YOLO family |
/// | `.ssd` | boxes + scores tensors | SSD MobileNet |
/// | `.efficientDet` | boxes + scores + classes | EfficientDet |
///
/// ## Usage
///
/// ```swift
/// // One-line output processing for YOLOv8
/// let result = try DetectionOutput.process(
///     outputData: modelOutput,
///     format: .yolov8(numClasses: 80),
///     confidenceThreshold: 0.5,
///     iouThreshold: 0.45,
///     labels: .coco
/// )
///
/// for detection in result.detections {
///     print("\(detection.label): \(String(format: "%.1f%%", detection.confidence * 100))")
///     print("  Box: \(detection.boundingBox)")
/// }
/// ```
public enum DetectionOutput {
    
    // MARK: - Output Format Types
    
    /// Detection model output format specification
    public enum OutputFormat {
        /// YOLOv5 format: [1, N, 5+numClasses] where each row is [cx, cy, w, h, obj_conf, class_scores...]
        case yolov5(numClasses: Int)
        
        /// YOLOv8 format: [1, 4+numClasses, N] transposed, no objectness score
        case yolov8(numClasses: Int)
        
        /// YOLOv8 format already transposed to [1, N, 4+numClasses]
        case yolov8Transposed(numClasses: Int)
        
        /// SSD format with separate boxes and scores tensors
        case ssd(numClasses: Int)
        
        /// EfficientDet format
        case efficientDet(numClasses: Int)
        
        /// Custom format with explicit dimensions
        /// - Parameters:
        ///   - boxOffset: Starting index of box coordinates in each detection
        ///   - classOffset: Starting index of class scores
        ///   - numClasses: Number of classes
        ///   - boxFormat: Format of bounding box coordinates
        ///   - hasObjectness: Whether there's an objectness score before class scores
        case custom(boxOffset: Int, classOffset: Int, numClasses: Int, boxFormat: BoxFormat, hasObjectness: Bool)
    }
    
    /// Label source specification (same as ClassificationOutput for consistency)
    public enum LabelSource {
        /// COCO labels (80 classes) - most common for detection
        case coco
        /// Pascal VOC labels (20 classes, no background)
        case voc
        /// Pascal VOC labels with background class at index 0 (21 classes)
        case vocWithBackground
        /// Custom labels array
        case custom([String])
        /// No labels (returns indices as labels)
        case none
    }
    
    /// Coordinate space of the model output
    ///
    /// Different models output bounding box coordinates in different spaces:
    /// - **Normalized (0-1)**: Coordinates are already normalized to 0-1 range
    /// - **Pixel space**: Coordinates are in model input pixel space (e.g., 0-640)
    ///
    /// Many TFLite YOLO models output **normalized** coordinates, while the original
    /// PyTorch YOLO outputs **pixel space** coordinates. Check your model's documentation
    /// or inspect the raw output values to determine which to use.
    public enum OutputCoordinateSpace {
        /// Coordinates are in model input pixel space (e.g., 0-640 for a 640x640 model)
        /// This is the default for original PyTorch YOLO exports
        case pixelSpace
        
        /// Coordinates are already normalized to 0-1 range
        /// Common in TFLite conversions and some ONNX exports
        case normalized
    }
    
    // MARK: - Main Processing Methods
    
    /// Process detection model output with automatic parsing, NMS, and label mapping.
    ///
    /// This is the main entry point for processing detection outputs. It handles:
    /// - Format-specific parsing (YOLO, SSD, etc.)
    /// - Confidence thresholding
    /// - Non-Maximum Suppression
    /// - Box coordinate scaling
    /// - Label mapping
    ///
    /// - Parameters:
    ///   - outputData: Raw output data from the model
    ///   - format: Output format specification
    ///   - confidenceThreshold: Minimum confidence to keep (default: 0.5)
    ///   - iouThreshold: IoU threshold for NMS (default: 0.45)
    ///   - maxDetections: Maximum number of detections to return (default: 100)
    ///   - labels: Label source for mapping indices to names
    ///   - imageSize: Original image size for pixel coordinate conversion (optional)
    ///   - modelInputSize: Model input size for box scaling (default: 640x640)
    ///   - outputCoordinateSpace: Whether model outputs normalized or pixel-space coordinates (default: .pixelSpace)
    /// - Returns: DetectionResult with processed detections
    /// - Throws: PixelUtilsError if processing fails
    public static func process(
        outputData: Data,
        format: OutputFormat,
        confidenceThreshold: Float = 0.5,
        iouThreshold: Float = 0.45,
        maxDetections: Int = 100,
        labels: LabelSource = .coco,
        imageSize: CGSize? = nil,
        modelInputSize: CGSize = CGSize(width: 640, height: 640),
        outputCoordinateSpace: OutputCoordinateSpace = .pixelSpace
    ) throws -> DetectionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Convert Data to Float array
        let floats = outputData.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
        
        return try process(
            floatOutput: floats,
            format: format,
            confidenceThreshold: confidenceThreshold,
            iouThreshold: iouThreshold,
            maxDetections: maxDetections,
            labels: labels,
            imageSize: imageSize,
            modelInputSize: modelInputSize,
            outputCoordinateSpace: outputCoordinateSpace,
            startTime: startTime
        )
    }
    
    /// Process detection model output from a Float array
    public static func process(
        floatOutput: [Float],
        format: OutputFormat,
        confidenceThreshold: Float = 0.5,
        iouThreshold: Float = 0.45,
        maxDetections: Int = 100,
        labels: LabelSource = .coco,
        imageSize: CGSize? = nil,
        modelInputSize: CGSize = CGSize(width: 640, height: 640),
        outputCoordinateSpace: OutputCoordinateSpace = .pixelSpace
    ) throws -> DetectionResult {
        try process(
            floatOutput: floatOutput,
            format: format,
            confidenceThreshold: confidenceThreshold,
            iouThreshold: iouThreshold,
            maxDetections: maxDetections,
            labels: labels,
            imageSize: imageSize,
            modelInputSize: modelInputSize,
            outputCoordinateSpace: outputCoordinateSpace,
            startTime: CFAbsoluteTimeGetCurrent()
        )
    }
    
    private static func process(
        floatOutput: [Float],
        format: OutputFormat,
        confidenceThreshold: Float,
        iouThreshold: Float,
        maxDetections: Int,
        labels: LabelSource,
        imageSize: CGSize?,
        modelInputSize: CGSize,
        outputCoordinateSpace: OutputCoordinateSpace,
        startTime: CFAbsoluteTime
    ) throws -> DetectionResult {
        
        // Step 1: Parse raw detections based on format
        let rawDetections: [RawDetection]
        
        switch format {
        case .yolov5(let numClasses):
            rawDetections = parseYOLOv5(floatOutput, numClasses: numClasses)
        case .yolov8(let numClasses):
            rawDetections = parseYOLOv8(floatOutput, numClasses: numClasses)
        case .yolov8Transposed(let numClasses):
            rawDetections = parseYOLOv8Transposed(floatOutput, numClasses: numClasses)
        case .ssd(let numClasses):
            rawDetections = parseSSD(floatOutput, numClasses: numClasses)
        case .efficientDet(let numClasses):
            rawDetections = parseEfficientDet(floatOutput, numClasses: numClasses)
        case .custom(let boxOffset, let classOffset, let numClasses, let boxFormat, let hasObjectness):
            rawDetections = parseCustom(floatOutput, boxOffset: boxOffset, classOffset: classOffset, numClasses: numClasses, boxFormat: boxFormat, hasObjectness: hasObjectness)
        }
        
        let rawCount = rawDetections.count
        
        // Step 2: Filter by confidence
        let confidenceFiltered = rawDetections.filter { $0.confidence >= confidenceThreshold }
        let postConfidenceCount = confidenceFiltered.count
        
        // Step 3: Apply NMS
        let nmsResults = applyNMS(
            detections: confidenceFiltered,
            iouThreshold: iouThreshold,
            maxDetections: maxDetections
        )
        
        // Step 4: Convert to final detections with labels and scaled coordinates
        let finalDetections = nmsResults.map { raw -> ObjectDetection in
            let label = getLabel(for: raw.classIndex, source: labels)
            
            // Get normalized coordinates (0-1 range) based on coordinate space
            let normalizedBox: CGRect
            switch outputCoordinateSpace {
            case .pixelSpace:
                // Model outputs pixel coordinates (e.g., 0-640), divide by model size to normalize
                normalizedBox = CGRect(
                    x: CGFloat(raw.box[0] / Float(modelInputSize.width)),
                    y: CGFloat(raw.box[1] / Float(modelInputSize.height)),
                    width: CGFloat(raw.box[2] / Float(modelInputSize.width)),
                    height: CGFloat(raw.box[3] / Float(modelInputSize.height))
                )
            case .normalized:
                // Model outputs already normalized coordinates (0-1), use as-is
                normalizedBox = CGRect(
                    x: CGFloat(raw.box[0]),
                    y: CGFloat(raw.box[1]),
                    width: CGFloat(raw.box[2]),
                    height: CGFloat(raw.box[3])
                )
            }
            
            // Calculate pixel coordinates if image size provided
            let pixelBox: CGRect?
            if let imgSize = imageSize {
                pixelBox = CGRect(
                    x: normalizedBox.minX * imgSize.width,
                    y: normalizedBox.minY * imgSize.height,
                    width: normalizedBox.width * imgSize.width,
                    height: normalizedBox.height * imgSize.height
                )
            } else {
                pixelBox = nil
            }
            
            return ObjectDetection(
                classIndex: raw.classIndex,
                label: label,
                confidence: raw.confidence,
                boundingBox: normalizedBox,
                pixelBoundingBox: pixelBox
            )
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return DetectionResult(
            detections: finalDetections,
            processingTimeMs: processingTime,
            rawDetectionCount: rawCount,
            postConfidenceFilterCount: postConfidenceCount
        )
    }
    
    // MARK: - Format Parsers
    
    private struct RawDetection {
        let box: [Float]  // [x, y, w, h] in model coordinates
        let classIndex: Int
        let confidence: Float
    }
    
    /// Parse YOLOv5 output: [1, N, 5+numClasses] or flattened [N * (5+numClasses)]
    private static func parseYOLOv5(_ output: [Float], numClasses: Int) -> [RawDetection] {
        let stride = 5 + numClasses  // cx, cy, w, h, obj_conf, class_scores...
        let numDetections = output.count / stride
        
        var detections: [RawDetection] = []
        detections.reserveCapacity(numDetections)
        
        for i in 0..<numDetections {
            let offset = i * stride
            guard offset + stride <= output.count else { break }
            
            let cx = output[offset]
            let cy = output[offset + 1]
            let w = output[offset + 2]
            let h = output[offset + 3]
            let objectness = output[offset + 4]
            
            // Find best class
            var maxClassScore: Float = 0
            var bestClass = 0
            for c in 0..<numClasses {
                let score = output[offset + 5 + c]
                if score > maxClassScore {
                    maxClassScore = score
                    bestClass = c
                }
            }
            
            // Final confidence = objectness * class_score
            let confidence = objectness * maxClassScore
            
            // Convert center to top-left
            let x = cx - w / 2
            let y = cy - h / 2
            
            detections.append(RawDetection(
                box: [x, y, w, h],
                classIndex: bestClass,
                confidence: confidence
            ))
        }
        
        return detections
    }
    
    /// Parse YOLOv8 output: [1, 4+numClasses, N] - needs transpose
    private static func parseYOLOv8(_ output: [Float], numClasses: Int) -> [RawDetection] {
        let channels = 4 + numClasses
        let numDetections = output.count / channels
        
        // Transpose from [channels, N] to [N, channels]
        var transposed = [Float](repeating: 0, count: output.count)
        for c in 0..<channels {
            for n in 0..<numDetections {
                transposed[n * channels + c] = output[c * numDetections + n]
            }
        }
        
        return parseYOLOv8Transposed(transposed, numClasses: numClasses)
    }
    
    /// Parse YOLOv8 output already transposed: [1, N, 4+numClasses]
    private static func parseYOLOv8Transposed(_ output: [Float], numClasses: Int) -> [RawDetection] {
        let stride = 4 + numClasses  // cx, cy, w, h, class_scores... (no objectness)
        let numDetections = output.count / stride
        
        var detections: [RawDetection] = []
        detections.reserveCapacity(numDetections)
        
        for i in 0..<numDetections {
            let offset = i * stride
            guard offset + stride <= output.count else { break }
            
            let cx = output[offset]
            let cy = output[offset + 1]
            let w = output[offset + 2]
            let h = output[offset + 3]
            
            // Find best class (no objectness in v8)
            var maxClassScore: Float = 0
            var bestClass = 0
            for c in 0..<numClasses {
                let score = output[offset + 4 + c]
                if score > maxClassScore {
                    maxClassScore = score
                    bestClass = c
                }
            }
            
            // Convert center to top-left
            let x = cx - w / 2
            let y = cy - h / 2
            
            detections.append(RawDetection(
                box: [x, y, w, h],
                classIndex: bestClass,
                confidence: maxClassScore
            ))
        }
        
        return detections
    }
    
    /// Parse SSD output format
    private static func parseSSD(_ output: [Float], numClasses: Int) -> [RawDetection] {
        // SSD typically outputs: [1, num_detections, 7]
        // where each detection is: [batch_id, class_id, score, x1, y1, x2, y2]
        let stride = 7
        let numDetections = output.count / stride
        
        var detections: [RawDetection] = []
        
        for i in 0..<numDetections {
            let offset = i * stride
            guard offset + stride <= output.count else { break }
            
            let classId = Int(output[offset + 1])
            let score = output[offset + 2]
            let x1 = output[offset + 3]
            let y1 = output[offset + 4]
            let x2 = output[offset + 5]
            let y2 = output[offset + 6]
            
            // Skip background class (usually 0)
            guard classId > 0 else { continue }
            
            let w = x2 - x1
            let h = y2 - y1
            
            detections.append(RawDetection(
                box: [x1, y1, w, h],
                classIndex: classId - 1,  // Adjust for background class
                confidence: score
            ))
        }
        
        return detections
    }
    
    /// Parse EfficientDet output format
    private static func parseEfficientDet(_ output: [Float], numClasses: Int) -> [RawDetection] {
        // EfficientDet typically outputs: [1, num_detections, 7]
        // [ymin, xmin, ymax, xmax, score, class, valid]
        let stride = 7
        let numDetections = output.count / stride
        
        var detections: [RawDetection] = []
        
        for i in 0..<numDetections {
            let offset = i * stride
            guard offset + stride <= output.count else { break }
            
            let ymin = output[offset]
            let xmin = output[offset + 1]
            let ymax = output[offset + 2]
            let xmax = output[offset + 3]
            let score = output[offset + 4]
            let classId = Int(output[offset + 5])
            
            let w = xmax - xmin
            let h = ymax - ymin
            
            detections.append(RawDetection(
                box: [xmin, ymin, w, h],
                classIndex: classId,
                confidence: score
            ))
        }
        
        return detections
    }
    
    /// Parse custom format
    private static func parseCustom(
        _ output: [Float],
        boxOffset: Int,
        classOffset: Int,
        numClasses: Int,
        boxFormat: BoxFormat,
        hasObjectness: Bool
    ) -> [RawDetection] {
        let stride = max(boxOffset + 4, classOffset + numClasses + (hasObjectness ? 1 : 0))
        let numDetections = output.count / stride
        
        var detections: [RawDetection] = []
        
        for i in 0..<numDetections {
            let offset = i * stride
            guard offset + stride <= output.count else { break }
            
            // Extract box
            var box = [
                output[offset + boxOffset],
                output[offset + boxOffset + 1],
                output[offset + boxOffset + 2],
                output[offset + boxOffset + 3]
            ]
            
            // Convert to xywh if needed
            if boxFormat != .xywh {
                box = BoundingBox.convertFormat([box], from: boxFormat, to: .xywh)[0]
            }
            
            // Find best class
            let objOffset = hasObjectness ? 1 : 0
            let objectness = hasObjectness ? output[offset + classOffset] : 1.0
            
            var maxClassScore: Float = 0
            var bestClass = 0
            for c in 0..<numClasses {
                let score = output[offset + classOffset + objOffset + c]
                if score > maxClassScore {
                    maxClassScore = score
                    bestClass = c
                }
            }
            
            let confidence = objectness * maxClassScore
            
            detections.append(RawDetection(
                box: box,
                classIndex: bestClass,
                confidence: confidence
            ))
        }
        
        return detections
    }
    
    // MARK: - NMS
    
    private static func applyNMS(
        detections: [RawDetection],
        iouThreshold: Float,
        maxDetections: Int
    ) -> [RawDetection] {
        guard !detections.isEmpty else { return [] }
        
        // Sort by confidence
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        var kept: [RawDetection] = []
        
        for detection in sorted {
            var shouldKeep = true
            
            for keptDetection in kept {
                let iou = computeIoU(detection.box, keptDetection.box)
                if iou > iouThreshold {
                    shouldKeep = false
                    break
                }
            }
            
            if shouldKeep {
                kept.append(detection)
                if kept.count >= maxDetections {
                    break
                }
            }
        }
        
        return kept
    }
    
    private static func computeIoU(_ box1: [Float], _ box2: [Float]) -> Float {
        // Boxes are in [x, y, w, h] format
        let x1_1 = box1[0]
        let y1_1 = box1[1]
        let x2_1 = box1[0] + box1[2]
        let y2_1 = box1[1] + box1[3]
        
        let x1_2 = box2[0]
        let y1_2 = box2[1]
        let x2_2 = box2[0] + box2[2]
        let y2_2 = box2[1] + box2[3]
        
        let intersectX1 = max(x1_1, x1_2)
        let intersectY1 = max(y1_1, y1_2)
        let intersectX2 = min(x2_1, x2_2)
        let intersectY2 = min(y2_1, y2_2)
        
        let intersectWidth = max(0, intersectX2 - intersectX1)
        let intersectHeight = max(0, intersectY2 - intersectY1)
        let intersection = intersectWidth * intersectHeight
        
        let area1 = box1[2] * box1[3]
        let area2 = box2[2] * box2[3]
        let union = area1 + area2 - intersection
        
        return union > 0 ? intersection / union : 0
    }
    
    // MARK: - Label Mapping
    
    private static func getLabel(for index: Int, source: LabelSource) -> String {
        switch source {
        case .coco:
            return LabelDatabase.getLabel(index, dataset: .coco) ?? "Unknown (\(index))"
        case .voc:
            // VOC labels include background at index 0, so add 1 to skip it for detection models
            // that don't include background class
            return LabelDatabase.getLabel(index + 1, dataset: .voc) ?? "Unknown (\(index))"
        case .vocWithBackground:
            return LabelDatabase.getLabel(index, dataset: .voc) ?? "Unknown (\(index))"
        case .custom(let labels):
            return labels[safe: index] ?? "Unknown (\(index))"
        case .none:
            return "Class \(index)"
        }
    }
}

// MARK: - Array Safe Subscript

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
