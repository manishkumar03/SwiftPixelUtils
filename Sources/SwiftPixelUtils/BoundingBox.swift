//
//  BoundingBox.swift
//  SwiftPixelUtils
//
//  Bounding box utilities for object detection
//

import Foundation
import CoreGraphics

/// Comprehensive utilities for working with bounding boxes in object detection pipelines.
///
/// ## Overview
///
/// Object detection models output bounding boxes in various formats. This module provides
/// tools for format conversion, scaling, clipping, overlap calculation, and filtering.
///
/// ## Bounding Box Formats
///
/// | Format | Components | Description |
/// |--------|------------|-------------|
/// | **XYXY** | [x1, y1, x2, y2] | Top-left and bottom-right corners |
/// | **XYWH** | [x, y, w, h] | Top-left corner plus width and height |
/// | **CXCYWH** | [cx, cy, w, h] | Center point plus width and height |
///
/// ### Format Selection by Model
///
/// - **YOLO**: Uses CXCYWH (center-based) - natural for anchor-based detection
/// - **COCO dataset**: Uses XYWH - common annotation format
/// - **Pascal VOC**: Uses XYXY - simplest for IoU calculation
/// - **SSD/Faster R-CNN**: Often uses CXCYWH internally, outputs XYXY
///
/// ## Common Operations
///
/// ### Post-Processing Pipeline
///
/// ```swift
/// // Typical detection post-processing:
/// let rawBoxes: [[Float]] = model.detect(image)
///
/// // 1. Convert format (if needed)
/// let xyxyBoxes = BoundingBox.convertFormat(rawBoxes, from: .cxcywh, to: .xyxy)
///
/// // 2. Scale to original image size
/// let scaledBoxes = BoundingBox.scale(xyxyBoxes, from: modelSize, to: originalSize, format: .xyxy)
///
/// // 3. Clip to image boundaries
/// let clippedBoxes = BoundingBox.clip(scaledBoxes, imageSize: originalSize, format: .xyxy)
///
/// // 4. Apply NMS to remove duplicates
/// let finalDetections = BoundingBox.nonMaxSuppression(detections: detections)
/// ```
public enum BoundingBox {
    
    // MARK: - Format Conversion
    
    /// Converts bounding boxes between different coordinate formats.
    ///
    /// ## Format Definitions
    ///
    /// ### XYXY (Corner Format)
    /// ```
    /// [x1, y1, x2, y2] where:
    /// - (x1, y1) = top-left corner
    /// - (x2, y2) = bottom-right corner
    /// ```
    /// **Advantages**: Direct IoU calculation, natural for clipping.
    ///
    /// ### XYWH (Top-Left + Size)
    /// ```
    /// [x, y, w, h] where:
    /// - (x, y) = top-left corner
    /// - w = width, h = height
    /// ```
    /// **Advantages**: Natural for annotations, common dataset format (COCO).
    ///
    /// ### CXCYWH (Center + Size)
    /// ```
    /// [cx, cy, w, h] where:
    /// - (cx, cy) = center point
    /// - w = width, h = height
    /// ```
    /// **Advantages**: Natural for anchor-based models, symmetric representation.
    ///
    /// ## Conversion Formulas
    ///
    /// ### XYWH → XYXY
    /// ```
    /// x2 = x1 + w
    /// y2 = y1 + h
    /// ```
    ///
    /// ### CXCYWH → XYXY
    /// ```
    /// x1 = cx - w/2
    /// y1 = cy - h/2
    /// x2 = cx + w/2
    /// y2 = cy + h/2
    /// ```
    ///
    /// ### XYXY → CXCYWH
    /// ```
    /// w = x2 - x1
    /// h = y2 - y1
    /// cx = x1 + w/2
    /// cy = y1 + h/2
    /// ```
    ///
    /// - Parameters:
    ///   - boxes: Array of boxes in source format (each box is [Float] with 4 elements)
    ///   - sourceFormat: The coordinate format of the input boxes
    ///   - targetFormat: The desired output coordinate format
    /// - Returns: Boxes converted to the target format
    public static func convertFormat(
        _ boxes: [[Float]],
        from sourceFormat: BoxFormat,
        to targetFormat: BoxFormat
    ) -> [[Float]] {
        if sourceFormat == targetFormat {
            return boxes
        }
        
        return boxes.map { box in
            convertSingleBox(box, from: sourceFormat, to: targetFormat)
        }
    }
    
    private static func convertSingleBox(
        _ box: [Float],
        from sourceFormat: BoxFormat,
        to targetFormat: BoxFormat
    ) -> [Float] {
        // First convert to xyxy as intermediate format
        let xyxy = toXYXY(box, from: sourceFormat)
        
        // Then convert from xyxy to target format
        switch targetFormat {
        case .xyxy:
            return xyxy
        case .xywh:
            return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
        case .cxcywh:
            let w = xyxy[2] - xyxy[0]
            let h = xyxy[3] - xyxy[1]
            let cx = xyxy[0] + w / 2
            let cy = xyxy[1] + h / 2
            return [cx, cy, w, h]
        }
    }
    
    private static func toXYXY(_ box: [Float], from format: BoxFormat) -> [Float] {
        switch format {
        case .xyxy:
            return box
        case .xywh:
            return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        case .cxcywh:
            let halfW = box[2] / 2
            let halfH = box[3] / 2
            return [box[0] - halfW, box[1] - halfH, box[0] + halfW, box[1] + halfH]
        }
    }
    
    // MARK: - Scaling
    
    /// Scale bounding boxes between different image dimensions
    /// - Parameters:
    ///   - boxes: Array of boxes
    ///   - from: Source image size
    ///   - to: Target image size
    ///   - format: Box format
    /// - Returns: Scaled boxes
    public static func scale(
        _ boxes: [[Float]],
        from sourceSize: CGSize,
        to targetSize: CGSize,
        format: BoxFormat
    ) -> [[Float]] {
        let scaleX = Float(targetSize.width / sourceSize.width)
        let scaleY = Float(targetSize.height / sourceSize.height)
        
        return boxes.map { box in
            switch format {
            case .xyxy:
                return [
                    box[0] * scaleX,
                    box[1] * scaleY,
                    box[2] * scaleX,
                    box[3] * scaleY
                ]
            case .xywh, .cxcywh:
                return [
                    box[0] * scaleX,
                    box[1] * scaleY,
                    box[2] * scaleX,
                    box[3] * scaleY
                ]
            }
        }
    }
    
    // MARK: - Clipping
    
    /// Clip bounding boxes to image boundaries
    /// - Parameters:
    ///   - boxes: Array of boxes in xyxy format
    ///   - imageSize: Image dimensions
    ///   - format: Box format
    /// - Returns: Clipped boxes
    public static func clip(
        _ boxes: [[Float]],
        imageSize: CGSize,
        format: BoxFormat
    ) -> [[Float]] {
        let width = Float(imageSize.width)
        let height = Float(imageSize.height)
        
        return boxes.map { box in
            let xyxy = toXYXY(box, from: format)
            let clipped = [
                max(0, min(width, xyxy[0])),
                max(0, min(height, xyxy[1])),
                max(0, min(width, xyxy[2])),
                max(0, min(height, xyxy[3]))
            ]
            return convertSingleBox(clipped, from: .xyxy, to: format)
        }
    }
    
    // MARK: - IoU Calculation
    
    /// Calculates Intersection over Union (IoU) between two bounding boxes.
    ///
    /// ## The Jaccard Index
    ///
    /// IoU (also known as the Jaccard Index or Jaccard Similarity Coefficient) measures
    /// the overlap between two sets. For bounding boxes, it's the ratio of their
    /// intersection area to their union area:
    ///
    /// ```
    ///           Area(A ∩ B)
    /// IoU(A,B) = ───────────
    ///           Area(A ∪ B)
    ///
    ///           Intersection Area
    ///         = ──────────────────────────────────────────
    ///           Area(A) + Area(B) - Intersection Area
    /// ```
    ///
    /// ## Value Interpretation
    ///
    /// | IoU Range | Interpretation |
    /// |-----------|----------------|
    /// | 0.0 | No overlap - boxes are completely separate |
    /// | 0.0 - 0.3 | Poor overlap - different objects |
    /// | 0.3 - 0.5 | Partial overlap - possibly same object |
    /// | 0.5 - 0.7 | Good overlap - likely same object |
    /// | 0.7 - 0.9 | Strong overlap - very likely same object |
    /// | 0.9 - 1.0 | Near-perfect match |
    /// | 1.0 | Perfect match - identical boxes |
    ///
    /// ## Geometric Calculation
    ///
    /// For two boxes A and B in XYXY format:
    ///
    /// ```
    /// Intersection coordinates:
    ///   x1_int = max(A.x1, B.x1)  // left edge of intersection
    ///   y1_int = max(A.y1, B.y1)  // top edge of intersection
    ///   x2_int = min(A.x2, B.x2)  // right edge of intersection
    ///   y2_int = min(A.y2, B.y2)  // bottom edge of intersection
    ///
    /// Intersection area:
    ///   width = max(0, x2_int - x1_int)  // 0 if no horizontal overlap
    ///   height = max(0, y2_int - y1_int) // 0 if no vertical overlap
    ///   area = width × height
    /// ```
    ///
    /// ## Use in Object Detection
    ///
    /// - **Training**: IoU determines positive/negative anchor assignments
    /// - **Evaluation**: mAP (mean Average Precision) uses IoU thresholds (typically 0.5 or 0.75)
    /// - **NMS**: IoU threshold determines when boxes are "duplicate" detections
    ///
    /// ## Visual Example
    ///
    /// ```
    /// ┌─────────────────┐
    /// │        A        │
    /// │    ┌───────┬────┘
    /// │    │   ∩   │
    /// └────┼───────┤
    ///      │   B   │
    ///      └───────┘
    ///
    /// IoU = Area(∩) / [Area(A) + Area(B) - Area(∩)]
    /// ```
    ///
    /// - Parameters:
    ///   - box1: First bounding box
    ///   - box2: Second bounding box
    ///   - format: Coordinate format of both boxes
    /// - Returns: IoU value in range [0, 1] where 1 means identical boxes
    public static func calculateIoU(
        _ box1: [Float],
        _ box2: [Float],
        format: BoxFormat
    ) -> Float {
        let xyxy1 = toXYXY(box1, from: format)
        let xyxy2 = toXYXY(box2, from: format)
        
        // Calculate intersection
        let x1 = max(xyxy1[0], xyxy2[0])
        let y1 = max(xyxy1[1], xyxy2[1])
        let x2 = min(xyxy1[2], xyxy2[2])
        let y2 = min(xyxy1[3], xyxy2[3])
        
        let intersectionArea = max(0, x2 - x1) * max(0, y2 - y1)
        
        // Calculate union
        let box1Area = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
        let box2Area = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
        let unionArea = box1Area + box2Area - intersectionArea
        
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }
    
    // MARK: - Non-Maximum Suppression
    
    /// Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
    ///
    /// ## Why NMS is Necessary
    ///
    /// Object detectors often produce multiple overlapping bounding boxes for the same
    /// object. This happens because:
    ///
    /// 1. **Sliding window/anchor redundancy**: Multiple anchors may fire for one object
    /// 2. **Multi-scale detection**: Same object detected at different scales
    /// 3. **Feature map overlap**: Adjacent cells may both detect the same object
    ///
    /// NMS filters these redundant detections, keeping only the best box per object.
    ///
    /// ## Greedy NMS Algorithm
    ///
    /// ```
    /// 1. Sort detections by confidence score (descending)
    /// 2. Select the highest-scoring detection, add to output
    /// 3. Remove all detections with IoU > threshold (same class only)
    /// 4. Repeat steps 2-3 until no detections remain
    /// ```
    ///
    /// ## Algorithm Visualization
    ///
    /// ```
    /// Before NMS:                 After NMS:
    /// ┌─────────────────┐        ┌─────────────────┐
    /// │ ┌───────────┐   │        │                 │
    /// │ │ ┌───────┐ │   │        │  ┌───────────┐  │
    /// │ │ │ DOG   │ │   │   →    │  │   DOG     │  │
    /// │ │ └───────┘ │   │        │  └───────────┘  │
    /// │ └───────────┘   │        │                 │
    /// └─────────────────┘        └─────────────────┘
    /// (3 overlapping boxes)       (1 best box kept)
    /// ```
    ///
    /// ## Threshold Selection
    ///
    /// ### IoU Threshold (default: 0.5)
    /// - **Lower (0.3-0.4)**: Aggressive suppression, fewer detections
    /// - **Higher (0.6-0.7)**: Lenient suppression, may keep duplicate detections
    ///
    /// | Use Case | Recommended IoU |
    /// |----------|----------------|
    /// | Dense scenes | 0.3-0.4 |
    /// | Standard detection | 0.5 |
    /// | Overlapping objects | 0.6-0.7 |
    ///
    /// ### Score Threshold (default: 0.0)
    /// - Pre-filters low-confidence detections before NMS
    /// - **Typical values**: 0.25-0.5 for most detectors
    /// - **Higher values**: Reduce false positives, may miss objects
    ///
    /// ## Per-Class vs Global NMS
    ///
    /// This implementation uses **per-class NMS**:
    /// - Only suppresses boxes of the same class
    /// - A dog box won't suppress an overlapping cat box
    /// - Standard for multi-class detection (YOLO, SSD, Faster R-CNN)
    ///
    /// ## Complexity
    ///
    /// - **Time**: O(N² × C) where N = detections, C = classes
    /// - **Space**: O(N) for tracking suppressed indices
    ///
    /// - Parameters:
    ///   - detections: Array of ``Detection`` objects with boxes and scores
    ///   - iouThreshold: Boxes with IoU above this are suppressed (default: 0.5)
    ///   - scoreThreshold: Minimum confidence to consider (default: 0.0, accepts all)
    ///   - maxDetections: Optional limit on output detections (nil = no limit)
    /// - Returns: Filtered detections with duplicates removed, sorted by score
    public static func nonMaxSuppression(
        detections: [Detection],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.0,
        maxDetections: Int? = nil
    ) -> [Detection] {
        // Filter by score threshold
        var filtered = detections.filter { $0.score >= scoreThreshold }
        
        // Sort by score descending
        filtered.sort { $0.score > $1.score }
        
        var kept: [Detection] = []
        var indices = Set<Int>(0..<filtered.count)
        
        for i in 0..<filtered.count {
            if !indices.contains(i) {
                continue
            }
            
            let detection = filtered[i]
            kept.append(detection)
            
            // Check if we've reached max detections
            if let maxDetections = maxDetections, kept.count >= maxDetections {
                break
            }
            
            // Suppress overlapping boxes
            for j in (i+1)..<filtered.count {
                if !indices.contains(j) {
                    continue
                }
                
                let other = filtered[j]
                
                // Only suppress boxes of the same class
                if detection.classIndex == other.classIndex {
                    let iou = calculateIoU(detection.box, other.box, format: .xyxy)
                    if iou > iouThreshold {
                        indices.remove(j)
                    }
                }
            }
        }
        
        return kept
    }
}
