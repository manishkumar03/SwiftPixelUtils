import Foundation

// MARK: - NMS Variants

/// Non-Maximum Suppression implementations for object detection
public enum NMSVariants {
    
    /// Mode for Soft-NMS score decay
    public enum SoftNMSMode {
        /// Linear decay: score = score * (1 - iou)
        case linear
        /// Gaussian decay: score = score * exp(-(iou^2) / sigma)
        case gaussian(sigma: Float)
    }
    
    /// Applies Soft-NMS which gradually reduces scores instead of removing boxes.
    /// - Parameters:
    ///   - boxes: Array of bounding boxes [x, y, width, height]
    ///   - scores: Confidence scores for each box
    ///   - mode: Decay mode (linear or gaussian)
    ///   - iouThreshold: IoU threshold for score modification
    ///   - scoreThreshold: Minimum score to keep a detection
    /// - Returns: Tuple of (indices of kept boxes, modified scores)
    public static func softNMS(
        boxes: [[Float]],
        scores: [Float],
        mode: SoftNMSMode = .gaussian(sigma: 0.5),
        iouThreshold: Float = 0.3,
        scoreThreshold: Float = 0.001
    ) -> (indices: [Int], scores: [Float]) {
        guard !boxes.isEmpty, boxes.count == scores.count else {
            return ([], [])
        }
        
        var modifiedScores = scores
        var indices: [Int] = []
        var remaining = Set(0..<boxes.count)
        
        while !remaining.isEmpty {
            // Find highest scoring box
            var maxScore: Float = -1
            var maxIdx = -1
            for idx in remaining {
                if modifiedScores[idx] > maxScore {
                    maxScore = modifiedScores[idx]
                    maxIdx = idx
                }
            }
            
            guard maxIdx >= 0, maxScore >= scoreThreshold else { break }
            
            indices.append(maxIdx)
            remaining.remove(maxIdx)
            
            // Update scores of remaining boxes based on IoU
            for idx in remaining {
                let iou = computeIoU(boxes[maxIdx], boxes[idx])
                if iou > iouThreshold {
                    switch mode {
                    case .linear:
                        modifiedScores[idx] *= (1 - iou)
                    case .gaussian(let sigma):
                        modifiedScores[idx] *= exp(-(iou * iou) / sigma)
                    }
                }
            }
            
            // Remove boxes that fell below threshold
            remaining = remaining.filter { modifiedScores[$0] >= scoreThreshold }
        }
        
        let finalScores = indices.map { modifiedScores[$0] }
        return (indices, finalScores)
    }
    
    /// Applies NMS separately for each class.
    /// - Parameters:
    ///   - boxes: Array of bounding boxes
    ///   - scores: 2D array where scores[i][j] is score of box i for class j
    ///   - iouThreshold: IoU threshold for suppression
    ///   - scoreThreshold: Minimum score to consider
    ///   - maxDetectionsPerClass: Maximum detections to keep per class
    /// - Returns: Array of (boxIndex, classIndex, score) tuples
    public static func perClassNMS(
        boxes: [[Float]],
        scores: [[Float]],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.5,
        maxDetectionsPerClass: Int = 100
    ) -> [(boxIndex: Int, classIndex: Int, score: Float)] {
        guard !boxes.isEmpty, !scores.isEmpty, boxes.count == scores.count else {
            return []
        }
        
        let numClasses = scores[0].count
        var results: [(boxIndex: Int, classIndex: Int, score: Float)] = []
        
        for classIdx in 0..<numClasses {
            // Get scores for this class
            let classScores = scores.map { $0[classIdx] }
            
            // Filter by score threshold
            let candidates = classScores.enumerated()
                .filter { $0.element >= scoreThreshold }
                .sorted { $0.element > $1.element }
            
            var kept: [Int] = []
            
            for (boxIdx, score) in candidates {
                var shouldKeep = true
                
                for keptIdx in kept {
                    let iou = computeIoU(boxes[boxIdx], boxes[keptIdx])
                    if iou > iouThreshold {
                        shouldKeep = false
                        break
                    }
                }
                
                if shouldKeep {
                    kept.append(boxIdx)
                    results.append((boxIdx, classIdx, score))
                    
                    if kept.count >= maxDetectionsPerClass {
                        break
                    }
                }
            }
        }
        
        // Sort by score descending
        return results.sorted { $0.score > $1.score }
    }
    
    /// Applies batched NMS for multiple images.
    /// - Parameters:
    ///   - batchBoxes: Array of box arrays, one per image
    ///   - batchScores: Array of score arrays, one per image
    ///   - iouThreshold: IoU threshold
    ///   - scoreThreshold: Score threshold
    /// - Returns: Array of kept indices arrays, one per image
    public static func batchedNMS(
        batchBoxes: [[[Float]]],
        batchScores: [[Float]],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.5
    ) -> [[Int]] {
        guard batchBoxes.count == batchScores.count else { return [] }
        
        return zip(batchBoxes, batchScores).map { boxes, scores in
            classAgnosticNMS(boxes: boxes, scores: scores, iouThreshold: iouThreshold, scoreThreshold: scoreThreshold)
        }
    }
    
    /// Standard NMS that ignores class labels.
    /// - Parameters:
    ///   - boxes: Array of bounding boxes [x, y, width, height]
    ///   - scores: Confidence scores
    ///   - iouThreshold: IoU threshold for suppression
    ///   - scoreThreshold: Minimum score to keep
    /// - Returns: Indices of kept boxes
    public static func classAgnosticNMS(
        boxes: [[Float]],
        scores: [Float],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.5
    ) -> [Int] {
        guard !boxes.isEmpty, boxes.count == scores.count else { return [] }
        
        // Sort by score descending
        let sorted = scores.enumerated()
            .filter { $0.element >= scoreThreshold }
            .sorted { $0.element > $1.element }
        
        var kept: [Int] = []
        
        for (idx, _) in sorted {
            var shouldKeep = true
            
            for keptIdx in kept {
                let iou = computeIoU(boxes[idx], boxes[keptIdx])
                if iou > iouThreshold {
                    shouldKeep = false
                    break
                }
            }
            
            if shouldKeep {
                kept.append(idx)
            }
        }
        
        return kept
    }
    
    // MARK: - Private Helpers
    
    private static func computeIoU(_ box1: [Float], _ box2: [Float]) -> Float {
        guard box1.count >= 4, box2.count >= 4 else { return 0 }
        
        // Assuming [x, y, width, height] format
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
}
