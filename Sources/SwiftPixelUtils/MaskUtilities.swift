import Foundation
import Accelerate

// MARK: - Mask Utilities

/// Utilities for processing segmentation masks
public enum MaskUtilities {
    
    /// Result of mask processing
    public struct MaskResult {
        /// Processed mask data
        public let mask: [Float]
        /// Width of the mask
        public let width: Int
        /// Height of the mask
        public let height: Int
        /// Number of classes (for multi-class masks)
        public let numClasses: Int?
        
        public init(mask: [Float], width: Int, height: Int, numClasses: Int? = nil) {
            self.mask = mask
            self.width = width
            self.height = height
            self.numClasses = numClasses
        }
    }
    
    /// Resizes a mask using nearest neighbor interpolation.
    /// - Parameters:
    ///   - mask: Input mask values
    ///   - fromSize: Original (width, height)
    ///   - toSize: Target (width, height)
    /// - Returns: Resized mask
    public static func resizeMask(
        _ mask: [Float],
        fromSize: (width: Int, height: Int),
        toSize: (width: Int, height: Int)
    ) -> MaskResult {
        guard !mask.isEmpty else {
            return MaskResult(mask: [], width: toSize.width, height: toSize.height)
        }
        
        let scaleX = Float(fromSize.width) / Float(toSize.width)
        let scaleY = Float(fromSize.height) / Float(toSize.height)
        
        var result = [Float](repeating: 0, count: toSize.width * toSize.height)
        
        for y in 0..<toSize.height {
            for x in 0..<toSize.width {
                let srcX = min(Int(Float(x) * scaleX), fromSize.width - 1)
                let srcY = min(Int(Float(y) * scaleY), fromSize.height - 1)
                let srcIdx = srcY * fromSize.width + srcX
                let dstIdx = y * toSize.width + x
                result[dstIdx] = mask[srcIdx]
            }
        }
        
        return MaskResult(mask: result, width: toSize.width, height: toSize.height)
    }
    
    /// Resizes a mask using bilinear interpolation for smoother results.
    /// - Parameters:
    ///   - mask: Input mask values
    ///   - fromSize: Original (width, height)
    ///   - toSize: Target (width, height)
    /// - Returns: Resized mask
    public static func resizeMaskBilinear(
        _ mask: [Float],
        fromSize: (width: Int, height: Int),
        toSize: (width: Int, height: Int)
    ) -> MaskResult {
        guard !mask.isEmpty else {
            return MaskResult(mask: [], width: toSize.width, height: toSize.height)
        }
        
        let scaleX = Float(fromSize.width - 1) / Float(max(1, toSize.width - 1))
        let scaleY = Float(fromSize.height - 1) / Float(max(1, toSize.height - 1))
        
        var result = [Float](repeating: 0, count: toSize.width * toSize.height)
        
        for y in 0..<toSize.height {
            for x in 0..<toSize.width {
                let srcX = Float(x) * scaleX
                let srcY = Float(y) * scaleY
                
                let x0 = Int(srcX)
                let y0 = Int(srcY)
                let x1 = min(x0 + 1, fromSize.width - 1)
                let y1 = min(y0 + 1, fromSize.height - 1)
                
                let xFrac = srcX - Float(x0)
                let yFrac = srcY - Float(y0)
                
                let v00 = mask[y0 * fromSize.width + x0]
                let v01 = mask[y0 * fromSize.width + x1]
                let v10 = mask[y1 * fromSize.width + x0]
                let v11 = mask[y1 * fromSize.width + x1]
                
                let v0 = v00 * (1 - xFrac) + v01 * xFrac
                let v1 = v10 * (1 - xFrac) + v11 * xFrac
                let v = v0 * (1 - yFrac) + v1 * yFrac
                
                result[y * toSize.width + x] = v
            }
        }
        
        return MaskResult(mask: result, width: toSize.width, height: toSize.height)
    }
    
    /// Applies a threshold to create a binary mask.
    /// - Parameters:
    ///   - mask: Input probability mask
    ///   - threshold: Threshold value
    /// - Returns: Binary mask (0.0 or 1.0)
    public static func threshold(_ mask: [Float], threshold: Float = 0.5) -> [Float] {
        return mask.map { $0 >= threshold ? 1.0 : 0.0 }
    }
    
    /// Converts multi-class logits to class indices using argmax.
    /// - Parameters:
    ///   - logits: Flattened array of shape [height, width, numClasses]
    ///   - width: Mask width
    ///   - height: Mask height
    ///   - numClasses: Number of classes
    /// - Returns: Array of class indices for each pixel
    public static func argmaxMask(
        logits: [Float],
        width: Int,
        height: Int,
        numClasses: Int
    ) -> [Int] {
        guard logits.count == width * height * numClasses else { return [] }
        
        var result = [Int](repeating: 0, count: width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = (y * width + x) * numClasses
                var maxVal = logits[pixelOffset]
                var maxIdx = 0
                
                for c in 1..<numClasses {
                    let val = logits[pixelOffset + c]
                    if val > maxVal {
                        maxVal = val
                        maxIdx = c
                    }
                }
                
                result[y * width + x] = maxIdx
            }
        }
        
        return result
    }
    
    /// Computes the area of a binary mask.
    /// - Parameter mask: Binary mask (0s and 1s)
    /// - Returns: Number of pixels with value 1
    public static func computeArea(_ mask: [Float]) -> Int {
        return mask.reduce(0) { $0 + ($1 > 0.5 ? 1 : 0) }
    }
    
    /// Computes IoU between two binary masks.
    /// - Parameters:
    ///   - mask1: First binary mask
    ///   - mask2: Second binary mask
    /// - Returns: Intersection over Union value
    public static func maskIoU(_ mask1: [Float], _ mask2: [Float]) -> Float {
        guard mask1.count == mask2.count, !mask1.isEmpty else { return 0 }
        
        var intersection: Float = 0
        var union: Float = 0
        
        for i in 0..<mask1.count {
            let v1 = mask1[i] > 0.5 ? 1 : 0
            let v2 = mask2[i] > 0.5 ? 1 : 0
            
            if v1 == 1 && v2 == 1 {
                intersection += 1
            }
            if v1 == 1 || v2 == 1 {
                union += 1
            }
        }
        
        return union > 0 ? intersection / union : 0
    }
}
