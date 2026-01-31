//
//  TensorOperations.swift
//  SwiftPixelUtils
//
//  Tensor manipulation utilities for ML pipelines
//

import Foundation
import Accelerate

/// Tensor manipulation utilities for ML pipelines.
///
/// ## Overview
///
/// This module provides common tensor operations needed when preparing data for
/// ML inference or processing model outputs. Operations include channel extraction,
/// dimension permutation, and batch assembly.
///
/// ## Tensor Terminology
///
/// | Term | Meaning | Example |
/// |------|---------|---------|
/// | **Shape** | Dimensions of the tensor | [1, 3, 224, 224] |
/// | **Rank** | Number of dimensions | 4 |
/// | **Layout** | Order of dimensions | NCHW, NHWC |
/// | **Stride** | Elements between adjacent values in each dimension | |
///
/// ## Common Operations
///
/// ### Channel Extraction
/// Extract individual color channels for channel-wise processing:
/// ```swift
/// let redChannel = TensorOperations.extractChannel(data: pixels, channelIndex: 0, ...)
/// ```
///
/// ### Dimension Permutation
/// Reorder dimensions (e.g., HWC to CHW):
/// ```swift
/// let chw = TensorOperations.permute(data: hwc, shape: [224, 224, 3], order: [2, 0, 1])
/// ```
///
/// ### Batch Assembly
/// Combine multiple images into a batch tensor:
/// ```swift
/// let batch = try TensorOperations.assembleBatch(results: [img1, img2], layout: .nchw)
/// ```
public enum TensorOperations {
    
    // MARK: - Channel Extraction
    
    /// Extracts a single channel from multi-channel pixel data.
    ///
    /// ## Use Cases
    ///
    /// - Extract grayscale from RGB (use channel 0 after grayscale conversion)
    /// - Process individual color channels separately
    /// - Extract alpha channel for masking
    ///
    /// ## Memory Layout
    ///
    /// For HWC layout (interleaved):
    /// ```
    /// Input:  [R₀, G₀, B₀, R₁, G₁, B₁, ...]
    /// Output: [R₀, R₁, R₂, ...]  (for channel 0)
    /// ```
    ///
    /// For CHW layout (planar):
    /// ```
    /// Input:  [R₀, R₁, ..., G₀, G₁, ..., B₀, B₁, ...]
    /// Output: [R₀, R₁, ...]  (for channel 0, just a slice)
    /// ```
    ///
    /// - Parameters:
    ///   - data: Input pixel data
    ///   - width: Image width
    ///   - height: Image height
    ///   - channels: Number of channels in input
    ///   - channelIndex: Channel to extract (0-based)
    ///   - dataLayout: Memory layout of input data
    /// - Returns: Single-channel data array
    /// - Throws: ``PixelUtilsError/invalidChannel(_:)`` if channelIndex is out of range
    public static func extractChannel(
        data: [Float],
        width: Int,
        height: Int,
        channels: Int,
        channelIndex: Int,
        dataLayout: DataLayout
    ) throws -> [Float] {
        guard channelIndex >= 0 && channelIndex < channels else {
            throw PixelUtilsError.invalidChannel("Channel index \(channelIndex) out of range [0, \(channels - 1)]")
        }
        
        let pixelCount = width * height
        var result = [Float](repeating: 0, count: pixelCount)
        
        switch dataLayout {
        case .hwc, .nhwc:
            // Interleaved: R₀G₀B₀R₁G₁B₁...
            for i in 0..<pixelCount {
                result[i] = data[i * channels + channelIndex]
            }
            
        case .chw, .nchw:
            // Planar: R₀R₁R₂...G₀G₁G₂...B₀B₁B₂...
            let channelOffset = channelIndex * pixelCount
            for i in 0..<pixelCount {
                result[i] = data[channelOffset + i]
            }
        }
        
        return result
    }
    
    // MARK: - Patch Extraction
    
    /// Extracts a rectangular patch from pixel data.
    ///
    /// ## Use Cases
    ///
    /// - Sliding window detection
    /// - Region-of-interest extraction
    /// - Patch-based processing pipelines
    ///
    /// ## Memory Layout Handling
    ///
    /// For HWC layout (interleaved):
    /// ```
    /// Input:  [R₀₀, G₀₀, B₀₀, R₀₁, G₀₁, B₀₁, ...]  // row by row
    /// Output: [Rᵢⱼ, Gᵢⱼ, Bᵢⱼ, ...]  // just the patch region
    /// ```
    ///
    /// For CHW layout (planar):
    /// ```
    /// Input:  [R₀₀, R₀₁, ..., G₀₀, G₀₁, ..., B₀₀, B₀₁, ...]
    /// Output: [Rᵢⱼ...channel, Gᵢⱼ...channel, Bᵢⱼ...channel]
    /// ```
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Extract a 64x64 patch starting at (100, 100)
    /// let patch = try TensorOperations.extractPatch(
    ///     data: pixels,
    ///     width: 640,
    ///     height: 480,
    ///     channels: 3,
    ///     patchOptions: PatchOptions(x: 100, y: 100, width: 64, height: 64),
    ///     dataLayout: .hwc
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - data: Input pixel data
    ///   - width: Image width
    ///   - height: Image height
    ///   - channels: Number of channels in input
    ///   - patchOptions: Region to extract (x, y, width, height)
    ///   - dataLayout: Memory layout of input data
    /// - Returns: ``PatchResult`` containing patch data and dimensions
    /// - Throws: ``PixelUtilsError/invalidPatch(_:)`` if patch is out of bounds
    public static func extractPatch(
        data: [Float],
        width: Int,
        height: Int,
        channels: Int,
        patchOptions: PatchOptions,
        dataLayout: DataLayout
    ) throws -> PatchResult {
        let px = patchOptions.x
        let py = patchOptions.y
        let pw = patchOptions.width
        let ph = patchOptions.height
        
        // Validate patch bounds
        guard px >= 0 && py >= 0 && pw > 0 && ph > 0 else {
            throw PixelUtilsError.invalidPatch("Patch position and dimensions must be non-negative")
        }
        
        guard px + pw <= width && py + ph <= height else {
            throw PixelUtilsError.invalidPatch("Patch (\(px), \(py), \(pw), \(ph)) exceeds image bounds (\(width)x\(height))")
        }
        
        let patchPixelCount = pw * ph
        let totalOutputSize = patchPixelCount * channels
        var result = [Float](repeating: 0, count: totalOutputSize)
        
        switch dataLayout {
        case .hwc, .nhwc:
            // Interleaved: copy row by row, including all channels
            var outIdx = 0
            for row in py..<(py + ph) {
                for col in px..<(px + pw) {
                    let srcIdx = (row * width + col) * channels
                    for c in 0..<channels {
                        result[outIdx] = data[srcIdx + c]
                        outIdx += 1
                    }
                }
            }
            
        case .chw, .nchw:
            // Planar: copy each channel's patch separately
            let imagePixelCount = width * height
            for c in 0..<channels {
                let channelOffset = c * imagePixelCount
                let outChannelOffset = c * patchPixelCount
                var outIdx = outChannelOffset
                for row in py..<(py + ph) {
                    for col in px..<(px + pw) {
                        let srcIdx = channelOffset + row * width + col
                        result[outIdx] = data[srcIdx]
                        outIdx += 1
                    }
                }
            }
        }
        
        // Determine output shape
        let shape: [Int]
        switch dataLayout {
        case .hwc:
            shape = [ph, pw, channels]
        case .chw:
            shape = [channels, ph, pw]
        case .nhwc:
            shape = [1, ph, pw, channels]
        case .nchw:
            shape = [1, channels, ph, pw]
        }
        
        return PatchResult(
            data: result,
            width: pw,
            height: ph,
            channels: channels,
            shape: shape,
            x: px,
            y: py
        )
    }
    
    // MARK: - Dimension Permutation
    
    /// Permutes (transposes) tensor dimensions according to the specified order.
    ///
    /// ## Common Permutations
    ///
    /// | From | To | Order |
    /// |------|-----|-------|
    /// | HWC [H, W, C] | CHW [C, H, W] | [2, 0, 1] |
    /// | CHW [C, H, W] | HWC [H, W, C] | [1, 2, 0] |
    /// | NHWC [N, H, W, C] | NCHW [N, C, H, W] | [0, 3, 1, 2] |
    /// | NCHW [N, C, H, W] | NHWC [N, H, W, C] | [0, 2, 3, 1] |
    ///
    /// ## Algorithm
    ///
    /// For each element in the output:
    /// 1. Compute output multi-dimensional index from flat index
    /// 2. Apply inverse permutation to get input index
    /// 3. Compute input flat index
    /// 4. Copy value
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Convert HWC to CHW
    /// let hwcData = [r00, g00, b00, r01, g01, b01, ...]  // 2x2x3
    /// let chwData = TensorOperations.permute(
    ///     data: hwcData,
    ///     shape: [2, 2, 3],  // H, W, C
    ///     order: [2, 0, 1]   // C, H, W
    /// )
    /// // Result shape: [3, 2, 2]
    /// // chwData = [r00, r01, r10, r11, g00, g01, g10, g11, b00, b01, b10, b11]
    /// ```
    ///
    /// - Parameters:
    ///   - data: Input tensor data (flattened)
    ///   - shape: Shape of the input tensor
    ///   - order: New dimension order (e.g., [2, 0, 1] for HWC→CHW)
    /// - Returns: Permuted tensor data with reordered dimensions
    /// - Throws: ``PixelUtilsError/dimensionMismatch(_:)`` if order doesn't match shape rank
    public static func permute(
        data: [Float],
        shape: [Int],
        order: [Int]
    ) throws -> PermuteResult {
        guard order.count == shape.count else {
            throw PixelUtilsError.dimensionMismatch("Order \(order) doesn't match shape rank \(shape.count)")
        }
        
        guard Set(order) == Set(0..<shape.count) else {
            throw PixelUtilsError.invalidOptions("Order must be a permutation of [0, \(shape.count - 1)]")
        }
        
        // Compute output shape
        let outputShape = order.map { shape[$0] }
        
        // Compute strides for input
        var inputStrides = [Int](repeating: 1, count: shape.count)
        for i in (0..<shape.count - 1).reversed() {
            inputStrides[i] = inputStrides[i + 1] * shape[i + 1]
        }
        
        // Compute strides for output
        var outputStrides = [Int](repeating: 1, count: outputShape.count)
        for i in (0..<outputShape.count - 1).reversed() {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }
        
        let totalElements = shape.reduce(1, *)
        var result = [Float](repeating: 0, count: totalElements)
        
        // Permute
        for outIdx in 0..<totalElements {
            // Convert flat output index to multi-dimensional
            var outCoords = [Int](repeating: 0, count: outputShape.count)
            var remaining = outIdx
            for d in 0..<outputShape.count {
                outCoords[d] = remaining / outputStrides[d]
                remaining %= outputStrides[d]
            }
            
            // Apply inverse permutation to get input coordinates
            var inCoords = [Int](repeating: 0, count: shape.count)
            for d in 0..<order.count {
                inCoords[order[d]] = outCoords[d]
            }
            
            // Convert input coordinates to flat index
            var inIdx = 0
            for d in 0..<shape.count {
                inIdx += inCoords[d] * inputStrides[d]
            }
            
            result[outIdx] = data[inIdx]
        }
        
        return PermuteResult(data: result, shape: outputShape)
    }
    
    // MARK: - Batch Assembly
    
    /// Assembles multiple pixel data results into a batch tensor.
    ///
    /// ## Batch Tensors
    ///
    /// ML models often process multiple images simultaneously for efficiency.
    /// This function combines multiple single-image results into a batched tensor.
    ///
    /// ## Output Shapes
    ///
    /// | Layout | Single Image | Batch of N |
    /// |--------|--------------|------------|
    /// | NHWC | [1, H, W, C] | [N, H, W, C] |
    /// | NCHW | [1, C, H, W] | [N, C, H, W] |
    ///
    /// ## Padding
    ///
    /// If `padToSize` is specified and greater than the number of inputs,
    /// zero-padding is added to reach the target batch size. This is useful
    /// for models that require fixed batch sizes.
    ///
    /// - Parameters:
    ///   - results: Array of ``PixelDataResult`` to combine
    ///   - layout: Target data layout (nhwc or nchw)
    ///   - padToSize: Optional target batch size (pads with zeros if needed)
    /// - Returns: ``BatchResult`` with combined data and batch shape
    /// - Throws: ``PixelUtilsError/emptyBatch(_:)`` if results is empty
    /// - Throws: ``PixelUtilsError/dimensionMismatch(_:)`` if results have different dimensions
    public static func assembleBatch(
        results: [PixelDataResult],
        layout: DataLayout = .nchw,
        padToSize: Int? = nil
    ) throws -> BatchResult {
        guard !results.isEmpty else {
            throw PixelUtilsError.emptyBatch("Cannot create batch from empty results array")
        }
        
        // Verify all results have same dimensions
        let first = results[0]
        for (i, result) in results.enumerated() {
            if result.width != first.width || result.height != first.height || result.channels != first.channels {
                throw PixelUtilsError.dimensionMismatch(
                    "Result \(i) has dimensions (\(result.width), \(result.height), \(result.channels)) " +
                    "but expected (\(first.width), \(first.height), \(first.channels))"
                )
            }
        }
        
        let batchSize = padToSize ?? results.count
        let actualBatchSize = max(batchSize, results.count)
        let elementsPerImage = first.width * first.height * first.channels
        
        // Allocate output
        var batchData = [Float](repeating: 0, count: actualBatchSize * elementsPerImage)
        
        // Copy data
        for (i, result) in results.enumerated() {
            let offset = i * elementsPerImage
            
            // Convert layout if needed
            let convertedData: [Float]
            if result.dataLayout != layout {
                switch (result.dataLayout, layout) {
                case (.hwc, .chw), (.hwc, .nchw), (.nhwc, .chw), (.nhwc, .nchw):
                    convertedData = convertHWCtoCHW(result.data, width: result.width, height: result.height, channels: result.channels)
                case (.chw, .hwc), (.chw, .nhwc), (.nchw, .hwc), (.nchw, .nhwc):
                    convertedData = convertCHWtoHWC(result.data, width: result.width, height: result.height, channels: result.channels)
                default:
                    convertedData = result.data
                }
            } else {
                convertedData = result.data
            }
            
            for j in 0..<elementsPerImage {
                batchData[offset + j] = convertedData[j]
            }
        }
        
        // Compute output shape
        let shape: [Int]
        switch layout {
        case .nhwc:
            shape = [actualBatchSize, first.height, first.width, first.channels]
        case .nchw:
            shape = [actualBatchSize, first.channels, first.height, first.width]
        case .hwc:
            shape = [actualBatchSize, first.height, first.width, first.channels]
        case .chw:
            shape = [actualBatchSize, first.channels, first.height, first.width]
        }
        
        return BatchResult(
            data: batchData,
            shape: shape,
            batchSize: actualBatchSize,
            layout: layout
        )
    }
    
    // MARK: - Reshape
    
    /// Reshapes tensor data to a new shape.
    ///
    /// The total number of elements must remain the same.
    ///
    /// - Parameters:
    ///   - data: Input tensor data
    ///   - fromShape: Original shape
    ///   - toShape: Target shape
    /// - Returns: Reshaped data (same underlying data, new logical shape)
    /// - Throws: ``PixelUtilsError/dimensionMismatch(_:)`` if element counts don't match
    public static func reshape(
        data: [Float],
        fromShape: [Int],
        toShape: [Int]
    ) throws -> ReshapeResult {
        let fromCount = fromShape.reduce(1, *)
        let toCount = toShape.reduce(1, *)
        
        guard fromCount == toCount else {
            throw PixelUtilsError.dimensionMismatch(
                "Cannot reshape from \(fromShape) (\(fromCount) elements) to \(toShape) (\(toCount) elements)"
            )
        }
        
        guard data.count == fromCount else {
            throw PixelUtilsError.dimensionMismatch(
                "Data has \(data.count) elements but shape implies \(fromCount)"
            )
        }
        
        // Data doesn't change, only the logical interpretation
        return ReshapeResult(data: data, shape: toShape)
    }
    
    // MARK: - Squeeze / Unsqueeze
    
    /// Removes dimensions of size 1 from the tensor shape.
    ///
    /// - Parameters:
    ///   - shape: Input shape
    ///   - dims: Specific dimensions to squeeze (nil = all size-1 dims)
    /// - Returns: Squeezed shape
    public static func squeeze(shape: [Int], dims: [Int]? = nil) -> [Int] {
        if let dims = dims {
            var result = shape
            // Remove in reverse order to maintain indices
            for dim in dims.sorted().reversed() {
                if dim < result.count && result[dim] == 1 {
                    result.remove(at: dim)
                }
            }
            return result
        } else {
            return shape.filter { $0 != 1 }
        }
    }
    
    /// Adds a dimension of size 1 at the specified position.
    ///
    /// - Parameters:
    ///   - shape: Input shape
    ///   - dim: Position to insert new dimension
    /// - Returns: Unsqueezed shape
    public static func unsqueeze(shape: [Int], dim: Int) -> [Int] {
        var result = shape
        let insertIdx = dim < 0 ? max(0, result.count + dim + 1) : min(dim, result.count)
        result.insert(1, at: insertIdx)
        return result
    }
    
    // MARK: - Private Helpers
    
    private static func convertHWCtoCHW(_ data: [Float], width: Int, height: Int, channels: Int) -> [Float] {
        let pixelCount = width * height
        var result = [Float](repeating: 0, count: data.count)
        
        for c in 0..<channels {
            for i in 0..<pixelCount {
                result[c * pixelCount + i] = data[i * channels + c]
            }
        }
        
        return result
    }
    
    private static func convertCHWtoHWC(_ data: [Float], width: Int, height: Int, channels: Int) -> [Float] {
        let pixelCount = width * height
        var result = [Float](repeating: 0, count: data.count)
        
        for i in 0..<pixelCount {
            for c in 0..<channels {
                result[i * channels + c] = data[c * pixelCount + i]
            }
        }
        
        return result
    }
}

// MARK: - Result Types

/// Result from permute operation.
public struct PermuteResult {
    /// Permuted tensor data
    public let data: [Float]
    /// Shape after permutation
    public let shape: [Int]
    
    public init(data: [Float], shape: [Int]) {
        self.data = data
        self.shape = shape
    }
}

/// Result from batch assembly.
public struct BatchResult {
    /// Batched tensor data
    public let data: [Float]
    /// Shape of the batch tensor
    public let shape: [Int]
    /// Number of images in the batch
    public let batchSize: Int
    /// Data layout of the batch
    public let layout: DataLayout
    
    public init(data: [Float], shape: [Int], batchSize: Int, layout: DataLayout) {
        self.data = data
        self.shape = shape
        self.batchSize = batchSize
        self.layout = layout
    }
}

/// Result from reshape operation.
public struct ReshapeResult {
    /// Reshaped tensor data (same as input)
    public let data: [Float]
    /// New shape
    public let shape: [Int]
    
    public init(data: [Float], shape: [Int]) {
        self.data = data
        self.shape = shape
    }
}

/// Options for patch extraction.
public struct PatchOptions {
    /// X coordinate of top-left corner
    public let x: Int
    /// Y coordinate of top-left corner
    public let y: Int
    /// Width of the patch
    public let width: Int
    /// Height of the patch
    public let height: Int
    
    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

/// Result from patch extraction.
public struct PatchResult {
    /// Extracted patch pixel data
    public let data: [Float]
    /// Width of the patch
    public let width: Int
    /// Height of the patch
    public let height: Int
    /// Number of channels
    public let channels: Int
    /// Shape of the patch tensor
    public let shape: [Int]
    /// X coordinate where patch was extracted
    public let x: Int
    /// Y coordinate where patch was extracted
    public let y: Int
    
    public init(data: [Float], width: Int, height: Int, channels: Int, shape: [Int], x: Int, y: Int) {
        self.data = data
        self.width = width
        self.height = height
        self.channels = channels
        self.shape = shape
        self.x = x
        self.y = y
    }
}
