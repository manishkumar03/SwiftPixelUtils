import XCTest
@testable import SwiftPixelUtils

/// Tests for ``TensorOperations`` - Tensor manipulation and transformation utilities.
///
/// ## Topics
///
/// ### Channel Extraction Tests
/// - Extract single channel from HWC/CHW layouts
/// - RGB to grayscale conversion
///
/// ### Layout Conversion Tests
/// - HWC ↔ CHW transformations
/// - NHWC ↔ NCHW with batch dimension
/// - In-place vs copy operations
///
/// ### Reshape Tests
/// - Flatten to 1D, reshape to arbitrary dimensions
/// - Batch dimension insertion/removal
///
/// ### Transpose Tests
/// - Axis permutation
/// - Memory layout optimization
final class TensorOperationsTests: XCTestCase {
    
    // MARK: - Channel Extraction Tests
    
    func testExtractChannelFromHWC() throws {
        // 2x2 RGB image in HWC format: [R, G, B, R, G, B, R, G, B, R, G, B]
        let data: [Float] = [
            1, 2, 3,    // pixel (0,0)
            4, 5, 6,    // pixel (1,0)
            7, 8, 9,    // pixel (0,1)
            10, 11, 12  // pixel (1,1)
        ]
        
        let red = try TensorOperations.extractChannel(
            data: data, width: 2, height: 2, channels: 3, 
            channelIndex: 0, dataLayout: .hwc
        )
        XCTAssertEqual(red, [1, 4, 7, 10])
        
        let green = try TensorOperations.extractChannel(
            data: data, width: 2, height: 2, channels: 3, 
            channelIndex: 1, dataLayout: .hwc
        )
        XCTAssertEqual(green, [2, 5, 8, 11])
        
        let blue = try TensorOperations.extractChannel(
            data: data, width: 2, height: 2, channels: 3, 
            channelIndex: 2, dataLayout: .hwc
        )
        XCTAssertEqual(blue, [3, 6, 9, 12])
    }
    
    func testExtractChannelFromCHW() throws {
        // 2x2 RGB image in CHW format: [R0, R1, R2, R3, G0, G1, G2, G3, B0, B1, B2, B3]
        let data: [Float] = [
            1, 4, 7, 10,  // Red channel
            2, 5, 8, 11,  // Green channel
            3, 6, 9, 12   // Blue channel
        ]
        
        let red = try TensorOperations.extractChannel(
            data: data, width: 2, height: 2, channels: 3, 
            channelIndex: 0, dataLayout: .chw
        )
        XCTAssertEqual(red, [1, 4, 7, 10])
        
        let green = try TensorOperations.extractChannel(
            data: data, width: 2, height: 2, channels: 3, 
            channelIndex: 1, dataLayout: .chw
        )
        XCTAssertEqual(green, [2, 5, 8, 11])
    }
    
    func testExtractChannelInvalidIndex() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        
        XCTAssertThrowsError(try TensorOperations.extractChannel(
            data: data, width: 2, height: 1, channels: 3,
            channelIndex: 3, dataLayout: .hwc
        )) { error in
            XCTAssertTrue(error is PixelUtilsError)
        }
        
        XCTAssertThrowsError(try TensorOperations.extractChannel(
            data: data, width: 2, height: 1, channels: 3,
            channelIndex: -1, dataLayout: .hwc
        ))
    }
    
    func testExtractChannelSingleChannel() throws {
        let data: [Float] = [1, 2, 3, 4]
        
        let channel = try TensorOperations.extractChannel(
            data: data, width: 2, height: 2, channels: 1,
            channelIndex: 0, dataLayout: .hwc
        )
        
        XCTAssertEqual(channel, [1, 2, 3, 4])
    }
    
    func testExtractChannelFromNHWC() throws {
        // Same as HWC but with batch dimension (ignored for extraction)
        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        let red = try TensorOperations.extractChannel(
            data: data, width: 2, height: 2, channels: 3,
            channelIndex: 0, dataLayout: .nhwc
        )
        
        XCTAssertEqual(red, [1, 4, 7, 10])
    }
    
    // MARK: - Patch Extraction Tests
    
    func testExtractPatchHWC() throws {
        // 4x4 RGB image
        var data = [Float]()
        for i in 0..<16 {
            data.append(contentsOf: [Float(i * 3), Float(i * 3 + 1), Float(i * 3 + 2)])
        }
        
        let patch = try TensorOperations.extractPatch(
            data: data, width: 4, height: 4, channels: 3,
            patchOptions: PatchOptions(x: 1, y: 1, width: 2, height: 2),
            dataLayout: .hwc
        )
        
        XCTAssertEqual(patch.width, 2)
        XCTAssertEqual(patch.height, 2)
        XCTAssertEqual(patch.channels, 3)
        XCTAssertEqual(patch.data.count, 2 * 2 * 3)
        XCTAssertEqual(patch.shape, [2, 2, 3])
    }
    
    func testExtractPatchCHW() throws {
        // 4x4 grayscale image in CHW format
        let data: [Float] = Array(0..<16).map { Float($0) }
        
        let patch = try TensorOperations.extractPatch(
            data: data, width: 4, height: 4, channels: 1,
            patchOptions: PatchOptions(x: 1, y: 1, width: 2, height: 2),
            dataLayout: .chw
        )
        
        XCTAssertEqual(patch.width, 2)
        XCTAssertEqual(patch.height, 2)
        XCTAssertEqual(patch.data.count, 4)
        // Extract positions (1,1), (2,1), (1,2), (2,2) from 4x4 grid
        // Row 1: indices 4,5,6,7; Row 2: indices 8,9,10,11
        // Extract (1,1)=5, (2,1)=6, (1,2)=9, (2,2)=10
        XCTAssertEqual(patch.data, [5, 6, 9, 10])
    }
    
    func testExtractPatchOutOfBounds() {
        let data = [Float](repeating: 0, count: 16 * 3)
        
        XCTAssertThrowsError(try TensorOperations.extractPatch(
            data: data, width: 4, height: 4, channels: 3,
            patchOptions: PatchOptions(x: 3, y: 0, width: 2, height: 2),
            dataLayout: .hwc
        )) { error in
            XCTAssertTrue(error is PixelUtilsError)
        }
    }
    
    func testExtractPatchNegativePosition() {
        let data = [Float](repeating: 0, count: 16 * 3)
        
        XCTAssertThrowsError(try TensorOperations.extractPatch(
            data: data, width: 4, height: 4, channels: 3,
            patchOptions: PatchOptions(x: -1, y: 0, width: 2, height: 2),
            dataLayout: .hwc
        ))
    }
    
    func testExtractPatchZeroSize() {
        let data = [Float](repeating: 0, count: 16 * 3)
        
        XCTAssertThrowsError(try TensorOperations.extractPatch(
            data: data, width: 4, height: 4, channels: 3,
            patchOptions: PatchOptions(x: 0, y: 0, width: 0, height: 2),
            dataLayout: .hwc
        ))
    }
    
    func testExtractPatchFullImage() throws {
        let data: [Float] = Array(0..<12).map { Float($0) }
        
        let patch = try TensorOperations.extractPatch(
            data: data, width: 2, height: 2, channels: 3,
            patchOptions: PatchOptions(x: 0, y: 0, width: 2, height: 2),
            dataLayout: .hwc
        )
        
        XCTAssertEqual(patch.data, data)
    }
    
    // MARK: - Permute Tests
    
    func testPermuteHWCtoCHW() throws {
        // 2x2x3 HWC tensor -> 3x2x2 CHW tensor
        let hwc: [Float] = [
            1, 2, 3,   // (0,0) RGB
            4, 5, 6,   // (1,0) RGB
            7, 8, 9,   // (0,1) RGB
            10, 11, 12 // (1,1) RGB
        ]
        
        let result = try TensorOperations.permute(
            data: hwc,
            shape: [2, 2, 3],
            order: [2, 0, 1]  // C, H, W
        )
        
        XCTAssertEqual(result.shape, [3, 2, 2])
        // Expected CHW: [R00, R10, R01, R11, G00, G10, G01, G11, B00, B10, B01, B11]
        XCTAssertEqual(result.data, [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12])
    }
    
    func testPermuteCHWtoHWC() throws {
        // 3x2x2 CHW tensor -> 2x2x3 HWC tensor
        let chw: [Float] = [
            1, 4, 7, 10,   // Red channel
            2, 5, 8, 11,   // Green channel
            3, 6, 9, 12    // Blue channel
        ]
        
        let result = try TensorOperations.permute(
            data: chw,
            shape: [3, 2, 2],
            order: [1, 2, 0]  // H, W, C
        )
        
        XCTAssertEqual(result.shape, [2, 2, 3])
        XCTAssertEqual(result.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    }
    
    func testPermuteNHWCtoNCHW() throws {
        // 1x2x2x3 NHWC -> 1x3x2x2 NCHW
        let nhwc: [Float] = Array(1...12).map { Float($0) }
        
        let result = try TensorOperations.permute(
            data: nhwc,
            shape: [1, 2, 2, 3],
            order: [0, 3, 1, 2]
        )
        
        XCTAssertEqual(result.shape, [1, 3, 2, 2])
    }
    
    func testPermuteIdentity() throws {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        
        let result = try TensorOperations.permute(
            data: data,
            shape: [2, 3],
            order: [0, 1]  // No change
        )
        
        XCTAssertEqual(result.data, data)
        XCTAssertEqual(result.shape, [2, 3])
    }
    
    func testPermuteInvalidOrder() {
        let data: [Float] = [1, 2, 3, 4]
        
        // Wrong number of dimensions in order
        XCTAssertThrowsError(try TensorOperations.permute(
            data: data,
            shape: [2, 2],
            order: [0, 1, 2]
        ))
        
        // Invalid permutation (not a permutation of [0, n-1])
        XCTAssertThrowsError(try TensorOperations.permute(
            data: data,
            shape: [2, 2],
            order: [0, 2]
        ))
    }
    
    // MARK: - Batch Assembly Tests
    
    func testAssembleBatchNCHW() throws {
        let result1 = PixelDataResult(
            data: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            width: 1, height: 3, channels: 3,
            colorFormat: .rgb,
            dataLayout: .chw,
            shape: [3, 3, 1],
            processingTimeMs: 0
        )
        
        let result2 = PixelDataResult(
            data: [10, 11, 12, 13, 14, 15, 16, 17, 18],
            width: 1, height: 3, channels: 3,
            colorFormat: .rgb,
            dataLayout: .chw,
            shape: [3, 3, 1],
            processingTimeMs: 0
        )
        
        let batch = try TensorOperations.assembleBatch(
            results: [result1, result2],
            layout: .nchw
        )
        
        XCTAssertEqual(batch.batchSize, 2)
        XCTAssertEqual(batch.shape, [2, 3, 3, 1])
        XCTAssertEqual(batch.data.count, 18)
    }
    
    func testAssembleBatchNHWC() throws {
        let result1 = PixelDataResult(
            data: [1, 2, 3, 4, 5, 6],
            width: 2, height: 1, channels: 3,
            colorFormat: .rgb,
            dataLayout: .hwc,
            shape: [1, 2, 3],
            processingTimeMs: 0
        )
        
        let result2 = PixelDataResult(
            data: [7, 8, 9, 10, 11, 12],
            width: 2, height: 1, channels: 3,
            colorFormat: .rgb,
            dataLayout: .hwc,
            shape: [1, 2, 3],
            processingTimeMs: 0
        )
        
        let batch = try TensorOperations.assembleBatch(
            results: [result1, result2],
            layout: .nhwc
        )
        
        XCTAssertEqual(batch.batchSize, 2)
        XCTAssertEqual(batch.shape, [2, 1, 2, 3])
    }
    
    func testAssembleBatchWithLayoutConversion() throws {
        // Input is HWC, output should be NCHW
        let result = PixelDataResult(
            data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            width: 2, height: 2, channels: 3,
            colorFormat: .rgb,
            dataLayout: .hwc,
            shape: [2, 2, 3],
            processingTimeMs: 0
        )
        
        let batch = try TensorOperations.assembleBatch(
            results: [result],
            layout: .nchw
        )
        
        XCTAssertEqual(batch.layout, .nchw)
        XCTAssertEqual(batch.shape, [1, 3, 2, 2])
    }
    
    func testAssembleBatchWithPadding() throws {
        let result = PixelDataResult(
            data: [1, 2, 3, 4],
            width: 2, height: 1, channels: 2,
            colorFormat: .rgb,
            dataLayout: .hwc,
            shape: [1, 2, 2],
            processingTimeMs: 0
        )
        
        let batch = try TensorOperations.assembleBatch(
            results: [result],
            layout: .nhwc,
            padToSize: 4
        )
        
        XCTAssertEqual(batch.batchSize, 4)
        // First image data + zeros for padding
        XCTAssertEqual(batch.data.prefix(4), [1, 2, 3, 4])
        XCTAssertEqual(batch.data.suffix(12), [Float](repeating: 0, count: 12))
    }
    
    func testAssembleBatchEmpty() {
        XCTAssertThrowsError(try TensorOperations.assembleBatch(results: [])) { error in
            XCTAssertTrue(error is PixelUtilsError)
        }
    }
    
    func testAssembleBatchDimensionMismatch() {
        let result1 = PixelDataResult(
            data: [1, 2, 3, 4],
            width: 2, height: 2, channels: 1,
            colorFormat: .grayscale,
            dataLayout: .hwc,
            shape: [2, 2, 1],
            processingTimeMs: 0
        )
        
        let result2 = PixelDataResult(
            data: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            width: 3, height: 3, channels: 1,
            colorFormat: .grayscale,
            dataLayout: .hwc,
            shape: [3, 3, 1],
            processingTimeMs: 0
        )
        
        XCTAssertThrowsError(try TensorOperations.assembleBatch(results: [result1, result2]))
    }
    
    // MARK: - Reshape Tests
    
    func testReshapeValid() throws {
        let data: [Float] = Array(0..<24).map { Float($0) }
        
        let result = try TensorOperations.reshape(
            data: data,
            fromShape: [2, 3, 4],
            toShape: [4, 6]
        )
        
        XCTAssertEqual(result.data, data)
        XCTAssertEqual(result.shape, [4, 6])
    }
    
    func testReshapeToHigherRank() throws {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        
        let result = try TensorOperations.reshape(
            data: data,
            fromShape: [6],
            toShape: [1, 2, 3]
        )
        
        XCTAssertEqual(result.shape, [1, 2, 3])
    }
    
    func testReshapeToLowerRank() throws {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        
        let result = try TensorOperations.reshape(
            data: data,
            fromShape: [2, 3],
            toShape: [6]
        )
        
        XCTAssertEqual(result.shape, [6])
    }
    
    func testReshapeInvalidElementCount() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        
        XCTAssertThrowsError(try TensorOperations.reshape(
            data: data,
            fromShape: [2, 3],
            toShape: [2, 4]  // 8 elements instead of 6
        ))
    }
    
    func testReshapeDataSizeMismatch() {
        let data: [Float] = [1, 2, 3, 4]
        
        XCTAssertThrowsError(try TensorOperations.reshape(
            data: data,
            fromShape: [2, 3],  // Claims 6 elements
            toShape: [6]
        ))
    }
    
    // MARK: - Squeeze Tests
    
    func testSqueezeAll() {
        let shape = [1, 3, 1, 224, 1, 224]
        let squeezed = TensorOperations.squeeze(shape: shape)
        XCTAssertEqual(squeezed, [3, 224, 224])
    }
    
    func testSqueezeSpecificDims() {
        let shape = [1, 3, 1, 224, 224]
        let squeezed = TensorOperations.squeeze(shape: shape, dims: [0])
        XCTAssertEqual(squeezed, [3, 1, 224, 224])
    }
    
    func testSqueezeMultipleDims() {
        let shape = [1, 3, 1, 224, 224]
        let squeezed = TensorOperations.squeeze(shape: shape, dims: [0, 2])
        XCTAssertEqual(squeezed, [3, 224, 224])
    }
    
    func testSqueezeNoOnesPresent() {
        let shape = [2, 3, 4]
        let squeezed = TensorOperations.squeeze(shape: shape)
        XCTAssertEqual(squeezed, [2, 3, 4])
    }
    
    func testSqueezeNonOneDimIgnored() {
        let shape = [1, 3, 1, 4]
        // Dimension 1 is 3, not 1, so it shouldn't be removed
        let squeezed = TensorOperations.squeeze(shape: shape, dims: [0, 1])
        XCTAssertEqual(squeezed, [3, 1, 4])
    }
    
    // MARK: - Unsqueeze Tests
    
    func testUnsqueezeAtBeginning() {
        let shape = [3, 224, 224]
        let unsqueezed = TensorOperations.unsqueeze(shape: shape, dim: 0)
        XCTAssertEqual(unsqueezed, [1, 3, 224, 224])
    }
    
    func testUnsqueezeAtEnd() {
        let shape = [3, 224, 224]
        let unsqueezed = TensorOperations.unsqueeze(shape: shape, dim: 3)
        XCTAssertEqual(unsqueezed, [3, 224, 224, 1])
    }
    
    func testUnsqueezeInMiddle() {
        let shape = [3, 224, 224]
        let unsqueezed = TensorOperations.unsqueeze(shape: shape, dim: 1)
        XCTAssertEqual(unsqueezed, [3, 1, 224, 224])
    }
    
    func testUnsqueezeNegativeDim() {
        let shape = [3, 224, 224]
        let unsqueezed = TensorOperations.unsqueeze(shape: shape, dim: -1)
        XCTAssertEqual(unsqueezed, [3, 224, 224, 1])
    }
    
    func testUnsqueezeEmptyShape() {
        let shape: [Int] = []
        let unsqueezed = TensorOperations.unsqueeze(shape: shape, dim: 0)
        XCTAssertEqual(unsqueezed, [1])
    }
    
    // MARK: - Result Types Tests
    
    func testPermuteResultInit() {
        let result = PermuteResult(data: [1, 2, 3], shape: [3])
        XCTAssertEqual(result.data, [1, 2, 3])
        XCTAssertEqual(result.shape, [3])
    }
    
    func testBatchResultInit() {
        let result = BatchResult(
            data: [1, 2, 3, 4],
            shape: [2, 2],
            batchSize: 2,
            layout: .nchw
        )
        XCTAssertEqual(result.batchSize, 2)
        XCTAssertEqual(result.layout, .nchw)
    }
    
    func testReshapeResultInit() {
        let result = ReshapeResult(data: [1, 2, 3, 4], shape: [2, 2])
        XCTAssertEqual(result.data, [1, 2, 3, 4])
        XCTAssertEqual(result.shape, [2, 2])
    }
    
    func testPatchOptionsInit() {
        let options = PatchOptions(x: 10, y: 20, width: 64, height: 64)
        XCTAssertEqual(options.x, 10)
        XCTAssertEqual(options.y, 20)
        XCTAssertEqual(options.width, 64)
        XCTAssertEqual(options.height, 64)
    }
    
    func testPatchResultInit() {
        let result = PatchResult(
            data: [1, 2, 3, 4],
            width: 2, height: 2, channels: 1,
            shape: [2, 2, 1],
            x: 10, y: 20
        )
        XCTAssertEqual(result.width, 2)
        XCTAssertEqual(result.height, 2)
        XCTAssertEqual(result.x, 10)
        XCTAssertEqual(result.y, 20)
    }
    
    // MARK: - Performance Tests
    
    func testExtractChannelPerformance() {
        let data = [Float](repeating: 0.5, count: 224 * 224 * 3)
        
        measure {
            let _ = try? TensorOperations.extractChannel(
                data: data, width: 224, height: 224, channels: 3,
                channelIndex: 0, dataLayout: .hwc
            )
        }
    }
    
    func testPermutePerformance() {
        let data = [Float](repeating: 0.5, count: 224 * 224 * 3)
        
        measure {
            let _ = try? TensorOperations.permute(
                data: data,
                shape: [224, 224, 3],
                order: [2, 0, 1]
            )
        }
    }
    
    func testBatchAssemblyPerformance() {
        let results = (0..<4).map { _ in
            PixelDataResult(
                data: [Float](repeating: 0.5, count: 224 * 224 * 3),
                width: 224, height: 224, channels: 3,
                colorFormat: .rgb,
                dataLayout: .chw,
                shape: [3, 224, 224],
                processingTimeMs: 0
            )
        }
        
        measure {
            let _ = try? TensorOperations.assembleBatch(results: results, layout: .nchw)
        }
    }
}
