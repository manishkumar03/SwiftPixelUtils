//
//  DepthEstimationOutputTests.swift
//  SwiftPixelUtilsTests
//
//  Tests for depth estimation output processing
//

import XCTest
import CoreVideo
@testable import SwiftPixelUtils

final class DepthEstimationOutputTests: XCTestCase {
    
    // MARK: - Basic Processing Tests
    
    func testProcessDepthOutput() throws {
        // Create sample depth output (10x10)
        let width = 10
        let height = 10
        var depthOutput = [Float](repeating: 0, count: width * height)
        
        // Create gradient: closer at top, farther at bottom
        for y in 0..<height {
            for x in 0..<width {
                depthOutput[y * width + x] = Float(height - y) * 100 // MiDaS-style inverse depth
            }
        }
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: width,
            height: height,
            modelType: .midas
        )
        
        XCTAssertEqual(result.width, width)
        XCTAssertEqual(result.height, height)
        XCTAssertEqual(result.depthMap.count, width * height)
        XCTAssertTrue(result.isInverse) // MiDaS outputs inverse depth
        XCTAssertFalse(result.isMetric)
    }
    
    func testProcess2DDepthOutput() throws {
        let width = 5
        let height = 5
        
        // Create 2D array
        var depth2D = [[Float]](repeating: [Float](repeating: 0, count: width), count: height)
        for y in 0..<height {
            for x in 0..<width {
                depth2D[y][x] = Float(x + y)
            }
        }
        
        let result = try DepthEstimationOutput.process(
            output2D: depth2D,
            modelType: .dpt
        )
        
        XCTAssertEqual(result.width, width)
        XCTAssertEqual(result.height, height)
        
        // Check corner values
        XCTAssertEqual(result.depthAt(x: 0, y: 0), 0)
        XCTAssertEqual(result.depthAt(x: 4, y: 4), 8)
    }
    
    func testInvalidOutputSize() {
        let depthOutput = [Float](repeating: 0, count: 50) // Wrong size
        
        XCTAssertThrowsError(try DepthEstimationOutput.process(
            output: depthOutput,
            width: 10,
            height: 10, // Expects 100 elements
            modelType: .midas
        ))
    }
    
    func testEmptyOutput() {
        XCTAssertThrowsError(try DepthEstimationOutput.process(
            output2D: [],
            modelType: .midas
        ))
    }
    
    // MARK: - Depth Query Tests
    
    func testDepthAtPixel() throws {
        let width = 10
        let height = 10
        var depthOutput = [Float](repeating: 0, count: width * height)
        
        // Set specific value
        depthOutput[5 * width + 3] = 42.0
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: width,
            height: height,
            modelType: .midas
        )
        
        XCTAssertEqual(result.depthAt(x: 3, y: 5), 42.0)
        XCTAssertNil(result.depthAt(x: -1, y: 0)) // Out of bounds
        XCTAssertNil(result.depthAt(x: 0, y: 100)) // Out of bounds
    }
    
    func testDepthAtNormalized() throws {
        let width = 10
        let height = 10
        var depthOutput = [Float](repeating: 0, count: width * height)
        
        // Set center value
        depthOutput[5 * width + 5] = 100.0
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: width,
            height: height,
            modelType: .midas
        )
        
        // Query near center (should interpolate)
        let centerDepth = result.depthAtNormalized(normalizedX: 0.5, normalizedY: 0.5)
        XCTAssertNotNil(centerDepth)
    }
    
    // MARK: - Normalization Tests
    
    func testNormalization() throws {
        var depthOutput = [Float](repeating: 0, count: 4)
        depthOutput[0] = 0
        depthOutput[1] = 50
        depthOutput[2] = 75
        depthOutput[3] = 100
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: 2,
            height: 2,
            modelType: .midas
        )
        
        let normalized = result.normalized()
        
        XCTAssertEqual(normalized[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(normalized[1], 0.5, accuracy: 0.01)
        XCTAssertEqual(normalized[2], 0.75, accuracy: 0.01)
        XCTAssertEqual(normalized[3], 1.0, accuracy: 0.01)
    }
    
    func testNormalizationInverted() throws {
        var depthOutput = [Float](repeating: 0, count: 4)
        depthOutput[0] = 0
        depthOutput[1] = 50
        depthOutput[2] = 75
        depthOutput[3] = 100
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: 2,
            height: 2,
            modelType: .midas
        )
        
        let normalized = result.normalized(invert: true)
        
        XCTAssertEqual(normalized[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(normalized[3], 0.0, accuracy: 0.01)
    }
    
    // MARK: - Statistics Tests
    
    func testStatistics() throws {
        let depthOutput: [Float] = [10, 20, 30, 40, 50]
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: 5,
            height: 1,
            modelType: .midas
        )
        
        let stats = result.statistics
        
        XCTAssertEqual(stats.min, 10)
        XCTAssertEqual(stats.max, 50)
        XCTAssertEqual(stats.mean, 30, accuracy: 0.01)
        XCTAssertEqual(stats.median, 30)
        XCTAssertEqual(stats.range, 40)
    }
    
    // MARK: - Resize Tests
    
    func testResize() throws {
        let width = 4
        let height = 4
        var depthOutput = [Float](repeating: 0, count: width * height)
        
        // Create simple pattern
        for y in 0..<height {
            for x in 0..<width {
                depthOutput[y * width + x] = Float(x * 10 + y)
            }
        }
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: width,
            height: height,
            modelType: .midas
        )
        
        // Resize to 8x8
        let resized = result.resized(to: 8, to: 8)
        
        XCTAssertEqual(resized.width, 8)
        XCTAssertEqual(resized.height, 8)
        XCTAssertEqual(resized.depthMap.count, 64)
        
        // Check corners are preserved approximately
        XCTAssertEqual(resized.depthAt(x: 0, y: 0) ?? 0, 0, accuracy: 0.1)
    }
    
    func testResizeToOriginal() throws {
        let modelWidth = 10
        let modelHeight = 10
        let originalWidth = 100
        let originalHeight = 100
        
        let depthOutput = [Float](repeating: 50, count: modelWidth * modelHeight)
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: modelWidth,
            height: modelHeight,
            modelType: .midas,
            originalWidth: originalWidth,
            originalHeight: originalHeight
        )
        
        let resized = result.resizedToOriginal()
        
        XCTAssertEqual(resized.width, originalWidth)
        XCTAssertEqual(resized.height, originalHeight)
    }
    
    // MARK: - Model Type Tests
    
    func testMiDaSModelType() {
        let modelType = DepthModelType.midas
        
        XCTAssertFalse(modelType.isMetric)
        XCTAssertTrue(modelType.isInverse)
        XCTAssertEqual(modelType.recommendedInputSize.width, 384)
        XCTAssertEqual(modelType.displayName, "MiDaS")
    }
    
    func testZoeDepthModelType() {
        let modelType = DepthModelType.zoeDepth
        
        XCTAssertTrue(modelType.isMetric)
        XCTAssertFalse(modelType.isInverse)
        XCTAssertEqual(modelType.recommendedInputSize.width, 512)
    }
    
    func testAllModelTypes() {
        for modelType in DepthModelType.allCases {
            XCTAssertFalse(modelType.displayName.isEmpty)
            XCTAssertGreaterThan(modelType.recommendedInputSize.width, 0)
            XCTAssertGreaterThan(modelType.recommendedInputSize.height, 0)
        }
    }
    
    // MARK: - Colormap Tests
    
    func testViridisColormap() {
        let colormap = DepthColormap.viridis
        
        // Test endpoints
        let color0 = colormap.color(forValue: 0.0)
        let color1 = colormap.color(forValue: 1.0)
        
        // Viridis starts dark blue, ends yellow
        XCTAssertLessThan(color0.r, 100) // Dark
        XCTAssertGreaterThan(color1.g, 200) // Yellow has high green
    }
    
    func testColormapClamping() {
        let colormap = DepthColormap.plasma
        
        // Values outside 0-1 should be clamped
        let colorNegative = colormap.color(forValue: -0.5)
        let colorExcess = colormap.color(forValue: 1.5)
        let color0 = colormap.color(forValue: 0.0)
        let color1 = colormap.color(forValue: 1.0)
        
        // Clamped values should match endpoints
        XCTAssertEqual(colorNegative.r, color0.r)
        XCTAssertEqual(colorExcess.r, color1.r)
    }
    
    func testAllColormaps() {
        let colormaps: [DepthColormap] = [
            .viridis, .plasma, .inferno, .magma, .turbo, .grayscale, .jet
        ]
        
        for colormap in colormaps {
            // Test mid-point color
            let midColor = colormap.color(forValue: 0.5)
            
            // All components should be valid
            XCTAssertLessThanOrEqual(midColor.r, 255)
            XCTAssertLessThanOrEqual(midColor.g, 255)
            XCTAssertLessThanOrEqual(midColor.b, 255)
        }
    }
    
    // MARK: - Visualization Tests
    
    func testToGrayscaleImage() throws {
        let depthOutput: [Float] = [0, 50, 50, 100]
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: 2,
            height: 2,
            modelType: .midas
        )
        
        let grayscale = result.toGrayscaleImage(invert: true)
        
        XCTAssertNotNil(grayscale)
        XCTAssertEqual(grayscale?.width, 2)
        XCTAssertEqual(grayscale?.height, 2)
    }
    
    func testToColoredImage() throws {
        let depthOutput: [Float] = [0, 33, 66, 100]
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: 2,
            height: 2,
            modelType: .midas
        )
        
        let colored = result.toColoredImage(colormap: .viridis)
        
        XCTAssertNotNil(colored)
        XCTAssertEqual(colored?.width, 2)
        XCTAssertEqual(colored?.height, 2)
    }
    
    func testToPlatformImage() throws {
        let depthOutput = [Float](repeating: 50, count: 100)
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: 10,
            height: 10,
            modelType: .midas
        )
        
        let platformImage = result.toPlatformImage(colormap: .plasma)
        
        XCTAssertNotNil(platformImage)
    }
    
    // MARK: - 2D Array Conversion
    
    func testAs2DArray() throws {
        let width = 3
        let height = 2
        var depthOutput = [Float](repeating: 0, count: width * height)
        
        // Row 0: [1, 2, 3]
        // Row 1: [4, 5, 6]
        depthOutput = [1, 2, 3, 4, 5, 6]
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: width,
            height: height,
            modelType: .midas
        )
        
        let array2D = result.as2DArray()
        
        XCTAssertEqual(array2D.count, 2) // 2 rows
        XCTAssertEqual(array2D[0].count, 3) // 3 columns
        XCTAssertEqual(array2D[0], [1, 2, 3])
        XCTAssertEqual(array2D[1], [4, 5, 6])
    }
    
    // MARK: - CVPixelBuffer Processing Tests
    
    func testProcessGrayscale8PixelBuffer() throws {
        // Create a grayscale 8-bit pixel buffer
        let width = 10
        let height = 10
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent8,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            XCTFail("Failed to create pixel buffer")
            return
        }
        
        // Fill with gradient values
        CVPixelBufferLockBaseAddress(buffer, [])
        if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<height {
                for x in 0..<width {
                    ptr[y * bytesPerRow + x] = UInt8((y * width + x) * 255 / (width * height - 1))
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        // Process with DepthEstimationOutput
        let result = try DepthEstimationOutput.process(
            pixelBuffer: buffer,
            modelType: .depthAnything,
            originalWidth: width,
            originalHeight: height
        )
        
        XCTAssertEqual(result.width, width)
        XCTAssertEqual(result.height, height)
        XCTAssertEqual(result.depthMap.count, width * height)
        
        // Values should be normalized 0-1
        XCTAssertGreaterThanOrEqual(result.minDepth, 0)
        XCTAssertLessThanOrEqual(result.maxDepth, 1)
    }
    
    func testProcessGrayscale32FloatPixelBuffer() throws {
        // Create a grayscale 32-bit float pixel buffer
        let width = 8
        let height = 8
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent32Float,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            XCTFail("Failed to create pixel buffer")
            return
        }
        
        // Fill with known values
        CVPixelBufferLockBaseAddress(buffer, [])
        if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
            let ptr = baseAddress.assumingMemoryBound(to: Float.self)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let stride = bytesPerRow / MemoryLayout<Float>.stride
            for y in 0..<height {
                for x in 0..<width {
                    ptr[y * stride + x] = Float(y * width + x) / Float(width * height - 1)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        // Process with DepthEstimationOutput
        let result = try DepthEstimationOutput.process(
            pixelBuffer: buffer,
            modelType: .midas,
            originalWidth: 100,
            originalHeight: 100
        )
        
        XCTAssertEqual(result.width, width)
        XCTAssertEqual(result.height, height)
        XCTAssertEqual(result.originalWidth, 100)
        XCTAssertEqual(result.originalHeight, 100)
        
        // Test first and last values
        let firstValue = result.depthAt(x: 0, y: 0)
        let lastValue = result.depthAt(x: width - 1, y: height - 1)
        XCTAssertNotNil(firstValue)
        XCTAssertNotNil(lastValue)
        XCTAssertEqual(Double(firstValue!), 0.0, accuracy: 0.01)
        XCTAssertEqual(Double(lastValue!), 1.0, accuracy: 0.01)
    }
    
    func testProcessPixelBufferPreservesModelType() throws {
        let width = 4
        let height = 4
        
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent8,
            nil,
            &pixelBuffer
        )
        
        guard let buffer = pixelBuffer else {
            XCTFail("Failed to create pixel buffer")
            return
        }
        
        // Test Depth Anything model type
        let resultDepthAnything = try DepthEstimationOutput.process(
            pixelBuffer: buffer,
            modelType: .depthAnything
        )
        XCTAssertTrue(resultDepthAnything.isInverse)
        XCTAssertFalse(resultDepthAnything.isMetric)
        
        // Test ZoeDepth model type
        let resultZoe = try DepthEstimationOutput.process(
            pixelBuffer: buffer,
            modelType: .zoeDepth
        )
        XCTAssertFalse(resultZoe.isInverse)
        XCTAssertTrue(resultZoe.isMetric)
    }
    
    func testProcessPixelBufferUnsupportedFormat() {
        let width = 4
        let height = 4
        
        var pixelBuffer: CVPixelBuffer?
        // Create BGRA format which is not supported for depth
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )
        
        guard let buffer = pixelBuffer else {
            XCTFail("Failed to create pixel buffer")
            return
        }
        
        XCTAssertThrowsError(try DepthEstimationOutput.process(
            pixelBuffer: buffer,
            modelType: .depthAnything
        ))
    }
    
    // MARK: - Custom Colormap Tests
    
    func testCustomColormapCreation() {
        // Create a simple red-to-blue colormap
        let colormap = DepthColormap.custom(
            name: "RedBlue",
            keyColors: [
                (r: 1.0, g: 0.0, b: 0.0),  // Red at 0
                (r: 0.0, g: 0.0, b: 1.0)   // Blue at 1
            ]
        )
        
        XCTAssertEqual(colormap.name, "RedBlue")
        
        // At value 0, should be red (allow small tolerance due to interpolation)
        let colorAt0 = colormap.color(forValue: 0.0)
        XCTAssertGreaterThan(colorAt0.r, 250)
        XCTAssertLessThan(colorAt0.g, 5)
        XCTAssertLessThan(colorAt0.b, 5)
        
        // At value 1, should be blue
        let colorAt1 = colormap.color(forValue: 1.0)
        XCTAssertLessThan(colorAt1.r, 5)
        XCTAssertLessThan(colorAt1.g, 5)
        XCTAssertGreaterThan(colorAt1.b, 250)
        
        // At value 0.5, should be purple-ish (interpolated)
        let colorAtMid = colormap.color(forValue: 0.5)
        XCTAssertTrue(colorAtMid.r > 100 && colorAtMid.r < 150)
        XCTAssertTrue(colorAtMid.b > 100 && colorAtMid.b < 150)
    }
    
    func testCustomColormapThreeColors() {
        // Create red -> green -> blue colormap
        let colormap = DepthColormap.custom(
            name: "RGB",
            keyColors: [
                (r: 1.0, g: 0.0, b: 0.0),  // Red at 0
                (r: 0.0, g: 1.0, b: 0.0),  // Green at 0.5
                (r: 0.0, g: 0.0, b: 1.0)   // Blue at 1
            ]
        )
        
        XCTAssertEqual(colormap.name, "RGB")
        
        // Check endpoints (allow small tolerance)
        let colorAt0 = colormap.color(forValue: 0.0)
        XCTAssertGreaterThan(colorAt0.r, 250)
        
        let colorAt1 = colormap.color(forValue: 1.0)
        XCTAssertGreaterThan(colorAt1.b, 250)
    }
    
    func testCustomColormapWithDepthResult() throws {
        // Create a simple depth map
        let depthOutput: [Float] = [0.0, 0.5, 0.5, 1.0]
        
        let result = try DepthEstimationOutput.process(
            output: depthOutput,
            width: 2,
            height: 2,
            modelType: .midas
        )
        
        // Create custom colormap
        let heatmap = DepthColormap.custom(
            name: "Heat",
            keyColors: [
                (r: 0.0, g: 0.0, b: 1.0),  // Blue (far)
                (r: 1.0, g: 1.0, b: 0.0),  // Yellow (mid)
                (r: 1.0, g: 0.0, b: 0.0)   // Red (near)
            ]
        )
        
        // Should be able to create colored image with custom colormap
        let image = result.toColoredImage(colormap: heatmap)
        XCTAssertNotNil(image)
    }
}

