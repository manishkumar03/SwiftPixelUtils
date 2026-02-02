//
//  DepthEstimationOutputTests.swift
//  SwiftPixelUtilsTests
//
//  Tests for depth estimation output processing
//

import XCTest
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
}
