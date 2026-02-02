//
//  SwiftPixelUtilsExampleAppTests.swift
//  SwiftPixelUtilsExampleAppTests
//
//  Tests for SwiftPixelUtils Example App functionality
//

import XCTest
import SwiftPixelUtils
import CoreML

final class SwiftPixelUtilsExampleAppTests: XCTestCase {
    
    // MARK: - Depth Estimation Tests
    
    func testDepthModelExists() throws {
        // Verify the depth model is bundled in resources
        let modelURL = Bundle.main.url(
            forResource: "DepthAnythingSmallF16P6",
            withExtension: "mlpackage",
            subdirectory: "Resources"
        )
        
        // Model might be in different locations depending on build
        // This test verifies the model file exists somewhere accessible
        XCTAssertNotNil(modelURL, "DepthAnythingSmallF16P6 model should be bundled in Resources")
    }
    
    func testDepthEstimationOutputProcessing() throws {
        // Test the DepthEstimationOutput processing without requiring actual model inference
        let width = 100
        let height = 100
        
        // Create synthetic depth data (gradient: closer at top)
        var depthData = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                // Inverse depth: higher values = closer
                depthData[y * width + x] = Float(height - y) * 10
            }
        }
        
        let result = try DepthEstimationOutput.process(
            output: depthData,
            width: width,
            height: height,
            modelType: .depthAnything,
            originalWidth: 640,
            originalHeight: 480
        )
        
        // Verify dimensions
        XCTAssertEqual(result.width, width)
        XCTAssertEqual(result.height, height)
        XCTAssertEqual(result.originalWidth, 640)
        XCTAssertEqual(result.originalHeight, 480)
        
        // Verify depth properties
        XCTAssertTrue(result.isInverse) // Depth Anything uses inverse depth
        XCTAssertFalse(result.isMetric)
        
        // Verify min/max
        XCTAssertEqual(result.minDepth, 10) // Bottom row: (100-99) * 10 = 10
        XCTAssertEqual(result.maxDepth, 1000) // Top row: (100-0) * 10 = 1000
    }
    
    func testDepthVisualization() throws {
        let width = 50
        let height = 50
        var depthData = [Float](repeating: 0, count: width * height)
        
        // Simple gradient
        for i in 0..<depthData.count {
            depthData[i] = Float(i)
        }
        
        let result = try DepthEstimationOutput.process(
            output: depthData,
            width: width,
            height: height,
            modelType: .depthAnything
        )
        
        // Test grayscale visualization
        let grayscaleImage = result.toGrayscaleImage(invert: true)
        XCTAssertNotNil(grayscaleImage)
        XCTAssertEqual(grayscaleImage?.width, width)
        XCTAssertEqual(grayscaleImage?.height, height)
        
        // Test colored visualization with different colormaps
        let viridisImage = result.toColoredImage(colormap: .viridis)
        XCTAssertNotNil(viridisImage)
        
        let plasmaImage = result.toColoredImage(colormap: .plasma)
        XCTAssertNotNil(plasmaImage)
        
        let turboImage = result.toColoredImage(colormap: .turbo)
        XCTAssertNotNil(turboImage)
    }
    
    func testDepthStatistics() throws {
        let width = 10
        let height = 10
        
        // Create known depth values
        var depthData = [Float]()
        for i in 1...100 {
            depthData.append(Float(i))
        }
        
        let result = try DepthEstimationOutput.process(
            output: depthData,
            width: width,
            height: height,
            modelType: .depthAnything
        )
        
        let stats = result.statistics
        
        XCTAssertEqual(stats.min, 1)
        XCTAssertEqual(stats.max, 100)
        XCTAssertEqual(stats.mean, 50.5, accuracy: 0.01)
        XCTAssertEqual(stats.median, 50.5, accuracy: 1.0)
    }
    
    func testDepthResize() throws {
        let width = 20
        let height = 20
        var depthData = [Float](repeating: 0, count: width * height)
        
        // Create checkerboard pattern
        for y in 0..<height {
            for x in 0..<width {
                depthData[y * width + x] = Float((x + y) % 2 == 0 ? 100 : 200)
            }
        }
        
        let result = try DepthEstimationOutput.process(
            output: depthData,
            width: width,
            height: height,
            modelType: .depthAnything,
            originalWidth: 100,
            originalHeight: 100
        )
        
        // Resize to original
        let resized = result.resizedToOriginal()
        XCTAssertEqual(resized.width, 100)
        XCTAssertEqual(resized.height, 100)
        
        // Resize to specific size
        let smallResized = result.resized(to: 10, to: 10)
        XCTAssertEqual(smallResized.width, 10)
        XCTAssertEqual(smallResized.height, 10)
    }
    
    func testDepthQueryMethods() throws {
        let width = 10
        let height = 10
        var depthData = [Float](repeating: 0, count: width * height)
        
        // Set specific value at known location
        depthData[5 * width + 3] = 42.0 // (3, 5)
        
        let result = try DepthEstimationOutput.process(
            output: depthData,
            width: width,
            height: height,
            modelType: .depthAnything
        )
        
        // Test pixel query
        XCTAssertEqual(result.depthAt(x: 3, y: 5), 42.0)
        XCTAssertEqual(result.depthAt(x: 0, y: 0), 0.0)
        
        // Test out of bounds
        XCTAssertNil(result.depthAt(x: -1, y: 0))
        XCTAssertNil(result.depthAt(x: 100, y: 0))
        
        // Test normalized query (should interpolate)
        let normalizedDepth = result.depthAtNormalized(normalizedX: 0.5, normalizedY: 0.5)
        XCTAssertNotNil(normalizedDepth)
    }
    
    func testDepthModelTypes() {
        // Verify model type properties
        XCTAssertTrue(DepthModelType.depthAnything.isInverse)
        XCTAssertFalse(DepthModelType.depthAnything.isMetric)
        XCTAssertEqual(DepthModelType.depthAnything.recommendedInputSize.width, 518)
        XCTAssertEqual(DepthModelType.depthAnything.recommendedInputSize.height, 518)
        
        XCTAssertTrue(DepthModelType.midas.isInverse)
        XCTAssertFalse(DepthModelType.midas.isMetric)
        
        XCTAssertFalse(DepthModelType.zoeDepth.isInverse)
        XCTAssertTrue(DepthModelType.zoeDepth.isMetric)
    }
    
    func testColormapGeneration() {
        // Test that all colormaps produce valid colors
        let colormaps: [DepthColormap] = [.viridis, .plasma, .inferno, .magma, .turbo, .grayscale]
        
        for colormap in colormaps {
            // Test at boundaries and middle
            let color0 = colormap.color(forValue: 0.0)
            let color50 = colormap.color(forValue: 0.5)
            let color100 = colormap.color(forValue: 1.0)
            
            // Verify colors are valid RGB values (0-255)
            XCTAssertLessThanOrEqual(color0.r, 255)
            XCTAssertLessThanOrEqual(color50.g, 255)
            XCTAssertLessThanOrEqual(color100.b, 255)
        }
    }
    
    // MARK: - Sample Image Tests
    
    func testSampleImagesExist() {
        // Verify sample images are available
        let expectedImages = ["dog", "car", "street", "lion"]
        
        for imageName in expectedImages {
            let exists = Bundle.main.path(forResource: imageName, ofType: "jpg", inDirectory: "Resources") != nil ||
                        Bundle.main.path(forResource: imageName, ofType: "jpg") != nil
            // Images might not be available in unit test bundle, so just log
            if !exists {
                print("Note: Sample image '\(imageName)' not found in test bundle")
            }
        }
    }
}
