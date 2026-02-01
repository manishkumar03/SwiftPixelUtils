//
//  SegmentationUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for DeepLabV3 Semantic Segmentation.
//  Uses accessibility identifiers for reliable, maintainable tests.
//
//  Created by Manish Kumar on 2026-01-31.
//

import XCTest

/// UI Tests for DeepLabV3 Semantic Segmentation
final class SegmentationUITests: XCTestCase {
    
    enum SampleImage: String, CaseIterable {
        case dog, car, aeroplane
        
        var accessibilityIdentifier: String { "segmentation-image-\(rawValue)" }
        var displayName: String { rawValue.capitalized }
    }
    
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Parameterized Tests
    
    @MainActor func testSegmentation_dog() throws { try runSegmentationTest(for: .dog) }
    @MainActor func testSegmentation_car() throws { try runSegmentationTest(for: .car) }
    @MainActor func testSegmentation_aeroplane() throws { try runSegmentationTest(for: .aeroplane) }
    
    // MARK: - Shared Test Implementation
    
    @MainActor
    private func runSegmentationTest(for image: SampleImage) throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to Inference tab
        let inferenceTab = app.tabBars.buttons["Inference"]
        XCTAssertTrue(inferenceTab.waitForExistence(timeout: 5))
        inferenceTab.tap()
        
        // Navigate to Segmentation
        let segmentationLink = app.buttons["inference-segmentation-link"]
        XCTAssertTrue(segmentationLink.waitForExistence(timeout: 3))
        segmentationLink.tap()
        
        // Select image
        let imageButton = app.buttons[image.accessibilityIdentifier]
        XCTAssertTrue(imageButton.waitForExistence(timeout: 3), "\(image.displayName) image button should exist")
        imageButton.tap()
        
        // Run segmentation
        let runButton = app.buttons["segmentation-run-button"]
        XCTAssertTrue(runButton.waitForExistence(timeout: 3))
        runButton.tap()
        
        // Wait for results
        let resultsHeader = app.staticTexts["segmentation-results-header"]
        XCTAssertTrue(resultsHeader.waitForExistence(timeout: 20), "Segmentation results should appear for \(image.displayName)")
        
        // Verify result image exists
        let resultImage = app.descendants(matching: .any)["segmentation-result-image"]
        XCTAssertTrue(resultImage.waitForExistence(timeout: 3))
        
        // Scroll to find inference time
        let inferenceTime = app.descendants(matching: .any)["segmentation-inference-time"]
        var scrollAttempts = 0
        while !inferenceTime.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(inferenceTime.waitForExistence(timeout: 5))
    }
    
    // MARK: - Additional Tests
    
    /// Test overlay toggle functionality
    @MainActor
    func testSegmentationOverlayToggle() throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to Segmentation and run
        let inferenceTab = app.tabBars.buttons["Inference"]
        XCTAssertTrue(inferenceTab.waitForExistence(timeout: 5))
        inferenceTab.tap()
        
        let segmentationLink = app.buttons["inference-segmentation-link"]
        XCTAssertTrue(segmentationLink.waitForExistence(timeout: 3))
        segmentationLink.tap()
        
        let runButton = app.buttons["segmentation-run-button"]
        XCTAssertTrue(runButton.waitForExistence(timeout: 3))
        runButton.tap()
        
        // Wait for results and toggle overlay
        let overlayToggle = app.descendants(matching: .any)["segmentation-overlay-toggle"]
        XCTAssertTrue(overlayToggle.waitForExistence(timeout: 20))
        overlayToggle.tap()
        
        // Verify toggle worked (opacity slider should disappear when toggle is off)
        // Toggle back on
        overlayToggle.tap()
        
        let opacitySlider = app.descendants(matching: .any)["segmentation-opacity-slider"]
        XCTAssertTrue(opacitySlider.waitForExistence(timeout: 3))
    }
}
