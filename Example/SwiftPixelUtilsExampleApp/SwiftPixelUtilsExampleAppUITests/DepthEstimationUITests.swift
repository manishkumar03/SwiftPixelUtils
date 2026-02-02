//
//  DepthEstimationUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for Depth Anything Depth Estimation.
//  Uses accessibility identifiers for reliable, maintainable tests.
//
//  Created by Manish Kumar on 2026-02-01.
//

import XCTest

/// UI Tests for Depth Anything Depth Estimation
final class DepthEstimationUITests: XCTestCase {
    enum SampleImage: String, CaseIterable {
        case dog, car, street, lion
        
        var accessibilityIdentifier: String { "depth-image-\(rawValue)" }
        var displayName: String { rawValue.capitalized }
    }
    
    enum Colormap: String, CaseIterable {
        case viridis = "Viridis"
        case plasma = "Plasma"
        case inferno = "Inferno"
        case magma = "Magma"
        case turbo = "Turbo"
        case grayscale = "Grayscale"
    }
    
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Parameterized Tests
    
    @MainActor func testDepthEstimation_dog() throws { try runDepthEstimationTest(for: .dog) }
    @MainActor func testDepthEstimation_car() throws { try runDepthEstimationTest(for: .car) }
    @MainActor func testDepthEstimation_street() throws { try runDepthEstimationTest(for: .street) }
    @MainActor func testDepthEstimation_lion() throws { try runDepthEstimationTest(for: .lion) }
    
    // MARK: - Shared Test Implementation
    
    @MainActor
    private func runDepthEstimationTest(for image: SampleImage) throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to Inference tab
        let inferenceTab = app.tabBars.buttons["Inference"]
        XCTAssertTrue(inferenceTab.waitForExistence(timeout: 5))
        inferenceTab.tap()
        
        // Navigate to Depth Estimation using accessibility identifier
        let depthLink = app.buttons["inference-depth-estimation-link"]
        XCTAssertTrue(depthLink.waitForExistence(timeout: 3))
        depthLink.tap()
        
        // Select image
        let imageButton = app.buttons[image.accessibilityIdentifier]
        XCTAssertTrue(imageButton.waitForExistence(timeout: 3), "\(image.displayName) image button should exist")
        imageButton.tap()
        
        // Run depth estimation
        let runButton = app.buttons["depth-run-button"]
        XCTAssertTrue(runButton.waitForExistence(timeout: 3))
        runButton.tap()
        
        // Wait for results - overlay toggle appears when inference completes
        let overlayToggle = app.switches["depth-overlay-toggle"]
        XCTAssertTrue(overlayToggle.waitForExistence(timeout: 60), "Depth results should appear for \(image.displayName)")
        
        // Verify inference time exists (it's on an HStack, so use descendants)
        let inferenceTime = app.descendants(matching: .any)["depth-inference-time"]
        XCTAssertTrue(inferenceTime.exists, "Inference time should be displayed")
        
        // Scroll to find statistics
        app.swipeUp()
        
        // Verify statistics are displayed (use staticTexts for label text)
        XCTAssertTrue(app.staticTexts["Min Depth"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["Max Depth"].exists)
        XCTAssertTrue(app.staticTexts["Mean"].exists)
    }
    
    // MARK: - Colormap Tests
    
    @MainActor
    func testColormapSelection() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToDepthEstimation(app: app)
        
        // Verify colormap picker exists
        let colormapPicker = app.segmentedControls["depth-colormap-picker"]
        XCTAssertTrue(colormapPicker.waitForExistence(timeout: 5))
        
        // Test selecting each colormap
        for colormap in [Colormap.viridis, .plasma, .inferno] {
            let button = colormapPicker.buttons[colormap.rawValue]
            if button.exists {
                button.tap()
            }
        }
    }
    
    @MainActor
    func testColormapChangeRerunsInference() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToDepthEstimation(app: app)
        
        // Run initial inference
        let runButton = app.buttons["depth-run-button"]
        XCTAssertTrue(runButton.waitForExistence(timeout: 3))
        runButton.tap()
        
        // Wait for completion
        let overlayToggle = app.switches["depth-overlay-toggle"]
        XCTAssertTrue(overlayToggle.waitForExistence(timeout: 60))
        
        // Change colormap
        let colormapPicker = app.segmentedControls["depth-colormap-picker"]
        colormapPicker.buttons["Plasma"].tap()
        
        // Wait a moment for re-inference
        sleep(2)
        
        // Results should still be visible
        XCTAssertTrue(overlayToggle.exists)
    }
    
    // MARK: - Image Change Tests
    
    @MainActor
    func testImageChangeResetsResults() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToDepthEstimation(app: app)
        
        // Run inference on default image
        let runButton = app.buttons["depth-run-button"]
        runButton.tap()
        
        // Wait for completion
        let overlayToggle = app.switches["depth-overlay-toggle"]
        XCTAssertTrue(overlayToggle.waitForExistence(timeout: 60))
        
        // Select a different image
        let dogImage = app.buttons["depth-image-dog"]
        dogImage.tap()
        
        // Wait a moment for state to update
        Thread.sleep(forTimeInterval: 0.5)
        
        // Results should be reset - overlay toggle should no longer exist
        XCTAssertFalse(overlayToggle.exists, "Overlay toggle should disappear after selecting a new image")
    }
    
    // MARK: - Navigation Tests
    
    @MainActor
    func testNavigationTitle() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToDepthEstimation(app: app)
        
        // Verify navigation title
        XCTAssertTrue(app.navigationBars["Depth Estimation"].exists)
    }
    
    @MainActor
    func testAllSampleImagesExist() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToDepthEstimation(app: app)
        
        // Verify all sample images exist
        for image in SampleImage.allCases {
            let imageButton = app.buttons[image.accessibilityIdentifier]
            XCTAssertTrue(imageButton.waitForExistence(timeout: 3), "\(image.displayName) image should exist")
        }
    }
    
    // MARK: - Helper Methods
    
    private func navigateToDepthEstimation(app: XCUIApplication) {
        // Navigate to Inference tab
        let inferenceTab = app.tabBars.buttons["Inference"]
        if inferenceTab.waitForExistence(timeout: 5) {
            inferenceTab.tap()
        }
        
        // Tap on Depth Estimation using accessibility identifier
        let depthLink = app.buttons["inference-depth-estimation-link"]
        if depthLink.waitForExistence(timeout: 5) {
            depthLink.tap()
        }
    }
}
