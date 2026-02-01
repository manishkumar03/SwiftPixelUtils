//
//  YOLODetectionUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  Created by Manish Kumar on 2026-01-31.
//

import XCTest

/// UI Tests for YOLO Detection with parameterized image testing.
///
/// These tests verify YOLO object detection works correctly for all sample images.
/// Uses a helper method to avoid code duplication while testing each image.
final class YOLODetectionUITests: XCTestCase {
    /// Sample images available in the YOLO Detection view
    enum SampleImage: String, CaseIterable {
        case dogBoat
        case street
        case twoBuses
        
        /// The accessibility identifier for this image's button
        var accessibilityIdentifier: String {
            "yolo-image-\(rawValue)"
        }
        
        /// Human-readable description for test output
        var displayName: String {
            switch self {
            case .dogBoat: return "Dog & Boat"
            case .street: return "Street Scene"
            case .twoBuses: return "Two Buses"
            }
        }
    }
    
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Parameterized Tests (one test method per image)
    
    @MainActor
    func testYOLODetection_dogBoat() throws {
        try runYOLODetectionTest(for: .dogBoat)
    }
    
    @MainActor
    func testYOLODetection_street() throws {
        try runYOLODetectionTest(for: .street)
    }
    
    @MainActor
    func testYOLODetection_twoBuses() throws {
        try runYOLODetectionTest(for: .twoBuses)
    }
    
    // MARK: - Shared Test Implementation
    
    /// Runs the YOLO detection test for a specific sample image.
    ///
    /// This helper method contains the shared test logic, verifying:
    /// 1. Navigation to YOLO Detection view
    /// 2. Image selection
    /// 3. Detection execution
    /// 4. Results display (detected objects header, detection rows, inference time)
    ///
    /// ## Accessibility Identifiers Used
    /// - `inference-yolo-detection-link`: NavigationLink to YOLO view
    /// - `yolo-image-{imageName}`: Image selection buttons
    /// - `yolo-run-detection-button`: Run Detection button
    /// - `yolo-detected-objects-header`: Detection results header
    /// - `yolo-detection-row-{index}`: Individual detection rows
    /// - `yolo-inference-time`: Inference time display
    @MainActor
    private func runYOLODetectionTest(for image: SampleImage) throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to Inference tab
        let inferenceTab = app.tabBars.buttons["Inference"]
        XCTAssertTrue(inferenceTab.waitForExistence(timeout: 5), "Inference tab should exist")
        inferenceTab.tap()
        
        // Navigate to YOLO Detection using accessibility identifier
        let yoloLink = app.buttons["inference-yolo-detection-link"]
        XCTAssertTrue(yoloLink.waitForExistence(timeout: 3), "YOLO Detection link should exist")
        yoloLink.tap()
        
        // Select the image using accessibility identifier
        let imageButton = app.buttons[image.accessibilityIdentifier]
        XCTAssertTrue(imageButton.waitForExistence(timeout: 3), "\(image.displayName) image button should exist")
        imageButton.tap()
        
        // Run detection using accessibility identifier
        let runButton = app.buttons["yolo-run-detection-button"]
        XCTAssertTrue(runButton.waitForExistence(timeout: 3), "Run Detection button should exist")
        runButton.tap()
        
        // Wait for detection results header to appear
        let detectedObjectsHeader = app.staticTexts["yolo-detected-objects-header"]
        XCTAssertTrue(
            detectedObjectsHeader.waitForExistence(timeout: 15),
            "Detection results should appear for \(image.displayName)"
        )
        
        // Verify at least one detection row exists
        let firstDetectionRow = app.descendants(matching: .any)["yolo-detection-row-0"]
        XCTAssertTrue(
            firstDetectionRow.waitForExistence(timeout: 3),
            "At least one detection should be found in \(image.displayName)"
        )
        
        // Find inference time using a11y identifier
        // Note: SwiftUI Lists use lazy loading, so off-screen elements may not exist until scrolled into view
        let inferenceTimeLabel = app.descendants(matching: .any)["yolo-inference-time"]
        
        // Scroll until the element exists or we've tried enough times
        var scrollAttempts = 0
        while !inferenceTimeLabel.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        
        XCTAssertTrue(
            inferenceTimeLabel.waitForExistence(timeout: 5),
            "Inference time should be displayed for \(image.displayName)"
        )
    }
}
