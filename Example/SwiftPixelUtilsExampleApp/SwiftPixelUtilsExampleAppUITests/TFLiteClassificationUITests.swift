//
//  TFLiteClassificationUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for TensorFlow Lite Classification.
//  Uses accessibility identifiers for reliable, maintainable tests.
//
//  Created by Manish Kumar on 2026-01-31.
//

import XCTest

/// UI Tests for TensorFlow Lite Classification
final class TFLiteClassificationUITests: XCTestCase {
    
    enum SampleImage: String, CaseIterable {
        case dog, car, lion
        
        var accessibilityIdentifier: String { "tflite-image-\(rawValue)" }
        var displayName: String { rawValue.capitalized }
    }
    
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Parameterized Tests
    
    @MainActor func testTFLiteClassification_dog() throws { try runClassificationTest(for: .dog) }
    @MainActor func testTFLiteClassification_car() throws { try runClassificationTest(for: .car) }
    @MainActor func testTFLiteClassification_lion() throws { try runClassificationTest(for: .lion) }
    
    // MARK: - Shared Test Implementation
    
    @MainActor
    private func runClassificationTest(for image: SampleImage) throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to Inference tab
        let inferenceTab = app.tabBars.buttons["Inference"]
        XCTAssertTrue(inferenceTab.waitForExistence(timeout: 5))
        inferenceTab.tap()
        
        // Navigate to TFLite Classification
        let tfliteLink = app.buttons["inference-tflite-classification-link"]
        XCTAssertTrue(tfliteLink.waitForExistence(timeout: 3))
        tfliteLink.tap()
        
        // Select image
        let imageButton = app.buttons[image.accessibilityIdentifier]
        XCTAssertTrue(imageButton.waitForExistence(timeout: 3), "\(image.displayName) image button should exist")
        imageButton.tap()
        
        // Run classification
        let runButton = app.buttons["tflite-run-classification-button"]
        XCTAssertTrue(runButton.waitForExistence(timeout: 3))
        runButton.tap()
        
        // Wait for results
        let resultsHeader = app.staticTexts["tflite-classification-results-header"]
        XCTAssertTrue(resultsHeader.waitForExistence(timeout: 15), "Classification results should appear for \(image.displayName)")
        
        // Verify at least one prediction row
        let firstPrediction = app.descendants(matching: .any)["tflite-prediction-row-0"]
        XCTAssertTrue(firstPrediction.waitForExistence(timeout: 3))
        
        // Scroll to find inference time
        let inferenceTime = app.descendants(matching: .any)["tflite-inference-time"]
        var scrollAttempts = 0
        while !inferenceTime.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(inferenceTime.waitForExistence(timeout: 5))
    }
}
