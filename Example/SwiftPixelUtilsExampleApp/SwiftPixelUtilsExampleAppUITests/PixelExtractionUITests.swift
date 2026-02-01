//
//  PixelExtractionUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for Pixel Extraction functionality.
//  Uses accessibility identifiers for reliable, maintainable tests.
//
//  Created by Manish Kumar on 2026-01-31.
//

import XCTest

/// UI Tests for Pixel Extraction functionality
final class PixelExtractionUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Model Preset Tests
    
    /// Test YOLOv8 preset extraction
    @MainActor
    func testPixelExtraction_YOLOv8Preset() throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to Pixels tab
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        // Tap YOLOv8 preset button
        let yoloButton = app.buttons["pixel-preset-yolov8"]
        XCTAssertTrue(yoloButton.waitForExistence(timeout: 3))
        yoloButton.tap()
        
        // Verify result text shows success
        let resultText = app.staticTexts["pixel-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 10))
        
        // Wait for processing to complete and check result contains success indicator
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
    
    /// Test MobileNet preset extraction
    @MainActor
    func testPixelExtraction_MobileNetPreset() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        let mobileNetButton = app.buttons["pixel-preset-mobilenet"]
        XCTAssertTrue(mobileNetButton.waitForExistence(timeout: 3))
        mobileNetButton.tap()
        
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
    
    /// Test ResNet50 preset extraction
    @MainActor
    func testPixelExtraction_ResNet50Preset() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        let resnetButton = app.buttons["pixel-preset-resnet50"]
        XCTAssertTrue(resnetButton.waitForExistence(timeout: 3))
        resnetButton.tap()
        
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
    
    /// Test ViT preset extraction
    @MainActor
    func testPixelExtraction_ViTPreset() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        let vitButton = app.buttons["pixel-preset-vit"]
        XCTAssertTrue(vitButton.waitForExistence(timeout: 3))
        vitButton.tap()
        
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
    
    /// Test CLIP preset extraction
    @MainActor
    func testPixelExtraction_CLIPPreset() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        let clipButton = app.buttons["pixel-preset-clip"]
        XCTAssertTrue(clipButton.waitForExistence(timeout: 3))
        clipButton.tap()
        
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
    
    // MARK: - Custom Options Tests
    
    /// Test custom RGB + ImageNet normalization
    @MainActor
    func testPixelExtraction_CustomRGBImageNet() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        let customButton = app.buttons["pixel-custom-rgb-imagenet"]
        XCTAssertTrue(customButton.waitForExistence(timeout: 3))
        customButton.tap()
        
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
    
    /// Test custom Grayscale + Scale normalization
    @MainActor
    func testPixelExtraction_CustomGrayscale() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        let customButton = app.buttons["pixel-custom-grayscale"]
        XCTAssertTrue(customButton.waitForExistence(timeout: 3))
        customButton.tap()
        
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
    
    /// Test custom BGR + TensorFlow normalization
    @MainActor
    func testPixelExtraction_CustomBGRTensorFlow() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        let customButton = app.buttons["pixel-custom-bgr-tensorflow"]
        XCTAssertTrue(customButton.waitForExistence(timeout: 3))
        customButton.tap()
        
        let processingTime = app.staticTexts["pixel-processing-time"]
        XCTAssertTrue(processingTime.waitForExistence(timeout: 15))
    }
}
