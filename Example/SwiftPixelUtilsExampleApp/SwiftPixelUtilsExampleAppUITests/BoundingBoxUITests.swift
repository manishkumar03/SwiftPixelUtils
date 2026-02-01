//
//  BoundingBoxUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for Bounding Box utilities.
//  Uses accessibility identifiers for reliable, maintainable tests.
//
//  Created by Manish Kumar on 2026-01-31.
//

import XCTest

/// UI Tests for Bounding Box utilities
final class BoundingBoxUITests: XCTestCase {
    
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Format Conversion Tests
    
    /// Test cxcywh to xyxy conversion
    @MainActor
    func testBoundingBox_ConvertCxcywhToXyxy() throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to Boxes tab
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        let convertButton = app.buttons["bbox-convert-cxcywh-xyxy"]
        XCTAssertTrue(convertButton.waitForExistence(timeout: 3))
        convertButton.tap()
        
        let resultText = app.staticTexts["bbox-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))
    }
    
    /// Test xyxy to xywh conversion
    @MainActor
    func testBoundingBox_ConvertXyxyToXywh() throws {
        let app = XCUIApplication()
        app.launch()
        
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        let convertButton = app.buttons["bbox-convert-xyxy-xywh"]
        XCTAssertTrue(convertButton.waitForExistence(timeout: 3))
        convertButton.tap()
        
        let resultText = app.staticTexts["bbox-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))
    }
    
    /// Test xywh to cxcywh conversion
    @MainActor
    func testBoundingBox_ConvertXywhToCxcywh() throws {
        let app = XCUIApplication()
        app.launch()
        
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        let convertButton = app.buttons["bbox-convert-xywh-cxcywh"]
        XCTAssertTrue(convertButton.waitForExistence(timeout: 3))
        convertButton.tap()
        
        let resultText = app.staticTexts["bbox-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))
    }
    
    // MARK: - Box Operations Tests
    
    /// Test IoU calculation
    @MainActor
    func testBoundingBox_IoUCalculation() throws {
        let app = XCUIApplication()
        app.launch()
        
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        let iouButton = app.buttons["bbox-calculate-iou"]
        XCTAssertTrue(iouButton.waitForExistence(timeout: 3))
        iouButton.tap()
        
        let resultText = app.staticTexts["bbox-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))
    }
    
    /// Test Non-Max Suppression
    @MainActor
    func testBoundingBox_NMS() throws {
        let app = XCUIApplication()
        app.launch()
        
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        let nmsButton = app.buttons["bbox-nms"]
        XCTAssertTrue(nmsButton.waitForExistence(timeout: 3))
        nmsButton.tap()
        
        let resultText = app.staticTexts["bbox-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))
    }
    
    /// Test box scaling
    @MainActor
    func testBoundingBox_Scaling() throws {
        let app = XCUIApplication()
        app.launch()
        
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        let scaleButton = app.buttons["bbox-scale-boxes"]
        XCTAssertTrue(scaleButton.waitForExistence(timeout: 3))
        scaleButton.tap()
        
        let resultText = app.staticTexts["bbox-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))
    }
    
    /// Test box clipping
    @MainActor
    func testBoundingBox_Clipping() throws {
        let app = XCUIApplication()
        app.launch()
        
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        let clipButton = app.buttons["bbox-clip-boxes"]
        XCTAssertTrue(clipButton.waitForExistence(timeout: 3))
        clipButton.tap()
        
        let resultText = app.staticTexts["bbox-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))
    }
}
