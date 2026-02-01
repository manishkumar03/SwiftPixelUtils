//
//  TabNavigationUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for main tab bar navigation.
//  Uses accessibility identifiers for reliable, maintainable tests.
//
//  Created by Manish Kumar on 2026-01-31.
//

import XCTest

/// UI Tests for main tab bar navigation
final class TabNavigationUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    /// Test all tabs are accessible
    @MainActor
    func testAllTabsAccessible() throws {
        let app = XCUIApplication()
        app.launch()
        
        let tabs = ["Inference", "Pixels", "Boxes", "More"]
        
        for tabName in tabs {
            let tab = app.tabBars.buttons[tabName]
            XCTAssertTrue(tab.waitForExistence(timeout: 3), "\(tabName) tab should exist")
            tab.tap()
            
            // Small delay to allow view to load
            Thread.sleep(forTimeInterval: 0.5)
        }
    }
    
    /// Test Inference tab loads correctly
    @MainActor
    func testInferenceTabLoads() throws {
        let app = XCUIApplication()
        app.launch()
        
        let inferenceTab = app.tabBars.buttons["Inference"]
        XCTAssertTrue(inferenceTab.waitForExistence(timeout: 5))
        inferenceTab.tap()
        
        // Verify inference options are visible
        let tfliteLink = app.buttons["inference-tflite-classification-link"]
        let execuTorchLink = app.buttons["inference-executorch-classification-link"]
        let yoloLink = app.buttons["inference-yolo-detection-link"]
        let segmentationLink = app.buttons["inference-segmentation-link"]
        
        XCTAssertTrue(tfliteLink.waitForExistence(timeout: 3), "TFLite link should be visible")
        XCTAssertTrue(execuTorchLink.waitForExistence(timeout: 3), "ExecuTorch link should be visible")
        XCTAssertTrue(yoloLink.waitForExistence(timeout: 3), "YOLO link should be visible")
        XCTAssertTrue(segmentationLink.waitForExistence(timeout: 3), "Segmentation link should be visible")
    }
    
    /// Test Pixels tab loads correctly
    @MainActor
    func testPixelsTabLoads() throws {
        let app = XCUIApplication()
        app.launch()
        
        let pixelsTab = app.tabBars.buttons["Pixels"]
        XCTAssertTrue(pixelsTab.waitForExistence(timeout: 5))
        pixelsTab.tap()
        
        // Verify preset buttons are visible
        let yoloPreset = app.buttons["pixel-preset-yolov8"]
        XCTAssertTrue(yoloPreset.waitForExistence(timeout: 3), "YOLOv8 preset button should be visible")
    }
    
    /// Test Boxes tab loads correctly
    @MainActor
    func testBoxesTabLoads() throws {
        let app = XCUIApplication()
        app.launch()
        
        let boxesTab = app.tabBars.buttons["Boxes"]
        XCTAssertTrue(boxesTab.waitForExistence(timeout: 5))
        boxesTab.tap()
        
        // Verify bounding box buttons are visible
        let iouButton = app.buttons["bbox-calculate-iou"]
        XCTAssertTrue(iouButton.waitForExistence(timeout: 3), "IoU button should be visible")
    }
    
    /// Test More tab loads correctly
    @MainActor
    func testMoreTabLoads() throws {
        let app = XCUIApplication()
        app.launch()
        
        let moreTab = app.tabBars.buttons["More"]
        XCTAssertTrue(moreTab.waitForExistence(timeout: 5))
        moreTab.tap()
        
        // Verify navigation links are visible
        let augmentationLink = app.buttons["more-augmentation-link"]
        XCTAssertTrue(augmentationLink.waitForExistence(timeout: 3), "Augmentation link should be visible")
    }
}
