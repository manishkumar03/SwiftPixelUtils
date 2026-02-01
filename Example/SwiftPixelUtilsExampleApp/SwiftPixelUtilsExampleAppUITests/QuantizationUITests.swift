//
//  QuantizationUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for Quantization functionality.
//  Tests both per-tensor and per-channel quantization modes.
//
//  Created by Manish Kumar on 2026-02-01.
//

import XCTest

/// UI Tests for Quantization functionality
final class QuantizationUITests: XCTestCase {
    
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Navigation Helper
    
    @MainActor
    private func navigateToQuantization(app: XCUIApplication) {
        // Navigate to More tab
        let moreTab = app.tabBars.buttons["More"]
        XCTAssertTrue(moreTab.waitForExistence(timeout: 5))
        moreTab.tap()
        
        // Navigate to Quantization
        let quantLink = app.buttons["more-quantization-link"]
        XCTAssertTrue(quantLink.waitForExistence(timeout: 3))
        quantLink.tap()
    }
    
    // MARK: - Per-Tensor Tests
    
    /// Test per-tensor UInt8 quantization
    @MainActor
    func testPerTensorUInt8Quantization() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-per-tensor-uint8"]
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        // Verify result contains success indicator
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Per-Tensor"), "Result should indicate per-tensor mode")
    }
    
    /// Test per-tensor Int8 quantization
    @MainActor
    func testPerTensorInt8Quantization() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-per-tensor-int8"]
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
    }
    
    /// Test per-tensor Int16 quantization
    @MainActor
    func testPerTensorInt16Quantization() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-per-tensor-int16"]
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
    }
    
    // MARK: - Per-Channel Tests
    
    /// Test per-channel Int8 quantization with CHW layout
    @MainActor
    func testPerChannelInt8CHWQuantization() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-per-channel-int8-chw"]
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Per-Channel"), "Result should indicate per-channel mode")
        XCTAssertTrue(result.contains("CHW"), "Result should indicate CHW layout")
        XCTAssertTrue(result.contains("R:") && result.contains("G:") && result.contains("B:"), 
                     "Result should show per-channel parameters")
    }
    
    /// Test per-channel UInt8 quantization with HWC layout
    @MainActor
    func testPerChannelUInt8HWCQuantization() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-per-channel-uint8-hwc"]
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("HWC"), "Result should indicate HWC layout")
    }
    
    /// Test comparison between per-tensor and per-channel modes
    @MainActor
    func testCompareModes() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        // May need to scroll to find the button
        let button = app.buttons["quant-compare-modes"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 3 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Per-TENSOR") && result.contains("Per-CHANNEL"), 
                     "Result should compare both modes")
        XCTAssertTrue(result.contains("more accurate"), 
                     "Result should show per-channel is more accurate")
    }
    
    // MARK: - Round Trip Tests
    
    /// Test per-tensor round trip
    @MainActor
    func testPerTensorRoundTrip() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-round-trip-per-tensor"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 3 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Round Trip"), "Result should indicate round trip")
        XCTAssertTrue(result.contains("Max Error"), "Result should show error metrics")
    }
    
    /// Test per-channel round trip
    @MainActor
    func testPerChannelRoundTrip() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-round-trip-per-channel"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 3 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Per-Channel Round Trip"), "Result should indicate per-channel round trip")
        XCTAssertTrue(result.contains("R Channel") && result.contains("G Channel") && result.contains("B Channel"),
                     "Result should show per-channel results")
    }
    
    // MARK: - Calibration Tests
    
    /// Test per-tensor calibration
    @MainActor
    func testPerTensorCalibration() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-calibrate-per-tensor"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 4 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Per-Tensor Calibration"), "Result should indicate calibration")
        XCTAssertTrue(result.contains("Scale") && result.contains("Zero Point"), 
                     "Result should show calibrated parameters")
    }
    
    /// Test per-channel calibration
    @MainActor
    func testPerChannelCalibration() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-calibrate-per-channel"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 4 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Per-Channel Calibration"), "Result should indicate per-channel calibration")
        XCTAssertTrue(result.contains("R:") && result.contains("G:") && result.contains("B:"),
                     "Result should show per-channel parameters")
        XCTAssertTrue(result.contains("Channel Value Ranges"), "Result should show detected ranges")
    }
    
    // MARK: - INT4 Tests (LLM/Edge)
    
    /// Test INT4 signed quantization
    @MainActor
    func testInt4SignedQuantization() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        // Scroll to find INT4 section
        let button = app.buttons["quant-int4-signed"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("INT4") || result.contains("int4"), "Result should indicate INT4")
        XCTAssertTrue(result.contains("Packed"), "Result should show packed bytes")
        XCTAssertTrue(result.contains("8") && result.contains("smaller"), "Result should show compression")
    }
    
    /// Test UINT4 unsigned quantization
    @MainActor
    func testUInt4UnsignedQuantization() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-uint4-unsigned"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Unsigned") || result.contains("uint4"), "Result should indicate unsigned")
    }
    
    /// Test INT4 round trip
    @MainActor
    func testInt4RoundTrip() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-int4-round-trip"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("Round Trip"), "Result should indicate round trip")
        XCTAssertTrue(result.contains("Original") && result.contains("Restored"), 
                     "Result should show original and restored values")
        XCTAssertTrue(result.contains("Max Error") || result.contains("Avg Error"), 
                     "Result should show error metrics")
    }
    
    /// Test INT4 vs INT8 comparison
    @MainActor
    func testCompareInt4VsInt8() throws {
        let app = XCUIApplication()
        app.launch()
        
        navigateToQuantization(app: app)
        
        let button = app.buttons["quant-compare-int4-int8"]
        var scrollAttempts = 0
        while !button.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        XCTAssertTrue(button.waitForExistence(timeout: 3))
        button.tap()
        
        let resultText = app.staticTexts["quant-result-text"]
        XCTAssertTrue(resultText.waitForExistence(timeout: 5))
        
        let result = resultText.label
        XCTAssertTrue(result.contains("✅"), "Result should show success")
        XCTAssertTrue(result.contains("INT4") && result.contains("INT8"), 
                     "Result should compare both types")
        XCTAssertTrue(result.contains("smaller"), "Result should show size comparison")
        XCTAssertTrue(result.contains("accurate"), "Result should show accuracy comparison")
    }
}
