//
//  CVPixelBufferUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for CVPixelBuffer conversion demo.
//
//  Created by Manish Kumar on 2026-02-02.
//

import XCTest

final class CVPixelBufferUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testRGB565ConversionButtons() throws {
        let app = XCUIApplication()
        app.launch()

        navigateToCVPixelBuffer(app: app)

        let leButton = app.buttons["cvpixelbuffer-test-rgb565-le"]
        let beButton = app.buttons["cvpixelbuffer-test-rgb565-be"]
        let resultText = app.staticTexts["cvpixelbuffer-result-text"]

        XCTAssertTrue(leButton.waitForExistence(timeout: 3))
        XCTAssertTrue(beButton.exists)
        XCTAssertTrue(resultText.exists)

        leButton.tap()
        XCTAssertTrue(resultText.waitForExistence(timeout: 3))

        beButton.tap()
        XCTAssertTrue(resultText.exists)
    }

    @MainActor
    private func navigateToCVPixelBuffer(app: XCUIApplication) {
        let moreTab = app.tabBars.buttons["More"]
        if moreTab.waitForExistence(timeout: 5) {
            moreTab.tap()
        }

        let link = app.buttons["more-cvpixelbuffer-link"]
        if !link.exists {
            app.swipeUp()
        }
        if link.waitForExistence(timeout: 5) {
            link.tap()
        }
    }
}
