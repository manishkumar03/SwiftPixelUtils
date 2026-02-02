//
//  ImageAnalysisUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for Image Analysis view.
//
//  Created by Manish Kumar on 2026-02-02.
//

import XCTest

final class ImageAnalysisUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testImageAnalysisViewElementsExist() throws {
        let app = XCUIApplication()
        app.launch()

        navigateToImageAnalysis(app: app)

        XCTAssertTrue(app.navigationBars["Image Analysis"].waitForExistence(timeout: 3))

        let statsButton = app.buttons["Get Statistics"]
        let blurButton = app.buttons["Detect Blur"]
        let metadataButton = app.buttons["Get Metadata"]

        XCTAssertTrue(statsButton.waitForExistence(timeout: 3))
        XCTAssertTrue(blurButton.exists)
        XCTAssertTrue(metadataButton.exists)
    }

    @MainActor
    private func navigateToImageAnalysis(app: XCUIApplication) {
        let moreTab = app.tabBars.buttons["More"]
        if moreTab.waitForExistence(timeout: 5) {
            moreTab.tap()
        }

        let link = app.buttons["more-image-analysis-link"]
        if !link.exists {
            app.swipeUp()
        }
        if link.waitForExistence(timeout: 5) {
            link.tap()
        }
    }
}
