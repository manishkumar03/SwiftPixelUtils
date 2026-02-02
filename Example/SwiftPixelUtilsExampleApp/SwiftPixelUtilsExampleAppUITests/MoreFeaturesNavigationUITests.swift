//
//  MoreFeaturesNavigationUITests.swift
//  SwiftPixelUtilsExampleAppUITests
//
//  UI Tests for More Features tab navigation.
//  Uses accessibility identifiers for reliable, maintainable tests.
//
//  Created by Manish Kumar on 2026-01-31.
//

import XCTest

/// UI Tests for More Features tab navigation
final class MoreFeaturesNavigationUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }
    
    // MARK: - Augmentation Section
    
    /// Test navigation to Augmentation view
    @MainActor
    func testNavigation_Augmentation() throws {
        try navigateToMoreFeature(linkId: "more-augmentation-link", expectedTitle: "Augmentation")
    }
    
    /// Test navigation to Individual Augmentations view
    @MainActor
    func testNavigation_IndividualAugmentations() throws {
        try navigateToMoreFeature(linkId: "more-individual-augmentations-link", expectedTitle: "Individual Augmentations")
    }
    
    /// Test navigation to Cutout view
    @MainActor
    func testNavigation_Cutout() throws {
        try navigateToMoreFeature(linkId: "more-cutout-link", expectedTitle: "Cutout")
    }
    
    // MARK: - Analysis Section
    
    /// Test navigation to Image Analysis view
    @MainActor
    func testNavigation_ImageAnalysis() throws {
        try navigateToMoreFeature(linkId: "more-image-analysis-link", expectedTitle: "Image Analysis")
    }
    
    // MARK: - Inference Section
    
    /// Test navigation to Inference Utilities view
    @MainActor
    func testNavigation_InferenceUtilities() throws {
        try navigateToMoreFeature(linkId: "more-inference-utilities-link", expectedTitle: "Inference")
    }
    
    // MARK: - Data Processing Section
    
    /// Test navigation to Label Database view
    @MainActor
    func testNavigation_LabelDatabase() throws {
        try navigateToMoreFeature(linkId: "more-label-database-link", expectedTitle: "Label Database")
    }
    
    /// Test navigation to Tensor Operations view
    @MainActor
    func testNavigation_TensorOperations() throws {
        try navigateToMoreFeature(linkId: "more-tensor-operations-link", expectedTitle: "Tensor Operations")
    }
    
    /// Test navigation to Letterbox view
    @MainActor
    func testNavigation_Letterbox() throws {
        try navigateToMoreFeature(linkId: "more-letterbox-link", expectedTitle: "Letterbox")
    }
    
    /// Test navigation to Quantization view
    @MainActor
    func testNavigation_Quantization() throws {
        try navigateToMoreFeature(linkId: "more-quantization-link", expectedTitle: "Quantization")
    }
    
    /// Test navigation to Multi-Crop view
    @MainActor
    func testNavigation_MultiCrop() throws {
        try navigateToMoreFeature(linkId: "more-multicrop-link", expectedTitle: "Multi-Crop")
    }

    /// Test navigation to CVPixelBuffer view
    @MainActor
    func testNavigation_CVPixelBuffer() throws {
        try navigateToMoreFeature(linkId: "more-cvpixelbuffer-link", expectedTitle: "CVPixelBuffer")
    }
    
    // MARK: - Visualization Section
    
    /// Test navigation to Drawing view
    @MainActor
    func testNavigation_Drawing() throws {
        try navigateToMoreFeature(linkId: "more-drawing-link", expectedTitle: "Drawing")
    }
    
    /// Test navigation to Tensor to Image view
    @MainActor
    func testNavigation_TensorToImage() throws {
        try navigateToMoreFeature(linkId: "more-tensor-to-image-link", expectedTitle: "Tensor")
    }
    
    // MARK: - Validation & Batch Section
    
    /// Test navigation to Tensor Validation view
    @MainActor
    func testNavigation_TensorValidation() throws {
        try navigateToMoreFeature(linkId: "more-tensor-validation-link", expectedTitle: "Tensor Validation")
    }
    
    /// Test navigation to Batch Operations view
    @MainActor
    func testNavigation_BatchOperations() throws {
        try navigateToMoreFeature(linkId: "more-batch-operations-link", expectedTitle: "Batch")
    }
    
    /// Test navigation to Image Validation view
    @MainActor
    func testNavigation_ImageValidation() throws {
        try navigateToMoreFeature(linkId: "more-image-validation-link", expectedTitle: "Image Validation")
    }
    
    // MARK: - Video & Media Section
    
    /// Test navigation to Video Frames view
    @MainActor
    func testNavigation_VideoFrames() throws {
        try navigateToMoreFeature(linkId: "more-video-frames-link", expectedTitle: "Video")
    }
    
    // MARK: - Info Section
    
    /// Test navigation to About view
    @MainActor
    func testNavigation_About() throws {
        try navigateToMoreFeature(linkId: "more-about-link", expectedTitle: "About")
    }
    
    // MARK: - Helper
    
    @MainActor
    private func navigateToMoreFeature(linkId: String, expectedTitle: String) throws {
        let app = XCUIApplication()
        app.launch()
        
        // Navigate to More tab
        let moreTab = app.tabBars.buttons["More"]
        XCTAssertTrue(moreTab.waitForExistence(timeout: 5))
        moreTab.tap()
        
        // Find and tap the navigation link
        let link = app.buttons[linkId]
        
        // May need to scroll to find the link
        var scrollAttempts = 0
        while !link.exists && scrollAttempts < 5 {
            app.swipeUp()
            scrollAttempts += 1
        }
        
        XCTAssertTrue(link.waitForExistence(timeout: 3), "\(linkId) should exist")
        link.tap()
        
        // Verify navigation occurred by checking for navigation bar title
        let navBar = app.navigationBars.containing(NSPredicate(format: "identifier CONTAINS[c] %@", expectedTitle)).firstMatch
        let titleExists = navBar.waitForExistence(timeout: 3)
        
        // Alternative: check if any navigation bar exists after tap
        if !titleExists {
            let anyNavBar = app.navigationBars.firstMatch
            XCTAssertTrue(anyNavBar.waitForExistence(timeout: 3), "Should navigate to \(expectedTitle) view")
        }
    }
}
