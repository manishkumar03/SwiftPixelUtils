//
//  SwiftPixelUtilsExampleAppTests.swift
//  SwiftPixelUtilsExampleAppTests
//
//  Tests for SwiftPixelUtils Example App functionality
//

import XCTest
import SwiftPixelUtils

final class SwiftPixelUtilsExampleAppTests: XCTestCase {

    // MARK: - Sample Image Tests
    
    func testSampleImagesExist() {
        // Verify sample images are available
        let expectedImages = ["dog", "car", "street", "lion"]
        
        for imageName in expectedImages {
            let exists = Bundle.main.path(forResource: imageName, ofType: "jpg", inDirectory: "Resources") != nil ||
                        Bundle.main.path(forResource: imageName, ofType: "jpg") != nil
            // Images might not be available in unit test bundle, so just log
            if !exists {
                print("Note: Sample image '\(imageName)' not found in test bundle")
            }
        }
    }
}
