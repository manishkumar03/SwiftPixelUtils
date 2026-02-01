import XCTest
@testable import SwiftPixelUtils

/// Tests for ``PixelUtilsError`` - Error types and handling.
///
/// ## Topics
///
/// ### Error Case Tests
/// - invalidSource, loadFailed, processingFailed
/// - conversionFailed, dimensionMismatch
/// - invalidOptions, videoError, emptyBatch
///
/// ### Error Description Tests
/// - LocalizedError conformance
/// - Descriptive error messages with context
///
/// ### Equatable Tests
/// - Error comparison for testing purposes
final class ErrorsTests: XCTestCase {
    
    // MARK: - Error Case Tests
    
    func testInvalidSourceError() {
        let error = PixelUtilsError.invalidSource("Image is nil")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Invalid source") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("Image is nil") ?? false)
    }
    
    func testLoadFailedError() {
        let error = PixelUtilsError.loadFailed("File not found")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Load failed") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("File not found") ?? false)
    }
    
    func testInvalidROIError() {
        let error = PixelUtilsError.invalidROI("ROI outside image bounds")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Invalid ROI") ?? false)
    }
    
    func testProcessingFailedError() {
        let error = PixelUtilsError.processingFailed("Conversion failed")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Processing failed") ?? false)
    }
    
    func testInvalidOptionsError() {
        let error = PixelUtilsError.invalidOptions("Missing mean for custom normalization")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Invalid options") ?? false)
    }
    
    func testInvalidChannelError() {
        let error = PixelUtilsError.invalidChannel("Channel index 5 out of range")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Invalid channel") ?? false)
    }
    
    func testInvalidPatchError() {
        let error = PixelUtilsError.invalidPatch("Patch exceeds image bounds")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Invalid patch") ?? false)
    }
    
    func testDimensionMismatchError() {
        let error = PixelUtilsError.dimensionMismatch("Expected 4 dimensions, got 3")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Dimension mismatch") ?? false)
    }
    
    func testEmptyBatchError() {
        let error = PixelUtilsError.emptyBatch("Cannot create batch from empty array")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Empty batch") ?? false)
    }
    
    func testUnknownError() {
        let error = PixelUtilsError.unknown("Unexpected error occurred")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Unknown error") ?? false)
    }
    
    // MARK: - Error Protocol Conformance Tests
    
    func testErrorConformsToError() {
        let error: Error = PixelUtilsError.invalidSource("test")
        XCTAssertNotNil(error)
    }
    
    func testErrorConformsToLocalizedError() {
        let error: LocalizedError = PixelUtilsError.invalidSource("test")
        XCTAssertNotNil(error.errorDescription)
    }
    
    // MARK: - Error Catching Tests
    
    func testCatchSpecificError() {
        func throwError() throws {
            throw PixelUtilsError.invalidSource("test")
        }
        
        do {
            try throwError()
            XCTFail("Should have thrown")
        } catch PixelUtilsError.invalidSource(let message) {
            XCTAssertEqual(message, "test")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }
    
    func testCatchAllPixelUtilsErrors() {
        let errors: [PixelUtilsError] = [
            .invalidSource("a"),
            .loadFailed("b"),
            .invalidROI("c"),
            .processingFailed("d"),
            .invalidOptions("e"),
            .invalidChannel("f"),
            .invalidPatch("g"),
            .dimensionMismatch("h"),
            .emptyBatch("i"),
            .unknown("j")
        ]
        
        for error in errors {
            do {
                throw error
            } catch let e as PixelUtilsError {
                XCTAssertNotNil(e.errorDescription)
            } catch {
                XCTFail("Should catch as PixelUtilsError")
            }
        }
    }
    
    // MARK: - Error Description Tests
    
    func testAllErrorsHaveDescriptions() {
        let errors: [PixelUtilsError] = [
            .invalidSource("msg"),
            .loadFailed("msg"),
            .invalidROI("msg"),
            .processingFailed("msg"),
            .invalidOptions("msg"),
            .invalidChannel("msg"),
            .invalidPatch("msg"),
            .dimensionMismatch("msg"),
            .emptyBatch("msg"),
            .unknown("msg")
        ]
        
        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription?.isEmpty ?? true)
        }
    }
    
    func testErrorDescriptionContainsMessage() {
        let customMessage = "Custom error message 12345"
        let error = PixelUtilsError.processingFailed(customMessage)
        
        XCTAssertTrue(error.errorDescription?.contains(customMessage) ?? false)
    }
    
    // MARK: - Empty Message Tests
    
    func testEmptyMessageError() {
        let error = PixelUtilsError.invalidSource("")
        
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Invalid source") ?? false)
    }
    
    // MARK: - Special Characters Tests
    
    func testErrorWithSpecialCharacters() {
        let message = "Error: file \"test.jpg\" not found at /path/to/file"
        let error = PixelUtilsError.loadFailed(message)
        
        XCTAssertTrue(error.errorDescription?.contains(message) ?? false)
    }
    
    func testErrorWithUnicode() {
        let message = "文件未找到 🚫"
        let error = PixelUtilsError.invalidSource(message)
        
        XCTAssertTrue(error.errorDescription?.contains(message) ?? false)
    }
    
    // MARK: - Error Equality Tests (by pattern matching)
    
    func testErrorPatternMatching() {
        let error1 = PixelUtilsError.invalidSource("msg1")
        let error2 = PixelUtilsError.invalidSource("msg2")
        let error3 = PixelUtilsError.loadFailed("msg1")
        
        // Same case, different message
        if case .invalidSource = error1, case .invalidSource = error2 {
            // Both are invalidSource
        } else {
            XCTFail("Pattern matching failed")
        }
        
        // Different cases
        if case .invalidSource = error1, case .loadFailed = error3 {
            // error1 is invalidSource, error3 is loadFailed
        } else {
            XCTFail("Pattern matching failed")
        }
    }
    
    // MARK: - Real-world Usage Simulation Tests
    
    func testThrowingFunction() {
        func validateImage(hasContent: Bool) throws {
            guard hasContent else {
                throw PixelUtilsError.invalidSource("Image has no content")
            }
        }
        
        XCTAssertNoThrow(try validateImage(hasContent: true))
        XCTAssertThrowsError(try validateImage(hasContent: false))
    }
    
    func testMultipleErrorHandling() {
        enum TestOperation {
            case load, process, convert
        }
        
        func performOperation(_ op: TestOperation) throws {
            switch op {
            case .load:
                throw PixelUtilsError.loadFailed("Network error")
            case .process:
                throw PixelUtilsError.processingFailed("Memory error")
            case .convert:
                throw PixelUtilsError.dimensionMismatch("Shape error")
            }
        }
        
        do {
            try performOperation(.load)
        } catch PixelUtilsError.loadFailed {
            // Expected
        } catch {
            XCTFail("Wrong error")
        }
        
        do {
            try performOperation(.process)
        } catch PixelUtilsError.processingFailed {
            // Expected
        } catch {
            XCTFail("Wrong error")
        }
    }
}
