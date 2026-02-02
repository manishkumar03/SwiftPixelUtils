import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

final class ImageAnalyzerColorSpaceTests: XCTestCase {
    
    func createTestImage(colorSpaceName: CFString) -> CGImage? {
        guard let colorSpace = CGColorSpace(name: colorSpaceName) else { return nil }
        
        let width = 10
        let height = 10
        
        // CMYK
        if colorSpace.model == .cmyk {
            // Note: Creating CMYK contexts can be platform dependent or fail if not supported
            // We'll try, but handle failure gracefully in tests
            let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            )
            return context?.makeImage()
        }
        
        // Grayscale
        if colorSpace.model == .monochrome {
             let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            )
            return context?.makeImage()
        }
        
        // RGB or others
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        
        return context?.makeImage()
    }
    
    func testGetMetadataColorSpaces() async throws {
        // sRGB
        if let image = createTestImage(colorSpaceName: CGColorSpace.sRGB) {
            let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
            XCTAssertEqual(metadata.colorSpace, "sRGB")
        } else {
            XCTFail("Could not create sRGB image")
        }
        
        // Display P3
        if let image = createTestImage(colorSpaceName: CGColorSpace.displayP3) {
            let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
            XCTAssertEqual(metadata.colorSpace, "Display P3")
        } 
        
        // Adobe RGB
        if let image = createTestImage(colorSpaceName: CGColorSpace.adobeRGB1998) {
            let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
            XCTAssertEqual(metadata.colorSpace, "Adobe RGB")
        }
        
        // Grayscale
        if let image = createTestImage(colorSpaceName: CGColorSpace.genericGrayGamma2_2) {
            let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
            XCTAssertEqual(metadata.colorSpace, "Grayscale")
        }
        
        // CMYK
        if let image = createTestImage(colorSpaceName: CGColorSpace.genericCMYK) {
            let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
            XCTAssertEqual(metadata.colorSpace, "CMYK")
        }
    }
}
