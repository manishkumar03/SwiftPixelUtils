//
//  CVPixelBufferFormatsView.swift
//  SwiftPixelUtilsExampleApp
//
//  CVPixelBuffer format conversion demo (including RGB565)
//
//  Created by Manish Kumar on 2026-02-02.
//

import SwiftUI
import SwiftPixelUtils
import CoreVideo

struct CVPixelBufferFormatsView: View {
    @State private var result = "Tap a button to test CVPixelBuffer conversion"

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("RGB565 Conversion") {
                    VStack(spacing: 12) {
                        Button("Test RGB565 (Little Endian)") {
                            testRGB565(littleEndian: true)
                        }
                        .buttonStyle(.borderedProminent)
                        .accessibilityIdentifier("cvpixelbuffer-test-rgb565-le")

                        Button("Test RGB565 (Big Endian)") {
                            testRGB565(littleEndian: false)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("cvpixelbuffer-test-rgb565-be")
                    }
                }

                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityIdentifier("cvpixelbuffer-result-text")
                }
            }
            .padding()
        }
        .navigationTitle("CVPixelBuffer")
    }

    private func testRGB565(littleEndian: Bool) {
        do {
            let result = try createRGB565PixelBuffer(width: 2, height: 2, r5: 31, g6: 0, b5: 0, littleEndian: littleEndian)
            let resultData = try CVPixelBufferUtilities.toTensorData(result.buffer)

            let endianLabel = result.actualLittleEndian ? "LE" : "BE"
            let fallbackNote = result.actualLittleEndian != littleEndian ? " (fallback to LE)" : ""

            self.result = """
            ✅ RGB565 Conversion
            Endian: \(endianLabel)\(fallbackNote)
            Size: \(resultData.tensorWidth)x\(resultData.tensorHeight)
            Channels: \(resultData.channels)
            First pixel (RGB): [\(String(format: "%.2f", resultData.data[0])), \(String(format: "%.2f", resultData.data[1])), \(String(format: "%.2f", resultData.data[2]))]
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }

    private func createRGB565PixelBuffer(
        width: Int,
        height: Int,
        r5: UInt16,
        g6: UInt16,
        b5: UInt16,
        littleEndian: Bool
    ) throws -> (buffer: CVPixelBuffer, actualLittleEndian: Bool) {
        let preferredFormat: OSType = littleEndian ? kCVPixelFormatType_16LE565 : kCVPixelFormatType_16BE565
        var pixelBuffer: CVPixelBuffer?
        var status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            preferredFormat,
            nil,
            &pixelBuffer
        )

        var actualLittleEndian = littleEndian

        if status != kCVReturnSuccess || pixelBuffer == nil {
            // Some devices may not support BE565; fall back to LE565 for demo
            status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                width,
                height,
                kCVPixelFormatType_16LE565,
                nil,
                &pixelBuffer
            )
            actualLittleEndian = true
        }

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw PixelUtilsError.processingFailed("Failed to create RGB565 pixel buffer")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw PixelUtilsError.processingFailed("Failed to get base address")
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
        let pixelsPerRow = bytesPerRow / MemoryLayout<UInt16>.size

        let packed = (r5 << 11) | (g6 << 5) | b5
        let stored = actualLittleEndian ? CFSwapInt16HostToLittle(packed) : CFSwapInt16HostToBig(packed)

        for y in 0..<height {
            for x in 0..<width {
                ptr[y * pixelsPerRow + x] = stored
            }
        }

        return (buffer: buffer, actualLittleEndian: actualLittleEndian)
    }
}

#Preview {
    NavigationStack {
        CVPixelBufferFormatsView()
    }
}
