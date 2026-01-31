import SwiftUI
import SwiftPixelUtils

struct TensorToImageView: View {
    @State private var result = "Tap to test tensor to image conversion"
    @State private var previewImage: PlatformImage?
    @State private var showImagePreview = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Convert Tensor to Image") {
                    VStack(spacing: 12) {
                        Button("RGB Tensor → Image") {
                            Task { await convertRGBTensor() }
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Grayscale Tensor → Image") {
                            Task { await convertGrayscaleTensor() }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .navigationTitle("Tensor → Image")
        .sheet(isPresented: $showImagePreview) {
            ImagePreviewSheet(image: previewImage, isPresented: $showImagePreview)
        }
    }
    
    func convertRGBTensor() async {
        do {
            // Create a simple 4x4 RGB gradient tensor (HWC format)
            var tensor: [Float] = []
            for y in 0..<4 {
                for x in 0..<4 {
                    let r = Float(x) / 3.0
                    let g = Float(y) / 3.0
                    let b = Float(x + y) / 6.0
                    tensor.append(contentsOf: [r, g, b])
                }
            }
            
            let start = CFAbsoluteTimeGetCurrent()
            let options = TensorToImageOptions(
                channels: 3,
                dataLayout: .hwc,
                denormalize: false
            )
            let converted = try await TensorToImage.convert(data: tensor, width: 4, height: 4, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ RGB Tensor Converted
            Input: 4x4x3 tensor (HWC)
            Output size: \(converted.width)x\(converted.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            #if canImport(UIKit)
            previewImage = UIImage(cgImage: converted.cgImage)
            #else
            previewImage = NSImage(cgImage: converted.cgImage, size: NSSize(width: converted.width, height: converted.height))
            #endif
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func convertGrayscaleTensor() async {
        do {
            // Create a simple 8x8 grayscale gradient tensor
            var tensor: [Float] = []
            for y in 0..<8 {
                for x in 0..<8 {
                    let value = Float(x + y) / 14.0
                    tensor.append(value)
                }
            }
            
            let start = CFAbsoluteTimeGetCurrent()
            let options = TensorToImageOptions(
                channels: 1,
                dataLayout: .hwc,
                denormalize: false
            )
            let converted = try await TensorToImage.convert(data: tensor, width: 8, height: 8, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Grayscale Tensor Converted
            Input: 8x8x1 tensor
            Output size: \(converted.width)x\(converted.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            #if canImport(UIKit)
            previewImage = UIImage(cgImage: converted.cgImage)
            #else
            previewImage = NSImage(cgImage: converted.cgImage, size: NSSize(width: converted.width, height: converted.height))
            #endif
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
