//
//  TensorOperationsView.swift
//  SwiftPixelUtilsExampleApp
//
//  Tensor operations demo view
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Tensor Operations Demo
struct TensorOperationsView: View {
    @State private var result = "Tap to test tensor operations"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Channel Operations") {
                    VStack(spacing: 12) {
                        Button("Extract Red Channel") {
                            Task { await extractChannel(index: 0, name: "Red") }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Extract Green Channel") {
                            Task { await extractChannel(index: 1, name: "Green") }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Extract Blue Channel") {
                            Task { await extractChannel(index: 2, name: "Blue") }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Shape Operations") {
                    VStack(spacing: 12) {
                        Button("Permute HWC → CHW") {
                            testPermute()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Squeeze") {
                            testSqueeze()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Unsqueeze (Add Batch)") {
                            testUnsqueeze()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Reshape") {
                            testReshape()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Patch Extraction") {
                    Button("Extract 64x64 Patch") {
                        Task { await extractPatch() }
                    }
                    .buttonStyle(.borderedProminent)
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .navigationTitle("Tensor Ops")
    }
    
    func extractChannel(index: Int, name: String) async {
        do {
            // Create sample RGB data (3x3 image)
            let data: [Float] = [
                // R, G, B for each pixel (HWC format)
                1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,
                0.5, 0.5, 0.0,  0.0, 0.5, 0.5,  0.5, 0.0, 0.5,
                1.0, 1.0, 1.0,  0.0, 0.0, 0.0,  0.5, 0.5, 0.5
            ]
            
            let channel = try TensorOperations.extractChannel(
                data: data,
                width: 3,
                height: 3,
                channels: 3,
                channelIndex: index,
                dataLayout: .hwc
            )
            
            result = """
            ✅ Channel Extraction
            Extracted: \(name) channel (index \(index))
            Input shape: [3, 3, 3] (HWC)
            Output size: \(channel.count)
            Output values: \(channel.map { String(format: "%.2f", $0) }.joined(separator: ", "))
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testPermute() {
        do {
            // Create sample data with shape [2, 3, 4] (H=2, W=3, C=4)
            let data: [Float] = Array(0..<24).map { Float($0) }
            
            let permuted = try TensorOperations.permute(
                data: data,
                shape: [2, 3, 4],
                order: [2, 0, 1]  // HWC -> CHW
            )
            
            result = """
            ✅ Permutation
            Input shape: [2, 3, 4] (HWC)
            Order: [2, 0, 1]
            Output shape: \(permuted.shape) (CHW)
            First 8 values: \(permuted.data.prefix(8).map { String(format: "%.0f", $0) }.joined(separator: ", "))
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testSqueeze() {
        let inputShape = [1, 2, 3]
        let squeezedShape = TensorOperations.squeeze(shape: inputShape, dims: [0])
        
        result = """
        ✅ Squeeze
        Input shape: \(inputShape)
        Squeeze dim: 0
        Output shape: \(squeezedShape)
        """
    }
    
    func testUnsqueeze() {
        let inputShape = [2, 3]
        let unsqueezedShape = TensorOperations.unsqueeze(shape: inputShape, dim: 0)
        
        result = """
        ✅ Unsqueeze
        Input shape: \(inputShape)
        Unsqueeze dim: 0
        Output shape: \(unsqueezedShape)
        """
    }
    
    func testReshape() {
        do {
            let data: [Float] = Array(0..<24).map { Float($0) }
            
            let reshaped = try TensorOperations.reshape(
                data: data,
                fromShape: [2, 3, 4],
                toShape: [4, 6]
            )
            
            result = """
            ✅ Reshape
            Input shape: [2, 3, 4]
            Output shape: \(reshaped.shape)
            Total elements: \(reshaped.data.count) (unchanged)
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func extractPatch() async {
        do {
            let imageData = try await downloadImageData(from: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg")
            let source = ImageSource.data(imageData)
            
            let pixelResult = try PixelExtractor.getPixelData(
                source: source,
                options: PixelDataOptions(
                    colorFormat: .rgb,
                    resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                    dataLayout: .hwc
                )
            )
            
            let patchOptions = PatchOptions(x: 50, y: 50, width: 64, height: 64)
            let patch = try TensorOperations.extractPatch(
                data: pixelResult.data,
                width: pixelResult.width,
                height: pixelResult.height,
                channels: pixelResult.channels,
                patchOptions: patchOptions,
                dataLayout: .hwc
            )
            
            result = """
            ✅ Patch Extraction
            Input: \(pixelResult.width)x\(pixelResult.height)
            Patch position: (50, 50)
            Patch size: \(patch.width)x\(patch.height)
            Channels: \(patch.channels)
            Data points: \(patch.data.count)
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
