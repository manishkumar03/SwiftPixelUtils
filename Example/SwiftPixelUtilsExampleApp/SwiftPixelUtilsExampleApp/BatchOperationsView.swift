import SwiftUI
import SwiftPixelUtils

struct BatchOperationsView: View {
    @State private var result = "Tap to test batch operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Batch Pixel Extraction") {
                    Button("Batch Get Pixel Data") {
                        Task { await testBatchPixelExtraction() }
                    }
                    .buttonStyle(.borderedProminent)
                }
                
                GroupBox("Batch Assembly") {
                    VStack(spacing: 12) {
                        Button("Assemble Batch Tensor") {
                            Task { await testBatchAssembly() }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("TensorOperations.assembleBatch") {
                            Task { await testTensorBatchAssembly() }
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
        .navigationTitle("Batch Ops")
    }
    
    func testBatchPixelExtraction() async {
        do {
            // Download images first
            let imageData = try await downloadImageData(from: sampleImageURL)
            let sources = [
                ImageSource.data(imageData),
                ImageSource.data(imageData)
            ]
            
            let start = CFAbsoluteTimeGetCurrent()
            let options = PixelDataOptions(
                colorFormat: .rgb,
                resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                dataLayout: .chw
            )
            
            let results = try PixelExtractor.batchGetPixelData(sources: sources, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Batch Pixel Extraction
            Images processed: \(results.count)
            Each image: \(results.first?.width ?? 0)x\(results.first?.height ?? 0)
            Channels: \(results.first?.channels ?? 0)
            Layout: CHW
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testBatchAssembly() async {
        do {
            // Download images first
            let imageData = try await downloadImageData(from: sampleImageURL)
            let sources = [
                ImageSource.data(imageData),
                ImageSource.data(imageData)
            ]
            
            let options = PixelDataOptions(
                colorFormat: .rgb,
                resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                dataLayout: .chw
            )
            
            let start = CFAbsoluteTimeGetCurrent()
            let pixelResults = try PixelExtractor.batchGetPixelData(sources: sources, options: options)
            let batch = try BatchAssembly.assemble(pixelResults)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Batch Assembly
            Batch size: \(batch.batchSize)
            Shape: \(batch.shape)
            Total elements: \(batch.data.count)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testTensorBatchAssembly() async {
        do {
            // Download images first
            let imageData = try await downloadImageData(from: sampleImageURL)
            let sources = [
                ImageSource.data(imageData),
                ImageSource.data(imageData)
            ]
            
            let options = PixelDataOptions(
                colorFormat: .rgb,
                resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                dataLayout: .chw
            )
            
            let start = CFAbsoluteTimeGetCurrent()
            let pixelResults = try PixelExtractor.batchGetPixelData(sources: sources, options: options)
            let batch = try TensorOperations.assembleBatch(results: pixelResults, layout: .nchw)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ TensorOperations.assembleBatch
            Batch size: \(batch.batchSize)
            Shape: \(batch.shape)
            Layout: \(batch.layout)
            Total elements: \(batch.data.count)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
