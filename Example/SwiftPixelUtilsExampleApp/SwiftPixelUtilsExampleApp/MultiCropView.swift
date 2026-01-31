import SwiftUI
import SwiftPixelUtils

struct MultiCropView: View {
    @State private var result = "Tap to test multi-crop operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Standard Crops") {
                    VStack(spacing: 12) {
                        Button("Five Crop (4 corners + center)") {
                            Task { await testFiveCrop() }
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Ten Crop (5 + flips)") {
                            Task { await testTenCrop() }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Grid Extraction") {
                    VStack(spacing: 12) {
                        Button("Extract 3x3 Grid") {
                            Task { await testGridExtraction() }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Random Crop") {
                    Button("5 Random 100x100 Crops") {
                        Task { await testRandomCrop() }
                    }
                    .buttonStyle(.bordered)
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .navigationTitle("Multi-Crop")
    }
    
    func testFiveCrop() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CropOptions(width: 224, height: 224)
            let crops = try await MultiCropOperations.fiveCrop(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Five Crop
            Crop size: 224x224
            Crops extracted: \(crops.crops.count)
            
            Positions:
            \(crops.positions.enumerated().map { i, pos in
                "• Crop \(i + 1): (\(pos.x), \(pos.y))"
            }.joined(separator: "\n"))
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testTenCrop() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CropOptions(width: 224, height: 224)
            let crops = try await MultiCropOperations.tenCrop(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Ten Crop
            Crop size: 224x224
            Crops extracted: \(crops.crops.count)
            (5 original + 5 horizontal flips)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testGridExtraction() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = GridOptions(columns: 3, rows: 3)
            let grid = try await MultiCropOperations.extractGrid(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Grid Extraction
            Grid: \(grid.columns)x\(grid.rows) = \(grid.patches.count) patches
            Patch dimensions: \(grid.patches.first?.pixelData.width ?? 0)x\(grid.patches.first?.pixelData.height ?? 0)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testRandomCrop() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = RandomCropOptions(width: 100, height: 100, count: 5, seed: 42)
            let crops = try await MultiCropOperations.randomCrop(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Random Crop
            Crop size: 100x100
            Count: \(crops.crops.count)
            Seed: 42 (reproducible)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
