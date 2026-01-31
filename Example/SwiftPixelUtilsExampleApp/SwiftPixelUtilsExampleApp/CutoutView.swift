import SwiftUI
import SwiftPixelUtils

struct CutoutView: View {
    @State private var result = "Tap to test cutout augmentation"
    @State private var previewImage: PlatformImage?
    @State private var showImagePreview = false
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Cutout Augmentation") {
                    VStack(spacing: 12) {
                        Button("Single Cutout") {
                            Task { await testSingleCutout() }
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Multiple Cutouts") {
                            Task { await testMultipleCutouts() }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Random Cutouts") {
                            Task { await testRandomCutouts() }
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
        .navigationTitle("Cutout")
        .sheet(isPresented: $showImagePreview) {
            ImagePreviewSheet(image: previewImage, isPresented: $showImagePreview)
        }
    }
    
    func testSingleCutout() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CutoutOptions(
                numCutouts: 1,
                minSize: 0.05,
                maxSize: 0.10,
                fillMode: .constant,
                fillValue: [0, 0, 0]  // Black fill
            )
            let cutout = try await ImageAugmentor.cutout(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Single Cutout Applied
            Cutouts: 1
            Area ratio: 5-10%
            Fill mode: Constant (black)
            Output size: \(cutout.size.width)x\(cutout.size.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = cutout
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testMultipleCutouts() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CutoutOptions(
                numCutouts: 3,
                minSize: 0.03,
                maxSize: 0.08,
                fillMode: .constant,
                fillValue: [128, 128, 128]  // Gray fill
            )
            let cutout = try await ImageAugmentor.cutout(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Multiple Cutouts Applied
            Number of cutouts: 3
            Area ratio: 3-8%
            Fill mode: Constant (gray)
            Output size: \(cutout.size.width)x\(cutout.size.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = cutout
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testRandomCutouts() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CutoutOptions(
                numCutouts: 5,
                minSize: 0.02,
                maxSize: 0.05,
                fillMode: .random,
                seed: 42
            )
            let cutout = try await ImageAugmentor.cutout(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Random Cutouts Applied
            Number of cutouts: 5
            Area ratio: 2-5%
            Fill mode: Random colors
            Seed: 42 (reproducible)
            Output size: \(cutout.size.width)x\(cutout.size.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = cutout
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
