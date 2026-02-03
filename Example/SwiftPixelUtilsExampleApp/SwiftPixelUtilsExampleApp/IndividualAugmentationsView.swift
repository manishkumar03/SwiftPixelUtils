//
//  IndividualAugmentationsView.swift
//  SwiftPixelUtilsExampleApp
//
//  Individual augmentation function demos
//

import SwiftUI
import SwiftPixelUtils

struct IndividualAugmentationsView: View {
    @State private var result = "Tap to apply individual augmentations"
    @State private var previewImage: PlatformImage?
    @State private var showImagePreview = false
    
    private let sampleImageURL = "https://picsum.photos/seed/vision1/400/400"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Rotation") {
                    HStack(spacing: 12) {
                        Button("45°") {
                            Task { await testRotate(degrees: 45) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("90°") {
                            Task { await testRotate(degrees: 90) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("180°") {
                            Task { await testRotate(degrees: 180) }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Flip") {
                    HStack(spacing: 12) {
                        Button("Horizontal") {
                            Task { await testFlipHorizontal() }
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Vertical") {
                            Task { await testFlipVertical() }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                
                GroupBox("Brightness") {
                    HStack(spacing: 12) {
                        Button("-30%") {
                            Task { await testBrightness(factor: -0.3) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("+30%") {
                            Task { await testBrightness(factor: 0.3) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("+50%") {
                            Task { await testBrightness(factor: 0.5) }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Contrast") {
                    HStack(spacing: 12) {
                        Button("-30%") {
                            Task { await testContrast(factor: -0.3) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("+30%") {
                            Task { await testContrast(factor: 0.3) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("+50%") {
                            Task { await testContrast(factor: 0.5) }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Saturation") {
                    HStack(spacing: 12) {
                        Button("-50%") {
                            Task { await testSaturation(factor: -0.5) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("+30%") {
                            Task { await testSaturation(factor: 0.3) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("+80%") {
                            Task { await testSaturation(factor: 0.8) }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Blur") {
                    VStack(spacing: 12) {
                        HStack(spacing: 12) {
                            Button("Gaussian r=3") {
                                Task { await testBlur(type: .gaussian, radius: 3) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Gaussian r=8") {
                                Task { await testBlur(type: .gaussian, radius: 8) }
                            }
                            .buttonStyle(.bordered)
                        }
                        
                        HStack(spacing: 12) {
                            Button("Box r=5") {
                                Task { await testBlur(type: .box, radius: 5) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Box r=10") {
                                Task { await testBlur(type: .box, radius: 10) }
                            }
                            .buttonStyle(.bordered)
                        }
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
        .navigationTitle("Individual Augmentations")
        .sheet(isPresented: $showImagePreview) {
            ImagePreviewSheet(image: previewImage, isPresented: $showImagePreview)
        }
    }
    
    func testRotate(degrees: Double) async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            let rotated = try ImageAugmentor.rotate(source: source, degrees: Float(degrees))
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let size = rotated.size
            
            result = """
            ✅ ImageAugmentor.rotate()
            Rotation: \(Int(degrees))°
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = rotated
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testFlipHorizontal() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            let flipped = try ImageAugmentor.flipHorizontal(source: source)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let size = flipped.size
            
            result = """
            ✅ ImageAugmentor.flipHorizontal()
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = flipped
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testFlipVertical() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            let flipped = try ImageAugmentor.flipVertical(source: source)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let size = flipped.size
            
            result = """
            ✅ ImageAugmentor.flipVertical()
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = flipped
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testBrightness(factor: Double) async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            let adjusted = try ImageAugmentor.adjustBrightness(source: source, value: Float(factor))
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let size = adjusted.size
            
            result = """
            ✅ ImageAugmentor.adjustBrightness()
            Factor: \(factor > 0 ? "+" : "")\(Int(factor * 100))%
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = adjusted
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testContrast(factor: Double) async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            let adjusted = try ImageAugmentor.adjustContrast(source: source, value: Float(factor))
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let size = adjusted.size
            
            result = """
            ✅ ImageAugmentor.adjustContrast()
            Factor: \(factor > 0 ? "+" : "")\(Int(factor * 100))%
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = adjusted
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testSaturation(factor: Double) async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            let adjusted = try ImageAugmentor.adjustSaturation(source: source, value: Float(factor))
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let size = adjusted.size
            
            result = """
            ✅ ImageAugmentor.adjustSaturation()
            Factor: \(factor > 0 ? "+" : "")\(Int(factor * 100))%
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = adjusted
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testBlur(type: BlurType, radius: Double) async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            // Note: blur() only takes radius. For type-specific blur, use augment() with BlurOptions
            let blurred = try ImageAugmentor.blur(
                source: source,
                radius: Float(radius)
            )
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let size = blurred.size
            
            result = """
            ✅ ImageAugmentor.blur()
            Type: \(type.rawValue)
            Radius: \(Int(radius))
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = blurred
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
