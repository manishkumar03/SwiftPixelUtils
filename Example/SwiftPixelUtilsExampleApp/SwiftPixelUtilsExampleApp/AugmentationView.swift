//
//  AugmentationView.swift
//  SwiftPixelUtilsExampleApp
//
//  Image augmentation demo tab
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Augmentation Demo
struct AugmentationView: View {
    @State private var result = "Tap to apply augmentations"
    @State private var previewImage: PlatformImage?
    @State private var showImagePreview = false
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Single Augmentations") {
                        VStack(spacing: 12) {
                            Button("Rotate 45°") {
                                Task { await applyAugmentation(AugmentationOptions(rotation: 45)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Horizontal Flip") {
                                Task { await applyAugmentation(AugmentationOptions(horizontalFlip: true)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Brightness +20%") {
                                Task { await applyAugmentation(AugmentationOptions(brightness: 0.2)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Contrast +30%") {
                                Task { await applyAugmentation(AugmentationOptions(contrast: 0.3)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Gaussian Blur") {
                                Task { await applyAugmentation(AugmentationOptions(blur: BlurOptions(type: .gaussian, radius: 5))) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Saturation -20%") {
                                Task { await applyAugmentation(AugmentationOptions(saturation: -0.2)) }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Combined Augmentations") {
                        Button("Rotate + Flip + Brightness") {
                            Task {
                                await applyAugmentation(AugmentationOptions(
                                    rotation: 15,
                                    horizontalFlip: true,
                                    brightness: 0.1
                                ))
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    GroupBox("Color Jitter") {
                        Button("Apply Color Jitter") {
                            Task { await applyColorJitter() }
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
            .navigationTitle("Augmentation")
            .sheet(isPresented: $showImagePreview) {
                ImagePreviewSheet(image: previewImage, isPresented: $showImagePreview)
            }
        }
    }
    
    func applyAugmentation(_ options: AugmentationOptions) async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            let augmented = try ImageAugmentor.applyAugmentations(to: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            let size = augmented.size
            result = """
            ✅ Augmentation Applied!
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = augmented
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func applyColorJitter() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            let jittered = try ImageAugmentor.colorJitter(
                source: source,
                options: ColorJitterOptions(
                    brightness: 0.2,
                    contrast: 0.2,
                    saturation: 0.3,
                    hue: 0.1
                )
            )
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Color Jitter Applied!
            Brightness: ±0.2
            Contrast: ±0.2
            Saturation: ±0.3
            Hue: ±0.1
            Applied values:
              Brightness: \(String(format: "%.3f", jittered.appliedBrightness))
              Contrast: \(String(format: "%.3f", jittered.appliedContrast))
              Saturation: \(String(format: "%.3f", jittered.appliedSaturation))
              Hue: \(String(format: "%.3f", jittered.appliedHue))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            previewImage = jittered.image
            showImagePreview = true
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Image Preview Sheet
struct ImagePreviewSheet: View {
    let image: PlatformImage?
    @Binding var isPresented: Bool
    
    var body: some View {
        NavigationView {
            VStack {
                if let image = image {
                    #if canImport(UIKit)
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding()
                    #else
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding()
                    #endif
                } else {
                    Text("No image")
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            .navigationTitle("Output Image")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("OK") {
                        isPresented = false
                    }
                }
            }
        }
    }
}

// MARK: - Multi-Image Preview Sheet
struct MultiImagePreviewSheet: View {
    let images: [PlatformImage]
    @Binding var isPresented: Bool
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150))], spacing: 12) {
                    ForEach(Array(images.enumerated()), id: \.offset) { index, image in
                        VStack {
                            #if canImport(UIKit)
                            Image(uiImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 150)
                            #else
                            Image(nsImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 150)
                            #endif
                            Text("\(index + 1)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Output Images (\(images.count))")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("OK") {
                        isPresented = false
                    }
                }
            }
        }
    }
}
