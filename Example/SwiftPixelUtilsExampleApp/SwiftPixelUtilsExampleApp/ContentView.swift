//
//  ContentView.swift
//  SwiftPixelUtilsExampleApp
//
//  Main tab view that connects all feature views
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            NavigationStack {
                InferenceTabView()
            }
            .tabItem {
                Label("Inference", systemImage: "brain")
            }
            
            PixelExtractionView()
                .tabItem {
                    Label("Pixels", systemImage: "square.grid.3x3")
                }
            
            BoundingBoxView()
                .tabItem {
                    Label("Boxes", systemImage: "rectangle.dashed")
                }
            
            MoreFeaturesView()
                .tabItem {
                    Label("More", systemImage: "ellipsis.circle")
                }
        }
    }
}

// MARK: - Inference Tab with Framework Selection
struct InferenceTabView: View {
    var body: some View {
        List {
            // MARK: - Classification Section
            Section {
                NavigationLink {
                    TFLiteClassificationView()
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: "t.square.fill")
                            .font(.title2)
                            .foregroundColor(.blue)
                            .frame(width: 32)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("TensorFlow Lite")
                                .font(.headline)
                            Text("MobileNet V2 • Quantized INT8")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .accessibilityIdentifier("inference-tflite-classification-link")
                
                NavigationLink {
                    ExecuTorchClassificationView()
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: "bolt.fill")
                            .font(.title2)
                            .foregroundColor(.orange)
                            .frame(width: 32)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("ExecuTorch")
                                .font(.headline)
                            Text("MobileNet V2 • Float32 • NCHW")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .accessibilityIdentifier("inference-executorch-classification-link")
            } header: {
                Label("Image Classification", systemImage: "photo.badge.checkmark")
            } footer: {
                Text("Classify images into 1000 ImageNet categories")
            }
            
            // MARK: - Object Detection Section
            Section {
                NavigationLink {
                    YOLODetectionView()
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: "viewfinder.rectangular")
                            .font(.title2)
                            .foregroundColor(.green)
                            .frame(width: 32)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("YOLOv5")
                                .font(.headline)
                            Text("TFLite • 80 COCO Classes • FP16")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .accessibilityIdentifier("inference-yolo-detection-link")
            } header: {
                Label("Object Detection", systemImage: "viewfinder.rectangular")
            } footer: {
                Text("Detect and locate objects with bounding boxes")
            }
            
            // MARK: - Semantic Segmentation Section
            Section {
                NavigationLink {
                    SegmentationView()
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: "square.grid.3x3.fill")
                            .font(.title2)
                            .foregroundColor(.purple)
                            .frame(width: 32)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("DeepLabV3")
                                .font(.headline)
                            Text("TFLite • 21 VOC Classes • Float32")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .accessibilityIdentifier("inference-segmentation-link")
            } header: {
                Label("Semantic Segmentation", systemImage: "square.grid.3x3.fill")
            } footer: {
                Text("Pixel-wise classification into semantic categories")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("ML Inference")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    ContentView()
}
