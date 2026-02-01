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
            
            ImageAnalysisView()
                .tabItem {
                    Label("Analyze", systemImage: "chart.bar.xaxis")
                }
            
            AugmentationView()
                .tabItem {
                    Label("Augment", systemImage: "wand.and.stars")
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
    @State private var selectedFramework: InferenceFramework = .tflite
    
    enum InferenceFramework: String, CaseIterable {
        case tflite = "TFLite"
        case execuTorch = "ExecuTorch"
        
        var icon: String {
            switch self {
            case .tflite: return "t.square.fill"
            case .execuTorch: return "bolt.fill"
            }
        }
        
        var color: Color {
            switch self {
            case .tflite: return .blue
            case .execuTorch: return .orange
            }
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Framework Selector
            Picker("Framework", selection: $selectedFramework) {
                ForEach(InferenceFramework.allCases, id: \.self) { framework in
                    Label(framework.rawValue, systemImage: framework.icon)
                        .tag(framework)
                }
            }
            .pickerStyle(.segmented)
            .padding()
            
            // Content based on selection
            switch selectedFramework {
            case .tflite:
                TFLiteInferenceView()
            case .execuTorch:
                ExecuTorchInferenceView()
            }
        }
        .navigationTitle("Image Classification")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    ContentView()
}
