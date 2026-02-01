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
                TFLiteInferenceView()
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

#Preview {
    ContentView()
}
