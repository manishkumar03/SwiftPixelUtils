// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftPixelUtils",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .tvOS(.v15),
        .watchOS(.v8)
    ],
    products: [
        // High-performance image preprocessing library for ML/AI inference pipelines
        // Native implementations using Core Image, Accelerate, and Core ML
        .library(
            name: "SwiftPixelUtils",
            targets: ["SwiftPixelUtils"]
        ),
    ],
    dependencies: [
        // No external dependencies - uses only Apple frameworks:
        // - CoreImage for image processing
        // - Accelerate for high-performance math operations
        // - CoreGraphics for image manipulation
        // - CoreML for ML model integration (optional)
    ],
    targets: [
        // Main library target containing all image processing utilities
        .target(
            name: "SwiftPixelUtils",
            dependencies: [],
            path: "Sources/SwiftPixelUtils",
            resources: [
                // Label databases for ML model inference
                // Contains class labels for ImageNet, CIFAR-100, Places365, ADE20K
                .process("Resources")
            ],
        ),
        
        // Test suite for SwiftPixelUtils
        .testTarget(
            name: "SwiftPixelUtilsTests",
            dependencies: ["SwiftPixelUtils"],
            path: "Tests/SwiftPixelUtilsTests"
        ),
    ]
)
