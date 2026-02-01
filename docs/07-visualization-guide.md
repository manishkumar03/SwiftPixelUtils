# Visualization Guide: Drawing ML Results

A comprehensive reference for visualizing machine learning outputs including bounding boxes, segmentation masks, classification results, heatmaps, and debugging visualizations using SwiftPixelUtils.

## Table of Contents

- [Introduction](#introduction)
- [Visualization Fundamentals](#visualization-fundamentals)
  - [Color Theory for ML Visualization](#color-theory-for-ml-visualization)
  - [Perceptual Considerations](#perceptual-considerations)
  - [Accessibility Guidelines](#accessibility-guidelines)
- [Drawing Bounding Boxes](#drawing-bounding-boxes)
  - [Box Formats and Coordinates](#box-formats-and-coordinates)
  - [Style Options](#style-options)
  - [Labels and Confidence](#labels-and-confidence)
  - [Advanced Box Rendering](#advanced-box-rendering)
- [Segmentation Masks](#segmentation-masks)
  - [Color Mapping](#color-mapping)
  - [Overlay Techniques](#overlay-techniques)
  - [Boundary Visualization](#boundary-visualization)
  - [Interactive Masks](#interactive-masks)
- [Classification Results](#classification-results)
  - [Top-K Display](#top-k-display)
  - [Confidence Bars](#confidence-bars)
  - [Confusion Indicators](#confusion-indicators)
- [Attention and Heatmaps](#attention-and-heatmaps)
  - [Grad-CAM Visualization](#grad-cam-visualization)
  - [Attention Maps](#attention-maps)
  - [Color Scales for Heatmaps](#color-scales-for-heatmaps)
- [Debugging Visualizations](#debugging-visualizations)
  - [Preprocessing Verification](#preprocessing-verification)
  - [Model Input Inspection](#model-input-inspection)
  - [Feature Map Visualization](#feature-map-visualization)
- [Animation and Video](#animation-and-video)
  - [Detection Tracking](#detection-tracking)
  - [Temporal Consistency](#temporal-consistency)
  - [Smooth Transitions](#smooth-transitions)
- [SwiftPixelUtils Visualization API](#swiftpixelutils-visualization-api)
  - [Basic Drawing](#basic-drawing)
  - [Style Configuration](#style-configuration)
  - [Composite Visualizations](#composite-visualizations)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Platform-Specific Rendering](#platform-specific-rendering)
  - [UIKit (iOS)](#uikit-ios)
  - [AppKit (macOS)](#appkit-macos)
  - [SwiftUI](#swiftui)
  - [Core Graphics](#core-graphics)
  - [Metal](#metal)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

---

## Introduction

Effective visualization is crucial for understanding, debugging, and presenting machine learning results. This guide covers techniques for rendering detection boxes, segmentation masks, classification results, and diagnostic visualizations using SwiftPixelUtils.

---

## Visualization Fundamentals

### Color Theory for ML Visualization

**Color spaces for visualization:**

```
RGB: Good for screens
┌─────────────────────────────────────┐
│ Red (255,0,0) + Green (0,255,0)     │
│        = Yellow (255,255,0)         │
│ Additive mixing                     │
└─────────────────────────────────────┘

HSV/HSL: Better for generating palettes
┌─────────────────────────────────────┐
│ H (Hue): Color wheel position 0-360 │
│ S (Saturation): Color intensity     │
│ V/L (Value/Lightness): Brightness   │
│                                     │
│ Easy to create distinct colors by   │
│ evenly spacing H values             │
└─────────────────────────────────────┘
```

**Generating distinct colors:**
```swift
func generateDistinctColors(count: Int) -> [UIColor] {
    return (0..<count).map { i in
        let hue = CGFloat(i) / CGFloat(count)
        return UIColor(hue: hue, saturation: 0.8, brightness: 0.9, alpha: 1.0)
    }
}

// For 10 classes, colors are spaced 36° apart on color wheel
// Result: Red, Orange, Yellow, Green, Cyan, Blue, Purple, Pink, etc.
```

**Color palette strategies:**

| Strategy | Use Case | Example |
|----------|----------|---------|
| Categorical | Distinct classes | COCO, VOC palettes |
| Sequential | Ordered data (low→high) | Heatmaps, confidence |
| Diverging | Two extremes | Error maps (+/-) |
| Perceptually uniform | Accurate comparison | Viridis, Plasma |

### Perceptual Considerations

**Luminance contrast:**
```
Dark background + bright colors: ✓ Good visibility
Light background + pastel colors: ✗ Hard to see

┌────────────────────┐  ┌────────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  │ ░░░░░░░░░░░░░░░░░░ │
│ ▓  ████████████  ▓ │  │ ░  ▒▒▒▒▒▒▒▒▒▒▒▒  ░ │
│ ▓  █ Visible █   ▓ │  │ ░  ▒ Faint  ▒    ░ │
│ ▓  ████████████  ▓ │  │ ░  ▒▒▒▒▒▒▒▒▒▒▒▒  ░ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  │ ░░░░░░░░░░░░░░░░░░ │
└────────────────────┘  └────────────────────┘
    Good contrast          Poor contrast
```

**Size and thickness:**
```swift
// Adaptive line width based on image size
func adaptiveLineWidth(imageSize: CGSize) -> CGFloat {
    let shortEdge = min(imageSize.width, imageSize.height)
    return max(1.0, shortEdge / 200.0)  // 1px min, scales with image
}

// Example:
// 640×480 → 2.4px line width
// 1920×1080 → 5.4px line width
```

### Accessibility Guidelines

**Color blindness considerations:**

```
Don't rely on color alone!

❌ Red/Green for good/bad
   (8% of men are red-green colorblind)

✓ Use patterns, shapes, or labels in addition to color
✓ Use colorblind-friendly palettes
```

**Colorblind-safe palettes:**
```swift
// Wong's colorblind-friendly palette
let colorblindSafe: [UIColor] = [
    UIColor(red: 0.00, green: 0.45, blue: 0.70, alpha: 1),  // Blue
    UIColor(red: 0.90, green: 0.62, blue: 0.00, alpha: 1),  // Orange
    UIColor(red: 0.00, green: 0.62, blue: 0.45, alpha: 1),  // Teal
    UIColor(red: 0.80, green: 0.47, blue: 0.65, alpha: 1),  // Pink
    UIColor(red: 0.94, green: 0.89, blue: 0.26, alpha: 1),  // Yellow
    UIColor(red: 0.34, green: 0.71, blue: 0.91, alpha: 1),  // Sky Blue
    UIColor(red: 0.84, green: 0.37, blue: 0.00, alpha: 1),  // Vermillion
]
```

---

## Drawing Bounding Boxes

### Box Formats and Coordinates

**Common formats:**
```
(x1, y1, x2, y2) - Corners       (cx, cy, w, h) - Center
┌─────────────────────┐          ┌─────────────────────┐
│(x1,y1)              │          │                     │
│ ●───────────────┐   │          │     (cx,cy)         │
│ │               │   │          │       ●─────w/2─────│
│ │    Object     │   │          │       │             │
│ │               │   │          │       h/2           │
│ └───────────────● (x2,y2)      │                     │
│                     │          │                     │
└─────────────────────┘          └─────────────────────┘

Normalized (0-1)                 Pixel coordinates
x, y ∈ [0, 1]                    x, y ∈ [0, width/height]
```

**Coordinate conversion:**
```swift
struct BoundingBox {
    var x1, y1, x2, y2: Float
    
    // Convert to center format
    var center: (cx: Float, cy: Float, w: Float, h: Float) {
        return (
            cx: (x1 + x2) / 2,
            cy: (y1 + y2) / 2,
            w: x2 - x1,
            h: y2 - y1
        )
    }
    
    // Convert to CGRect for drawing
    func toCGRect(imageSize: CGSize) -> CGRect {
        return CGRect(
            x: CGFloat(x1),
            y: CGFloat(y1),
            width: CGFloat(x2 - x1),
            height: CGFloat(y2 - y1)
        )
    }
    
    // Scale from normalized to pixel
    func scaled(to size: CGSize) -> BoundingBox {
        return BoundingBox(
            x1: x1 * Float(size.width),
            y1: y1 * Float(size.height),
            x2: x2 * Float(size.width),
            y2: y2 * Float(size.height)
        )
    }
}
```

### Style Options

**Box styles:**
```
Solid border:           Dashed border:          Filled (transparent):
┌───────────────┐       ┌─ ─ ─ ─ ─ ─ ─┐         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
│               │       │               │         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
│               │       │               │         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
└───────────────┘       └ ─ ─ ─ ─ ─ ─ ─┘         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Corner markers:         Rounded corners:        With shadow:
●─────────────●         ╭───────────────╮        ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
│               │       │               │        ▒┌────────────┐▒
│               │       │               │        ▒│            │▒
●─────────────●         ╰───────────────╯        ▒└────────────┘▒
                                                 ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
```

```swift
struct BoxStyle {
    var strokeColor: UIColor = .red
    var strokeWidth: CGFloat = 2.0
    var fillColor: UIColor? = nil
    var cornerRadius: CGFloat = 0
    var dashPattern: [CGFloat]? = nil
    var shadowRadius: CGFloat = 0
    var shadowOffset: CGSize = .zero
}

// Preset styles
extension BoxStyle {
    static let solid = BoxStyle(strokeWidth: 3)
    static let dashed = BoxStyle(dashPattern: [5, 3])
    static let filled = BoxStyle(fillColor: UIColor.red.withAlphaComponent(0.3))
    static let rounded = BoxStyle(cornerRadius: 5)
    
    static func yoloStyle(for classIndex: Int) -> BoxStyle {
        let hue = CGFloat(classIndex % 20) / 20.0
        let color = UIColor(hue: hue, saturation: 0.8, brightness: 0.9, alpha: 1)
        return BoxStyle(strokeColor: color, strokeWidth: 3)
    }
}
```

### Labels and Confidence

**Label placement options:**
```
Above box:              Inside top:             Below box:
┌────────────────┐
│ Person 95%     │
├────────────────┤      ┌────────────────┐      ┌────────────────┐
│                │      │ Person 95%     │      │                │
│    Object      │      │                │      │    Object      │
│                │      │    Object      │      │                │
└────────────────┘      │                │      ├────────────────┤
                        └────────────────┘      │ Person 95%     │
                                                └────────────────┘

Beside box:             Tooltip style:
┌────────────────┐ Person 95%    ┌────────────────┐
│                │               │         ●──────┤ Person 95%
│    Object      │               │    Object      │
│                │               │                │
└────────────────┘               └────────────────┘
```

**Label drawing:**
```swift
func drawLabel(
    text: String,
    at position: CGPoint,
    style: LabelStyle,
    context: CGContext
) {
    let attributes: [NSAttributedString.Key: Any] = [
        .font: UIFont.boldSystemFont(ofSize: style.fontSize),
        .foregroundColor: style.textColor
    ]
    
    let size = (text as NSString).size(withAttributes: attributes)
    
    // Draw background
    let padding: CGFloat = 4
    let backgroundRect = CGRect(
        x: position.x - padding,
        y: position.y - size.height - padding,
        width: size.width + padding * 2,
        height: size.height + padding * 2
    )
    
    context.setFillColor(style.backgroundColor.cgColor)
    context.fill(backgroundRect)
    
    // Draw text
    (text as NSString).draw(
        at: CGPoint(x: position.x, y: position.y - size.height),
        withAttributes: attributes
    )
}

// Format confidence
func formatConfidence(_ confidence: Float) -> String {
    return String(format: "%.0f%%", confidence * 100)
}
```

### Advanced Box Rendering

**Multi-class detection display:**
```swift
func drawDetections(
    _ detections: [Detection],
    on image: UIImage,
    colorMap: [Int: UIColor]
) -> UIImage {
    let renderer = UIGraphicsImageRenderer(size: image.size)
    
    return renderer.image { context in
        image.draw(at: .zero)
        let cgContext = context.cgContext
        
        // Sort by area (draw larger boxes first, smaller on top)
        let sorted = detections.sorted { $0.area > $1.area }
        
        for detection in sorted {
            let box = detection.boundingBox.toCGRect(imageSize: image.size)
            let color = colorMap[detection.classIndex] ?? .red
            
            // Draw box
            cgContext.setStrokeColor(color.cgColor)
            cgContext.setLineWidth(3)
            cgContext.stroke(box)
            
            // Draw label
            let label = "\(detection.label) \(formatConfidence(detection.confidence))"
            drawLabel(
                text: label,
                at: CGPoint(x: box.minX, y: box.minY),
                style: LabelStyle(backgroundColor: color),
                context: cgContext
            )
        }
    }
}
```

**Confidence-based opacity:**
```swift
// More confident = more opaque
func opacityForConfidence(_ confidence: Float) -> CGFloat {
    // Map 0.25-1.0 confidence to 0.5-1.0 opacity
    return CGFloat(0.5 + (confidence - 0.25) * 0.67)
}

// High confidence: solid box
// Low confidence: semi-transparent
```

---

## Segmentation Masks

### Color Mapping

**Standard palette (Pascal VOC):**
```swift
let vocPalette: [(UInt8, UInt8, UInt8)] = [
    (0, 0, 0),       // 0: background
    (128, 0, 0),     // 1: aeroplane
    (0, 128, 0),     // 2: bicycle
    (128, 128, 0),   // 3: bird
    (0, 0, 128),     // 4: boat
    (128, 0, 128),   // 5: bottle
    (0, 128, 128),   // 6: bus
    (128, 128, 128), // 7: car
    (64, 0, 0),      // 8: cat
    (192, 0, 0),     // 9: chair
    (64, 128, 0),    // 10: cow
    (192, 128, 0),   // 11: diningtable
    (64, 0, 128),    // 12: dog
    (192, 0, 128),   // 13: horse
    (64, 128, 128),  // 14: motorbike
    (192, 128, 128), // 15: person
    (0, 64, 0),      // 16: pottedplant
    (128, 64, 0),    // 17: sheep
    (0, 192, 0),     // 18: sofa
    (128, 192, 0),   // 19: train
    (0, 64, 128)     // 20: tvmonitor
]
```

**Create colored mask:**
```swift
func colorMask(
    _ mask: [UInt8],
    width: Int,
    height: Int,
    palette: [(UInt8, UInt8, UInt8)]
) -> [UInt8] {
    var colored = [UInt8](repeating: 0, count: width * height * 4)
    
    for i in 0..<(width * height) {
        let classIndex = Int(mask[i])
        let (r, g, b) = palette[classIndex]
        
        colored[i * 4 + 0] = r
        colored[i * 4 + 1] = g
        colored[i * 4 + 2] = b
        colored[i * 4 + 3] = 255  // Alpha
    }
    
    return colored
}
```

### Overlay Techniques

**Alpha blending:**
```
Overlay formula:
output = image × (1 - α) + mask × α

α = 0.0: Original image only
α = 0.5: 50% blend
α = 1.0: Mask only

┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│    Original    │ + │   Mask (α=0.5) │ = │    Overlay     │
│     Image      │   │                │   │                │
│    ░░░░░░░     │   │    ▓▓▓▓▓▓▓     │   │    ▒▒▒▒▒▒▒     │
└────────────────┘   └────────────────┘   └────────────────┘
```

```swift
func overlayMask(
    image: UIImage,
    mask: [UInt8],
    palette: [(UInt8, UInt8, UInt8)],
    alpha: Float = 0.5
) -> UIImage {
    guard let cgImage = image.cgImage else { return image }
    
    let width = cgImage.width
    let height = cgImage.height
    
    // Get image pixels
    var imagePixels = getPixels(from: cgImage)
    
    // Blend with mask
    for i in 0..<(width * height) {
        let classIndex = Int(mask[i])
        
        // Skip background (optional)
        if classIndex == 0 { continue }
        
        let (r, g, b) = palette[classIndex]
        let idx = i * 4
        
        imagePixels[idx + 0] = UInt8(Float(imagePixels[idx + 0]) * (1 - alpha) + Float(r) * alpha)
        imagePixels[idx + 1] = UInt8(Float(imagePixels[idx + 1]) * (1 - alpha) + Float(g) * alpha)
        imagePixels[idx + 2] = UInt8(Float(imagePixels[idx + 2]) * (1 - alpha) + Float(b) * alpha)
    }
    
    return createImage(from: imagePixels, width: width, height: height)
}
```

**Mask-only regions:**
```swift
// Show mask only where class is detected, original elsewhere
func selectiveMask(
    image: UIImage,
    mask: [UInt8],
    targetClasses: Set<Int>,
    palette: [(UInt8, UInt8, UInt8)]
) -> UIImage {
    // Only colorize pixels belonging to target classes
    // Keep original image for other pixels
}
```

### Boundary Visualization

**Extract and draw boundaries:**
```swift
func drawBoundaries(
    mask: [UInt8],
    width: Int,
    height: Int,
    lineWidth: Int = 2
) -> [UInt8] {
    var output = [UInt8](repeating: 0, count: width * height * 4)
    
    for y in 1..<(height - 1) {
        for x in 1..<(width - 1) {
            let idx = y * width + x
            let currentClass = mask[idx]
            
            // Check 4-neighbors for class boundary
            let isBoundary = 
                mask[idx - 1] != currentClass ||      // left
                mask[idx + 1] != currentClass ||      // right
                mask[idx - width] != currentClass ||  // top
                mask[idx + width] != currentClass     // bottom
            
            if isBoundary && currentClass != 0 {
                // Draw boundary pixel (white)
                output[idx * 4 + 0] = 255
                output[idx * 4 + 1] = 255
                output[idx * 4 + 2] = 255
                output[idx * 4 + 3] = 255
            }
        }
    }
    
    return output
}
```

```
Full mask:                  Boundary only:
┌────────────────────┐     ┌────────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓       │     │ ┌──────────┐       │
│ ▓▓▓▓▓▓▓▓▓▓▓▓       │     │ │          │       │
│ ▓▓▓▓▓▓▓▓▓▓▓▓       │  →  │ │          │       │
│ ▓▓▓▓▓▓▓▓▓▓▓▓       │     │ │          │       │
│ ▓▓▓▓▓▓▓▓▓▓▓▓       │     │ └──────────┘       │
└────────────────────┘     └────────────────────┘
```

### Interactive Masks

**Hover/touch highlighting:**
```swift
class InteractiveSegmentationView: UIView {
    var mask: [UInt8]?
    var highlightedClass: Int?
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first,
              let mask = mask else { return }
        
        let point = touch.location(in: self)
        let x = Int(point.x * CGFloat(maskWidth) / bounds.width)
        let y = Int(point.y * CGFloat(maskHeight) / bounds.height)
        
        let classIndex = Int(mask[y * maskWidth + x])
        highlightedClass = classIndex
        
        updateVisualization()
    }
    
    func updateVisualization() {
        // Highlight all pixels of the touched class
        // Dim other classes
    }
}
```

---

## Classification Results

### Top-K Display

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Classification Results                            │
│                                                     │
│   1. Golden Retriever  ████████████████████  94.2%  │
│   2. Labrador          ███                    3.1%  │
│   3. Cocker Spaniel    █                      1.2%  │
│   4. Irish Setter      █                      0.8%  │
│   5. Brittany          █                      0.4%  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

```swift
struct ClassificationResultView: View {
    let predictions: [Prediction]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(predictions.prefix(5)) { prediction in
                HStack {
                    Text("\(prediction.rank). \(prediction.label)")
                        .frame(width: 150, alignment: .leading)
                    
                    GeometryReader { geometry in
                        Rectangle()
                            .fill(Color.blue)
                            .frame(width: geometry.size.width * CGFloat(prediction.confidence))
                    }
                    .frame(height: 20)
                    
                    Text(String(format: "%.1f%%", prediction.confidence * 100))
                        .frame(width: 50, alignment: .trailing)
                }
            }
        }
    }
}
```

### Confidence Bars

**Horizontal bar chart:**
```swift
func drawConfidenceBars(
    predictions: [Prediction],
    in rect: CGRect,
    context: CGContext
) {
    let barHeight: CGFloat = 25
    let spacing: CGFloat = 5
    let maxBarWidth = rect.width * 0.6
    
    for (i, prediction) in predictions.prefix(5).enumerated() {
        let y = rect.minY + CGFloat(i) * (barHeight + spacing)
        
        // Background bar
        let bgRect = CGRect(x: rect.minX + 150, y: y, 
                           width: maxBarWidth, height: barHeight)
        context.setFillColor(UIColor.lightGray.cgColor)
        context.fill(bgRect)
        
        // Confidence bar
        let barWidth = maxBarWidth * CGFloat(prediction.confidence)
        let barRect = CGRect(x: rect.minX + 150, y: y,
                            width: barWidth, height: barHeight)
        
        // Color based on confidence
        let color = confidenceColor(prediction.confidence)
        context.setFillColor(color.cgColor)
        context.fill(barRect)
        
        // Label
        let label = prediction.label
        drawText(label, at: CGPoint(x: rect.minX, y: y + 5), in: context)
        
        // Percentage
        let pct = String(format: "%.1f%%", prediction.confidence * 100)
        drawText(pct, at: CGPoint(x: rect.minX + 160 + maxBarWidth, y: y + 5), in: context)
    }
}

func confidenceColor(_ confidence: Float) -> UIColor {
    if confidence > 0.8 {
        return .systemGreen
    } else if confidence > 0.5 {
        return .systemYellow
    } else {
        return .systemOrange
    }
}
```

### Confusion Indicators

**Visual uncertainty indicators:**
```swift
// Show when model is uncertain
func uncertaintyIndicator(for result: ClassificationResult) -> some View {
    let top = result.predictions[0]
    let second = result.predictions[1]
    let gap = top.confidence - second.confidence
    
    if gap < 0.1 {
        // Very uncertain - close predictions
        return AnyView(
            HStack {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundColor(.orange)
                Text("Uncertain between \(top.label) and \(second.label)")
            }
        )
    } else if top.confidence < 0.5 {
        // Low confidence overall
        return AnyView(
            HStack {
                Image(systemName: "questionmark.circle")
                    .foregroundColor(.yellow)
                Text("Low confidence: \(top.label)")
            }
        )
    } else {
        // Confident
        return AnyView(
            HStack {
                Image(systemName: "checkmark.circle")
                    .foregroundColor(.green)
                Text(top.label)
            }
        )
    }
}
```

---

## Attention and Heatmaps

### Grad-CAM Visualization

**Gradient-weighted Class Activation Mapping:**

```
Grad-CAM shows where the model "looks" for its prediction

Original        Grad-CAM Heatmap    Overlay
┌──────────┐    ┌──────────┐       ┌──────────┐
│    🐕    │    │   ░▒▓█▓  │       │    🐕    │
│    dog   │ →  │  ░▒▓██▓░ │   →   │  ◐◑◒◓◔   │
│    here  │    │   ░▒▓▒░  │       │  heated  │
└──────────┘    └──────────┘       └──────────┘
                Red = high attention
```

```swift
func visualizeGradCAM(
    heatmap: [Float],  // Normalized 0-1
    width: Int,
    height: Int,
    colormap: Colormap = .jet
) -> [UInt8] {
    var output = [UInt8](repeating: 0, count: width * height * 4)
    
    for i in 0..<(width * height) {
        let value = heatmap[i]
        let (r, g, b) = colormap.getColor(value)
        
        output[i * 4 + 0] = r
        output[i * 4 + 1] = g
        output[i * 4 + 2] = b
        output[i * 4 + 3] = 255
    }
    
    return output
}

func overlayGradCAM(
    image: UIImage,
    heatmap: [Float],
    alpha: Float = 0.5
) -> UIImage {
    // 1. Upsample heatmap to image size (bilinear)
    // 2. Apply colormap
    // 3. Alpha blend with original image
}
```

### Attention Maps

**Transformer attention visualization:**

```
Multi-head attention patterns:

Query patches attend to key patches:
┌───┬───┬───┬───┐
│ . │ . │ █ │ . │   █ = high attention
├───┼───┼───┼───┤   . = low attention
│ . │ █ │ █ │ . │
├───┼───┼───┼───┤   Query at center
│ █ │ █ │ Q │ █ │   attends to nearby
├───┼───┼───┼───┤   and semantically
│ . │ █ │ █ │ . │   related regions
└───┴───┴───┴───┘
```

```swift
func visualizeAttention(
    attentionWeights: [[Float]],  // [numHeads, numPatches, numPatches]
    patchSize: Int,
    queryPatch: Int,
    head: Int = 0
) -> [Float] {
    // Get attention from query to all keys
    let attention = attentionWeights[head][queryPatch]
    
    // Reshape to spatial grid
    let gridSize = Int(sqrt(Double(attention.count)))
    
    // Upsample to pixel resolution
    return upsample(attention, from: gridSize, to: gridSize * patchSize)
}
```

### Color Scales for Heatmaps

**Popular colormaps:**
```
Jet:        🔵 → 🟢 → 🟡 → 🔴  (Classic, not perceptually uniform)
Viridis:    🟣 → 🔵 → 🟢 → 🟡  (Perceptually uniform, colorblind safe)
Plasma:     🔵 → 🟣 → 🔴 → 🟡  (Perceptually uniform)
Hot:        ⚫ → 🔴 → 🟡 → ⚪  (Black to white through red)
Grayscale:  ⚫ →→→→→→→→→→→ ⚪  (Simple, accessible)
```

```swift
enum Colormap {
    case jet
    case viridis
    case plasma
    case hot
    case grayscale
    
    func getColor(_ value: Float) -> (UInt8, UInt8, UInt8) {
        let v = max(0, min(1, value))  // Clamp to [0, 1]
        
        switch self {
        case .jet:
            return jetColor(v)
        case .viridis:
            return viridisColor(v)
        case .grayscale:
            let gray = UInt8(v * 255)
            return (gray, gray, gray)
        // ... other colormaps
        }
    }
}

func jetColor(_ v: Float) -> (UInt8, UInt8, UInt8) {
    var r: Float = 0, g: Float = 0, b: Float = 0
    
    if v < 0.25 {
        r = 0
        g = 4 * v
        b = 1
    } else if v < 0.5 {
        r = 0
        g = 1
        b = 1 - 4 * (v - 0.25)
    } else if v < 0.75 {
        r = 4 * (v - 0.5)
        g = 1
        b = 0
    } else {
        r = 1
        g = 1 - 4 * (v - 0.75)
        b = 0
    }
    
    return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
}
```

---

## Debugging Visualizations

### Preprocessing Verification

**Visualize preprocessing steps:**

```swift
func visualizePreprocessing(
    original: UIImage,
    steps: [PreprocessingStep]
) -> UIImage {
    let renderer = UIGraphicsImageRenderer(
        size: CGSize(width: 800, height: 200 * steps.count)
    )
    
    return renderer.image { context in
        var y: CGFloat = 0
        var currentImage = original
        
        for step in steps {
            // Draw step name
            step.name.draw(at: CGPoint(x: 10, y: y + 10))
            
            // Draw before
            currentImage.draw(in: CGRect(x: 10, y: y + 30, 
                                        width: 150, height: 150))
            
            // Apply step
            currentImage = step.apply(to: currentImage)
            
            // Draw after
            currentImage.draw(in: CGRect(x: 180, y: y + 30,
                                        width: 150, height: 150))
            
            // Draw arrow
            drawArrow(from: CGPoint(x: 160, y: y + 100),
                     to: CGPoint(x: 180, y: y + 100),
                     in: context.cgContext)
            
            y += 200
        }
    }
}
```

```
Preprocessing visualization:

Step 1: Resize
┌─────────┐    →    ┌─────────┐
│ 1920×   │         │  224×   │
│ 1080    │         │  224    │
└─────────┘         └─────────┘

Step 2: Normalize
┌─────────┐    →    ┌─────────┐
│ [0,255] │         │ [-1,1]  │
│         │         │         │
└─────────┘         └─────────┘
```

### Model Input Inspection

```swift
func visualizeModelInput(
    data: Data,
    shape: [Int],  // e.g., [1, 224, 224, 3]
    format: DataFormat
) -> UIImage {
    // Convert model input back to viewable image
    let floats = data.withUnsafeBytes {
        Array(UnsafeBufferPointer<Float>(
            start: $0.baseAddress?.assumingMemoryBound(to: Float.self),
            count: data.count / 4
        ))
    }
    
    // Denormalize
    let denormalized: [UInt8]
    switch format {
    case .zeroToOne:
        denormalized = floats.map { UInt8($0 * 255) }
    case .negOneToOne:
        denormalized = floats.map { UInt8(($0 + 1) / 2 * 255) }
    case .imagenetNormalized:
        // Reverse ImageNet normalization
        denormalized = reverseImageNetNorm(floats)
    }
    
    return createImage(from: denormalized, 
                       width: shape[2], 
                       height: shape[1])
}
```

### Feature Map Visualization

**Visualize intermediate layer outputs:**

```swift
func visualizeFeatureMaps(
    features: [Float],  // [C, H, W] or [H, W, C]
    numChannels: Int,
    width: Int,
    height: Int,
    gridCols: Int = 8
) -> UIImage {
    let numRows = (numChannels + gridCols - 1) / gridCols
    let cellWidth = 64
    let cellHeight = 64
    
    let renderer = UIGraphicsImageRenderer(
        size: CGSize(width: gridCols * cellWidth,
                    height: numRows * cellHeight)
    )
    
    return renderer.image { context in
        for c in 0..<numChannels {
            let row = c / gridCols
            let col = c % gridCols
            
            // Extract channel
            let channelData = extractChannel(features, channel: c,
                                            width: width, height: height)
            
            // Normalize to 0-255
            let normalized = normalizeForDisplay(channelData)
            
            // Create small image
            let channelImage = createGrayscaleImage(normalized,
                                                   width: width, 
                                                   height: height)
            
            // Draw in grid
            channelImage.draw(in: CGRect(
                x: col * cellWidth,
                y: row * cellHeight,
                width: cellWidth,
                height: cellHeight
            ))
        }
    }
}
```

```
Feature map grid (64 channels):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │ 06 │ 07 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 08 │ 09 │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ ...                                   │
└────┴────┴────┴────┴────┴────┴────┴────┘

Each cell shows what that filter responds to
```

---

## Animation and Video

### Detection Tracking

```swift
class DetectionTracker {
    private var previousDetections: [TrackedDetection] = []
    private var nextId = 0
    
    func update(detections: [Detection]) -> [TrackedDetection] {
        var tracked: [TrackedDetection] = []
        var usedPrevious = Set<Int>()
        
        for detection in detections {
            // Find matching previous detection
            let match = findBestMatch(detection, in: previousDetections,
                                     excluding: usedPrevious)
            
            if let match = match {
                // Continue existing track
                tracked.append(TrackedDetection(
                    id: match.id,
                    detection: detection,
                    age: match.age + 1
                ))
                usedPrevious.insert(match.index)
            } else {
                // New track
                tracked.append(TrackedDetection(
                    id: nextId,
                    detection: detection,
                    age: 0
                ))
                nextId += 1
            }
        }
        
        previousDetections = tracked
        return tracked
    }
}

// Draw with consistent colors per track
func drawTrackedDetections(_ tracked: [TrackedDetection], on image: UIImage) -> UIImage {
    // Use track ID for consistent color
    for t in tracked {
        let color = colorForTrackId(t.id)
        drawBox(t.detection.boundingBox, color: color)
        drawLabel("ID: \(t.id) \(t.detection.label)", color: color)
    }
}
```

### Temporal Consistency

**Smooth box transitions:**
```swift
class SmoothBoxRenderer {
    private var previousBoxes: [Int: CGRect] = [:]
    private let smoothingFactor: CGFloat = 0.3
    
    func smoothBox(current: CGRect, trackId: Int) -> CGRect {
        guard let previous = previousBoxes[trackId] else {
            previousBoxes[trackId] = current
            return current
        }
        
        // Exponential moving average
        let smoothed = CGRect(
            x: previous.minX + (current.minX - previous.minX) * smoothingFactor,
            y: previous.minY + (current.minY - previous.minY) * smoothingFactor,
            width: previous.width + (current.width - previous.width) * smoothingFactor,
            height: previous.height + (current.height - previous.height) * smoothingFactor
        )
        
        previousBoxes[trackId] = smoothed
        return smoothed
    }
}
```

### Smooth Transitions

**Fade in/out for appearing/disappearing detections:**
```swift
class FadeTransitionRenderer {
    private var trackStates: [Int: TrackState] = [:]
    
    struct TrackState {
        var alpha: CGFloat
        var framesVisible: Int
        var framesInvisible: Int
    }
    
    func updateAndDraw(detections: [TrackedDetection]) -> UIImage {
        let currentIds = Set(detections.map { $0.id })
        
        // Update states
        for id in trackStates.keys {
            if currentIds.contains(id) {
                // Visible - fade in
                trackStates[id]?.alpha = min(1.0, (trackStates[id]?.alpha ?? 0) + 0.2)
                trackStates[id]?.framesVisible += 1
                trackStates[id]?.framesInvisible = 0
            } else {
                // Invisible - fade out
                trackStates[id]?.alpha = max(0.0, (trackStates[id]?.alpha ?? 1) - 0.1)
                trackStates[id]?.framesInvisible += 1
            }
        }
        
        // Add new tracks
        for detection in detections {
            if trackStates[detection.id] == nil {
                trackStates[detection.id] = TrackState(alpha: 0.2, 
                                                       framesVisible: 1,
                                                       framesInvisible: 0)
            }
        }
        
        // Draw with alpha
        for detection in detections {
            let alpha = trackStates[detection.id]?.alpha ?? 1.0
            drawDetection(detection, alpha: alpha)
        }
        
        // Cleanup old tracks
        trackStates = trackStates.filter { $0.value.framesInvisible < 30 }
    }
}
```

---

## SwiftPixelUtils Visualization API

### Basic Drawing

```swift
import SwiftPixelUtils

// Draw detection boxes
let visualized = try Visualizer.drawDetections(
    on: image,
    detections: result.detections,
    style: .yolo
)

// Draw segmentation mask
let overlay = try Visualizer.drawSegmentation(
    on: image,
    mask: result.classMask,
    palette: .pascalVOC,
    alpha: 0.5
)

// Draw classification results
let labeled = try Visualizer.drawClassification(
    on: image,
    predictions: result.predictions,
    topK: 5
)
```

### Style Configuration

```swift
let customStyle = VisualizationStyle(
    // Boxes
    boxStrokeWidth: 3.0,
    boxCornerRadius: 5.0,
    
    // Labels
    labelFont: .systemFont(ofSize: 14, weight: .bold),
    labelBackground: true,
    labelPadding: 4,
    
    // Colors
    colorPalette: .yolo,
    confidenceColorCoding: true,
    
    // Segmentation
    maskAlpha: 0.5,
    showBoundaries: true,
    
    // General
    darkMode: false
)

let visualized = try Visualizer.draw(
    on: image,
    results: results,
    style: customStyle
)
```

### Composite Visualizations

```swift
// Multiple result types
let composite = try Visualizer.drawComposite(
    on: image,
    detections: detectionResult.detections,
    segmentation: segmentationResult.classMask,
    classification: classificationResult.predictions,
    layout: .sideBySide  // or .overlay, .grid
)
```

---

## Complete Implementation Examples

### Example 1: Full Detection Visualization

```swift
func visualizeDetectionResults(
    image: UIImage,
    result: DetectionResult
) -> UIImage {
    let renderer = UIGraphicsImageRenderer(size: image.size)
    
    return renderer.image { context in
        // Draw original image
        image.draw(at: .zero)
        
        let cgContext = context.cgContext
        
        // Sort detections (larger boxes first)
        let sorted = result.detections.sorted { $0.area > $1.area }
        
        for detection in sorted {
            let box = detection.boundingBox.toCGRect(imageSize: image.size)
            let color = colorForClass(detection.classIndex)
            
            // Semi-transparent fill
            cgContext.setFillColor(color.withAlphaComponent(0.2).cgColor)
            cgContext.fill(box)
            
            // Solid border
            cgContext.setStrokeColor(color.cgColor)
            cgContext.setLineWidth(max(2, image.size.width / 200))
            cgContext.stroke(box)
            
            // Label with background
            let label = "\(detection.label) \(Int(detection.confidence * 100))%"
            drawLabel(label, at: box.origin, color: color, in: cgContext)
        }
    }
}

func colorForClass(_ classIndex: Int) -> UIColor {
    let hue = CGFloat(classIndex * 37 % 360) / 360.0  // Golden angle distribution
    return UIColor(hue: hue, saturation: 0.8, brightness: 0.9, alpha: 1.0)
}
```

### Example 2: Segmentation with Legend

```swift
func visualizeSegmentationWithLegend(
    image: UIImage,
    result: SegmentationResult
) -> UIImage {
    let legendWidth: CGFloat = 150
    let totalWidth = image.size.width + legendWidth
    
    let renderer = UIGraphicsImageRenderer(
        size: CGSize(width: totalWidth, height: image.size.height)
    )
    
    return renderer.image { context in
        // Draw segmented image
        let segmentedImage = overlayMask(image: image,
                                         mask: result.classMask,
                                         palette: vocPalette)
        segmentedImage.draw(at: .zero)
        
        // Draw legend
        let cgContext = context.cgContext
        let x = image.size.width + 10
        var y: CGFloat = 20
        
        for stat in result.classStatistics where stat.pixelCount > 0 {
            let (r, g, b) = vocPalette[stat.classIndex]
            let color = UIColor(red: CGFloat(r)/255, 
                               green: CGFloat(g)/255,
                               blue: CGFloat(b)/255, alpha: 1)
            
            // Color swatch
            cgContext.setFillColor(color.cgColor)
            cgContext.fill(CGRect(x: x, y: y, width: 20, height: 20))
            
            // Label
            let label = "\(stat.label) (\(String(format: "%.1f", stat.percentage))%)"
            (label as NSString).draw(
                at: CGPoint(x: x + 25, y: y + 2),
                withAttributes: [.font: UIFont.systemFont(ofSize: 12)]
            )
            
            y += 25
        }
    }
}
```

---

## Platform-Specific Rendering

### UIKit (iOS)

```swift
class DetectionOverlayView: UIView {
    var detections: [Detection] = [] {
        didSet { setNeedsDisplay() }
    }
    
    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        for detection in detections {
            let box = detection.boundingBox.toCGRect(imageSize: bounds.size)
            
            context.setStrokeColor(UIColor.red.cgColor)
            context.setLineWidth(2)
            context.stroke(box)
        }
    }
}
```

### AppKit (macOS)

```swift
class DetectionOverlayView: NSView {
    var detections: [Detection] = [] {
        didSet { needsDisplay = true }
    }
    
    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        
        // Note: macOS coordinates are flipped (origin at bottom-left)
        context.saveGState()
        context.translateBy(x: 0, y: bounds.height)
        context.scaleBy(x: 1, y: -1)
        
        for detection in detections {
            // Draw...
        }
        
        context.restoreGState()
    }
}
```

### SwiftUI

```swift
struct DetectionOverlay: View {
    let detections: [Detection]
    let imageSize: CGSize
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(detections, id: \.id) { detection in
                let box = scaledBox(detection.boundingBox, 
                                   from: imageSize, 
                                   to: geometry.size)
                
                Rectangle()
                    .stroke(colorForClass(detection.classIndex), lineWidth: 2)
                    .frame(width: box.width, height: box.height)
                    .position(x: box.midX, y: box.midY)
                
                Text("\(detection.label) \(Int(detection.confidence * 100))%")
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(2)
                    .background(colorForClass(detection.classIndex))
                    .position(x: box.minX + 40, y: box.minY - 10)
            }
        }
    }
}

// Usage
ZStack {
    Image(uiImage: originalImage)
        .resizable()
        .aspectRatio(contentMode: .fit)
    
    DetectionOverlay(detections: result.detections, 
                    imageSize: originalImage.size)
}
```

### Core Graphics

```swift
func renderWithCoreGraphics(image: CGImage, detections: [Detection]) -> CGImage? {
    let width = image.width
    let height = image.height
    
    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    
    // Draw image
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    
    // Draw detections
    for detection in detections {
        let box = detection.boundingBox.scaled(to: CGSize(width: width, height: height))
        
        context.setStrokeColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.setLineWidth(2)
        context.stroke(box.toCGRect())
    }
    
    return context.makeImage()
}
```

### Metal

```swift
// For high-performance real-time rendering
class MetalDetectionRenderer {
    private let device: MTLDevice
    private let pipelineState: MTLRenderPipelineState
    
    func renderDetections(
        _ detections: [Detection],
        onto texture: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        // Convert detections to vertex buffer
        let vertices = detections.flatMap { detection -> [Float] in
            let box = detection.boundingBox
            return [
                // Line strip for box
                box.x1, box.y1,
                box.x2, box.y1,
                box.x2, box.y2,
                box.x1, box.y2,
                box.x1, box.y1
            ]
        }
        
        // Create render encoder and draw
        // ... Metal rendering code
    }
}
```

---

## Performance Optimization

### Caching

```swift
class VisualizationCache {
    private var colorCache: [Int: UIColor] = [:]
    private var labelSizeCache: [String: CGSize] = [:]
    
    func colorForClass(_ classIndex: Int) -> UIColor {
        if let cached = colorCache[classIndex] {
            return cached
        }
        let color = generateColor(for: classIndex)
        colorCache[classIndex] = color
        return color
    }
    
    func sizeForLabel(_ label: String, font: UIFont) -> CGSize {
        let key = "\(label)_\(font.pointSize)"
        if let cached = labelSizeCache[key] {
            return cached
        }
        let size = (label as NSString).size(withAttributes: [.font: font])
        labelSizeCache[key] = size
        return size
    }
}
```

### Batch Rendering

```swift
// Draw all boxes with single stroke call
func drawAllBoxes(_ boxes: [CGRect], color: CGColor, context: CGContext) {
    context.setStrokeColor(color)
    context.setLineWidth(2)
    
    for box in boxes {
        context.addRect(box)
    }
    
    context.strokePath()  // Single draw call
}
```

---

## Best Practices

### 1. Consistent Color Assignment
```swift
// Use class index, not random colors
// Colors should be consistent across frames
let color = palette[detection.classIndex % palette.count]
```

### 2. Adaptive Sizing
```swift
// Scale UI elements with image size
let lineWidth = max(1, imageWidth / 300)
let fontSize = max(10, imageWidth / 50)
```

### 3. Performance vs Quality
```swift
// Real-time: Simple boxes, no labels
// Still image: Full labels, shadows, effects
```

### 4. Dark Mode Support
```swift
let textColor = UITraitCollection.current.userInterfaceStyle == .dark 
    ? UIColor.white 
    : UIColor.black
```
