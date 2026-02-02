# Object Detection: YOLO and Beyond

A comprehensive reference for object detection concepts, YOLO architecture, output processing, non-maximum suppression, and implementation patterns using SwiftPixelUtils.

## Table of Contents

- [Introduction](#introduction)
- [What is Object Detection?](#what-is-object-detection)
  - [Task Definition](#task-definition)
  - [Detection vs Classification](#detection-vs-classification)
  - [Applications](#applications)
- [Detection Pipeline Overview](#detection-pipeline-overview)
  - [Two-Stage Detectors](#two-stage-detectors)
  - [One-Stage Detectors](#one-stage-detectors)
  - [Anchor-Based vs Anchor-Free](#anchor-based-vs-anchor-free)
- [YOLO Architecture Deep Dive](#yolo-architecture-deep-dive)
  - [YOLO Philosophy](#yolo-philosophy)
  - [Evolution of YOLO](#evolution-of-yolo)
  - [YOLOv8 Architecture](#yolov8-architecture)
  - [Detection Heads](#detection-heads)
- [Understanding YOLO Output](#understanding-yolo-output)
  - [Raw Output Format](#raw-output-format)
  - [Bounding Box Formats](#bounding-box-formats)
  - [Objectness Score](#objectness-score)
  - [Class Probabilities](#class-probabilities)
  - [Output Shape Variations](#output-shape-variations)
- [Coordinate Systems](#coordinate-systems)
  - [Normalized Coordinates](#normalized-coordinates)
  - [Pixel Coordinates](#pixel-coordinates)
  - [Grid-Relative Coordinates](#grid-relative-coordinates)
  - [Coordinate Conversions](#coordinate-conversions)
- [Non-Maximum Suppression (NMS)](#non-maximum-suppression-nms)
  - [Why NMS is Needed](#why-nms-is-needed)
  - [Standard NMS Algorithm](#standard-nms-algorithm)
  - [IoU Calculation](#iou-calculation)
  - [Soft-NMS](#soft-nms)
  - [Class-Agnostic vs Class-Specific NMS](#class-agnostic-vs-class-specific-nms)
  - [NMS Variants](#nms-variants)
- [Confidence Thresholds](#confidence-thresholds)
  - [Objectness Threshold](#objectness-threshold)
  - [Class Confidence Threshold](#class-confidence-threshold)
  - [Combined Confidence](#combined-confidence)
  - [Choosing Thresholds](#choosing-thresholds)
- [Multi-Scale Detection](#multi-scale-detection)
  - [Feature Pyramid Networks](#feature-pyramid-networks)
  - [PANet and BiFPN](#panet-and-bifpn)
  - [Detecting Objects of Different Sizes](#detecting-objects-of-different-sizes)
- [Anchor Boxes](#anchor-boxes)
  - [What Are Anchors?](#what-are-anchors)
  - [Anchor Generation](#anchor-generation)
  - [K-Means for Anchor Clustering](#k-means-for-anchor-clustering)
  - [Anchor-Free Approaches](#anchor-free-approaches)
- [Popular Detection Models](#popular-detection-models)
  - [YOLO Family](#yolo-family)
  - [SSD](#ssd)
  - [EfficientDet](#efficientdet)
  - [DETR (Transformer-based)](#detr-transformer-based)
  - [Model Comparison](#model-comparison)
- [Label Databases for Detection](#label-databases-for-detection)
  - [COCO (80 Classes)](#coco-80-classes)
  - [Pascal VOC (20 Classes)](#pascal-voc-20-classes)
  - [Open Images](#open-images)
  - [Custom Datasets](#custom-datasets)
- [SwiftPixelUtils Detection API](#swiftpixelutils-detection-api)
  - [Basic Usage](#basic-usage)
  - [Configuration Options](#configuration-options)
  - [Detection Result Structure](#detection-result-structure)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Post-Processing Techniques](#post-processing-techniques)
- [Performance Optimization](#performance-optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [Mathematical Foundations](#mathematical-foundations)

---

## Introduction

Object detection extends image classification by not only identifying what objects are in an image, but also where they are located. This guide provides comprehensive coverage of detection theory, YOLO architectures, and practical implementation using SwiftPixelUtils.

---

## What is Object Detection?

### Task Definition

Given an input image, identify and localize all objects of interest:

```
Input Image
┌─────────────────────────────────────┐
│                                     │
│    ┌───────┐      ┌─────────────┐   │
│    │ 🐕    │      │   🚗        │   │
│    │  Dog  │      │    Car      │   │
│    └───────┘      └─────────────┘   │
│                                     │
│         ┌──────────┐                │
│         │ 🧑 Person│                │
│         └──────────┘                │
│                                     │
└─────────────────────────────────────┘

Output:
[
  { class: "dog",    box: [50, 100, 150, 250],  confidence: 0.94 },
  { class: "car",    box: [200, 80, 400, 200],  confidence: 0.89 },
  { class: "person", box: [120, 200, 220, 450], confidence: 0.97 }
]
```

**Formal Definition:**
- Input: Image $x \in \mathbb{R}^{H \times W \times 3}$
- Output: Set of detections $\{(c_i, b_i, s_i)\}_{i=1}^{N}$
  - $c_i$: Class label
  - $b_i = (x, y, w, h)$: Bounding box
  - $s_i$: Confidence score

### Detection vs Classification

| Aspect | Classification | Detection |
|--------|---------------|-----------|
| Output | Single label | Multiple objects with boxes |
| Location | No | Yes (bounding boxes) |
| Count | Assumes one object | Multiple objects |
| Architecture | Backbone + FC | Backbone + Detection head |
| Loss | Cross-entropy | Box regression + Classification |
| Complexity | Lower | Higher |

### Applications

| Domain | Use Case | Typical Classes |
|--------|----------|-----------------|
| **Autonomous Driving** | Obstacle detection | car, pedestrian, cyclist, sign |
| **Surveillance** | Security monitoring | person, vehicle, suspicious item |
| **Retail** | Inventory, checkout | products, price tags |
| **Medical** | Lesion detection | tumor, nodule, abnormality |
| **Agriculture** | Crop monitoring | fruit, pest, disease |
| **Wildlife** | Animal tracking | species identification |
| **Manufacturing** | Quality control | defects, components |
| **Robotics** | Object manipulation | items to pick/place |
| **Augmented Reality** | Scene understanding | surfaces, objects |
| **Sports** | Player/ball tracking | players, ball, equipment |

---

## Detection Pipeline Overview

### Two-Stage Detectors

**Examples:** R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN

```
┌────────────────────────────────────────────────────┐
│                  Stage 1: RPN                      │
│  Generate ~2000 region proposals                   │
│  (where objects might be)                          │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│               Stage 2: Classification              │
│  For each proposal:                                │
│    - Classify object vs background                 │
│    - Refine bounding box                           │
└────────────────────────────────────────────────────┘
```

**Pros:**
- High accuracy
- Good for small objects
- Well-suited for precise localization

**Cons:**
- Slower (two passes)
- Complex architecture
- Not real-time friendly

### One-Stage Detectors

**Examples:** YOLO, SSD, RetinaNet, CenterNet

```
┌────────────────────────────────────────────────────┐
│              Single Forward Pass                   │
│                                                    │
│  Image → Backbone → Detection Head → Predictions   │
│                                                    │
│  Directly predicts:                                │
│    - Bounding boxes                                │
│    - Class probabilities                           │
│    - Confidence scores                             │
└────────────────────────────────────────────────────┘
```

**Pros:**
- Fast (real-time capable)
- Simpler architecture
- End-to-end training

**Cons:**
- May struggle with small objects
- Dense prediction can be redundant

### Anchor-Based vs Anchor-Free

**Anchor-Based (YOLO v1-v7, SSD, RetinaNet):**
- Predefined boxes at various scales/ratios
- Predict offsets from anchors
- More established, well-understood

**Anchor-Free (CenterNet, FCOS, YOLO v8+):**
- Predict box centers directly
- No predefined anchors
- Simpler, fewer hyperparameters

---

## YOLO Architecture Deep Dive

### YOLO Philosophy

**"You Only Look Once"** - process the entire image in a single forward pass.

```
Traditional Approach:
Image → [Propose regions] → [Classify each region] → Results
        ~2000 proposals     ~2000 classifications
        
YOLO Approach:
Image → [Single network pass] → Results
        All predictions at once
```

**Key insight:** Detection as regression problem, not classification of proposals.

### Evolution of YOLO

| Version | Year | Key Innovation | Speed/Accuracy |
|---------|------|----------------|----------------|
| YOLOv1 | 2016 | Single-shot detection | Fast, lower accuracy |
| YOLOv2 | 2016 | Batch norm, anchor boxes | Better accuracy |
| YOLOv3 | 2018 | Multi-scale detection | Good balance |
| YOLOv4 | 2020 | CSP, PANet, Mish | State-of-art at time |
| YOLOv5 | 2020 | PyTorch, export tools | Practical/easy to use |
| YOLOv6 | 2022 | Efficient reparameterization | Industrial deployment |
| YOLOv7 | 2022 | E-ELAN, model scaling | Best at publication |
| YOLOv8 | 2023 | Anchor-free, decoupled head | Current best practice |
| YOLOv9 | 2024 | Programmable Gradient Info | Improved accuracy |
| YOLOv10 | 2024 | NMS-free training | End-to-end efficient |
| YOLO11 | 2024 | Ultralytics latest | Efficiency improvements |

### YOLOv8 Architecture

```
Input: 640×640×3
         │
         ▼
┌──────────────────────────────────┐
│           BACKBONE               │
│   CSPDarknet (or other)          │
│                                  │
│   Conv → C2f → Conv → C2f...     │
│                                  │
│   Extracts multi-scale features  │
└──────────────────────────────────┘
         │
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
   P3   P4   P5   P6   (Feature pyramid levels)
   80   40   20   10   (Spatial resolution at 640 input)
         │
         ▼
┌──────────────────────────────────┐
│           NECK (PANet)           │
│                                  │
│   Top-down + Bottom-up fusion    │
│   Combines multi-scale features  │
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│       DETECTION HEAD             │
│   (Anchor-free, Decoupled)       │
│                                  │
│   ┌──────────┬──────────┐        │
│   │ Box Head │ Cls Head │        │
│   │ (x,y,w,h)│ (80 cls) │        │
│   └──────────┴──────────┘        │
└──────────────────────────────────┘
         │
         ▼
   Raw Predictions
   (8400 boxes × 84 values for COCO)
```

### Detection Heads

**Coupled Head (older YOLOs):**
```
Features → Conv → [box, objectness, classes] together
```

**Decoupled Head (YOLOv8):**
```
Features ─┬→ Box branch → [x, y, w, h]
          │
          └→ Cls branch → [class probabilities]
          
Benefits:
- Separate optimization for localization vs classification
- Better convergence
- Improved accuracy
```

---

## Understanding YOLO Output

### Raw Output Format

YOLOv8 output shape: `[batch, num_predictions, 4 + num_classes]`

For COCO (80 classes):
```
Output shape: [1, 8400, 84]
                 │     │
                 │     └─ 4 box coords + 80 class scores
                 └─ Number of predictions (varies by input size)

8400 predictions come from:
- 80×80 grid = 6400 (small objects)
- 40×40 grid = 1600 (medium objects)  
- 20×20 grid = 400  (large objects)
- Total: 8400
```

**Per-prediction format:**
```
Index:  [0, 1, 2, 3, 4, 5, 6, ..., 83]
        [x, y, w, h, cls0, cls1, cls2, ..., cls79]
        └─ box ─┘   └────── class scores ───────┘
```

### Bounding Box Formats

**YOLO native (center format):**
```
(cx, cy, w, h)
cx, cy: Center coordinates
w, h: Width and height

┌───────────────┐
│               │
│    (cx,cy)    │
│       ●───w───│
│       │       │
│       h       │
│               │
└───────────────┘
```

**Corner format (x1, y1, x2, y2):**
```
(x1, y1): Top-left corner
(x2, y2): Bottom-right corner

(x1,y1)
    ●───────────┐
    │           │
    │           │
    │           │
    └───────────●
            (x2,y2)
```

**Normalized vs Pixel coordinates:**
```
Normalized (0-1):           Pixel (image size):
x=0.5, y=0.5, w=0.2, h=0.3  x=320, y=320, w=128, h=192
(center of 640×640 image)   (actual pixel values)
```

### Objectness Score

Some YOLO versions include objectness separate from class scores:

```
YOLOv3/v4/v5 output per anchor:
[tx, ty, tw, th, objectness, cls0, cls1, ..., clsN]
                     │
                     └─ P(object exists in this box)

Final confidence = objectness × class_probability
```

**YOLOv8 (anchor-free):**
- No explicit objectness
- Class scores directly represent confidence
- Confidence = max(class_scores)

### Class Probabilities

**Conditional probability:**
$$P(\text{class}_i | \text{object}) = \text{class score}_i$$

**Final detection confidence:**
$$\text{confidence} = P(\text{object}) \times P(\text{class}_i | \text{object})$$

For YOLOv8:
$$\text{confidence} = \text{class score}_i$$

### Output Shape Variations

| Model | Typical Shape | Notes |
|-------|---------------|-------|
| YOLOv8n | [1, 84, 8400] | Transposed in some exports |
| YOLOv8s-l | [1, 84, 8400] | Same structure |
| YOLOv5 | [1, 25200, 85] | 85 = 4 + 1 + 80 (with objectness) |
| YOLOv3 | [1, 10647, 85] | Three scales |

**Transposition issues:**
```swift
// Some exports: [1, 84, 8400] - features × predictions
// Some exports: [1, 8400, 84] - predictions × features
// Check and transpose if needed!
```

---

## Coordinate Systems

### Normalized Coordinates

Values in range [0, 1], relative to image dimensions:

```
┌─────────────────────────┐
│(0,0)             (1,0)  │
│                         │
│     ┌───────┐           │
│     │ box   │           │
│     │(0.3,0.4)          │  cx=0.3, cy=0.4
│     │ w=0.2 │           │  w=0.2, h=0.15
│     │ h=0.15│           │
│     └───────┘           │
│                         │
│(0,1)             (1,1)  │
└─────────────────────────┘
```

**Advantages:**
- Resolution independent
- Easy to scale to any size
- Standard for YOLO

### Pixel Coordinates

Absolute pixel positions:

```
For 640×480 image:
normalized: cx=0.3, cy=0.4, w=0.2, h=0.15
pixel:      cx=192, cy=192, w=128, h=72
```

### Grid-Relative Coordinates

Used internally in some YOLO versions:

```
Grid cell (3, 5) out of 13×13 grid
tx, ty: Offset within cell (0-1)
Absolute position: x = (3 + sigmoid(tx)) / 13
```

### Coordinate Conversions

```swift
// Center to corner format
func centerToCorner(cx: Float, cy: Float, w: Float, h: Float) 
    -> (x1: Float, y1: Float, x2: Float, y2: Float) {
    let x1 = cx - w / 2
    let y1 = cy - h / 2
    let x2 = cx + w / 2
    let y2 = cy + h / 2
    return (x1, y1, x2, y2)
}

// Corner to center format
func cornerToCenter(x1: Float, y1: Float, x2: Float, y2: Float)
    -> (cx: Float, cy: Float, w: Float, h: Float) {
    let cx = (x1 + x2) / 2
    let cy = (y1 + y2) / 2
    let w = x2 - x1
    let h = y2 - y1
    return (cx, cy, w, h)
}

// Normalized to pixel
func normalizedToPixel(box: (Float, Float, Float, Float), 
                       imageWidth: Int, imageHeight: Int)
    -> (Float, Float, Float, Float) {
    return (
        box.0 * Float(imageWidth),
        box.1 * Float(imageHeight),
        box.2 * Float(imageWidth),
        box.3 * Float(imageHeight)
    )
}
```

---

## Non-Maximum Suppression (NMS)

### Why NMS is Needed

YOLO predicts thousands of boxes, many overlapping:

```
Without NMS:                With NMS:
┌───────────────────┐       ┌───────────────────┐
│ ┌─────┐           │       │ ┌─────┐           │
│ │┌────┴┐          │       │ │ Dog │           │
│ ││Dog  │ ← Multiple│  →   │ │ 94% │           │
│ │└────┬┘   boxes  │       │ └─────┘           │
│ └─────┘    for    │       │                   │
│            same   │       │     Single box    │
│            object │       │                   │
└───────────────────┘       └───────────────────┘
```

### Standard NMS Algorithm

```
Algorithm: Non-Maximum Suppression
─────────────────────────────────────────────────
Input:
  B = list of boxes with scores
  threshold = IoU threshold (e.g., 0.5)

Output:
  D = list of final detections

1. Sort B by confidence score (descending)
2. D = empty list
3. While B is not empty:
   a. Take box with highest score → b_max
   b. Add b_max to D
   c. Remove b_max from B
   d. For each remaining box b in B:
      - If IoU(b_max, b) > threshold:
        - Remove b from B  (suppressed)
4. Return D
```

**Visual example:**
```
Step 1: Sorted boxes
Box A: 95% ─────────────►  Selected (highest)
Box B: 92% (IoU with A = 0.8) → Suppressed
Box C: 88% (IoU with A = 0.7) → Suppressed  
Box D: 75% (IoU with A = 0.1) → Keep
Box E: 60% (IoU with D = 0.6) → Suppressed

Result: Box A, Box D
```

### IoU Calculation

**Intersection over Union:**

$$IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

```
┌───────────────┐
│     Box A     │
│    ┌──────────┼──────┐
│    │Intersect │      │
│    │   ████   │      │
└────┼──────────┘      │
     │      Box B      │
     └─────────────────┘

IoU = Area(████) / (Area(A) + Area(B) - Area(████))
```

```swift
func calculateIoU(box1: BoundingBox, box2: BoundingBox) -> Float {
    // Calculate intersection
    let x1 = max(box1.x1, box2.x1)
    let y1 = max(box1.y1, box2.y1)
    let x2 = min(box1.x2, box2.x2)
    let y2 = min(box1.y2, box2.y2)
    
    // Check for no intersection
    if x2 < x1 || y2 < y1 {
        return 0
    }
    
    let intersectionArea = (x2 - x1) * (y2 - y1)
    let box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    let box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    let unionArea = box1Area + box2Area - intersectionArea
    
    return intersectionArea / unionArea
}
```

### Soft-NMS

Instead of hard suppression, decay scores:

$$s_i = \begin{cases} s_i & \text{if } IoU < threshold \\ s_i \cdot e^{-\frac{IoU^2}{\sigma}} & \text{otherwise} \end{cases}$$

```
Standard NMS:  Score 92% → 0% (removed)
Soft-NMS:      Score 92% → 45% (decayed based on IoU)

Advantage: Handles overlapping objects better
           (e.g., crowd of people)
```

### Class-Agnostic vs Class-Specific NMS

**Class-specific (per-class):**
```swift
// NMS separately for each class
for classId in 0..<numClasses {
    let classBoxes = boxes.filter { $0.classId == classId }
    let nmsResult = nms(classBoxes, threshold: 0.5)
    results.append(contentsOf: nmsResult)
}

// Allows: Two overlapping boxes if different classes
// Dog box and Cat box can overlap
```

**Class-agnostic (global):**
```swift
// Single NMS for all boxes regardless of class
let nmsResult = nms(allBoxes, threshold: 0.5)

// Prevents: Any overlapping boxes
// Only one box per region
```

### NMS Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| Standard NMS | Hard threshold removal | General detection |
| Soft-NMS | Score decay instead of removal | Crowded scenes |
| Batched NMS | Class-specific NMS | Multi-class detection |
| DIoU-NMS | Uses distance in suppression | Better for overlapping |
| Matrix NMS | Parallel GPU computation | Fast inference |
| Weighted NMS | Merge overlapping boxes | Ensemble models |

---

## Confidence Thresholds

### Objectness Threshold

Filter boxes with low object probability:

```swift
// YOLOv5-style: separate objectness
let objectnessThreshold: Float = 0.5

for detection in rawDetections {
    if detection.objectness > objectnessThreshold {
        // Keep for further processing
    }
}
```

### Class Confidence Threshold

Filter by class score:

```swift
let classThreshold: Float = 0.25

let maxClassScore = detection.classScores.max()!
let predictedClass = detection.classScores.argmax()!

if maxClassScore > classThreshold {
    // Accept detection
}
```

### Combined Confidence

```swift
// YOLOv5/older: combined score
let finalConfidence = objectness * maxClassScore

// YOLOv8: class score is the confidence
let finalConfidence = maxClassScore
```

### Choosing Thresholds

| Scenario | Confidence | NMS IoU | Notes |
|----------|------------|---------|-------|
| High precision | 0.5-0.7 | 0.3-0.4 | Fewer false positives |
| High recall | 0.1-0.25 | 0.5-0.6 | Catch all objects |
| Balanced | 0.25-0.4 | 0.45-0.5 | Default recommendation |
| Dense scenes | 0.2-0.3 | 0.5-0.7 | Allow more overlap |
| Sparse scenes | 0.4-0.5 | 0.3-0.4 | Stricter filtering |

```swift
// Threshold tuning guidelines
let config = DetectionConfig(
    confidenceThreshold: 0.25,  // Start here
    nmsThreshold: 0.45,         // COCO default
    maxDetections: 100          // Limit output count
)

// For safety-critical applications (e.g., autonomous driving):
let safeConfig = DetectionConfig(
    confidenceThreshold: 0.1,   // Lower = more detections
    nmsThreshold: 0.5,          // Higher = more overlap allowed
    maxDetections: 300
)
```

---

## Multi-Scale Detection

### Feature Pyramid Networks

Detect objects at multiple scales:

```
Backbone output:
                 ┌────┐
P5 (20×20)  ────►│    │──► Small receptive field (large objects)
                 │    │
                 │ F  │
P4 (40×40)  ────►│ P  │──► Medium receptive field
                 │ N  │
                 │    │
P3 (80×80)  ────►│    │──► Large receptive field (small objects)
                 └────┘

Each level detects objects of appropriate size:
- P3: Small objects (e.g., distant cars)
- P4: Medium objects (e.g., people)
- P5: Large objects (e.g., trucks, close objects)
```

### PANet and BiFPN

**PANet (Path Aggregation Network):**
```
Top-down path (FPN):
P5 → P4 → P3

Bottom-up path (PANet addition):
P3 → P4 → P5

Information flows both directions for better fusion
```

**BiFPN (Bidirectional FPN):**
- Weighted feature fusion
- Repeated bidirectional flow
- Used in EfficientDet

### Detecting Objects of Different Sizes

```
Input: 640×640

Grid     Stride   Typical Object Size
─────────────────────────────────────
80×80    8px      8-64 pixels (small)
40×40    16px     32-128 pixels (medium)
20×20    32px     64-512 pixels (large)

Assignment:
- Object size / stride ≈ 3-5 for good detection
- Small object (24px) → 80×80 grid (stride 8): 24/8 = 3 ✓
- Large object (256px) → 20×20 grid (stride 32): 256/32 = 8 ✓
```

---

## Anchor Boxes

### What Are Anchors?

Predefined boxes that serve as reference for predictions:

```
Without anchors:
Network must predict absolute coordinates from scratch

With anchors:
Network predicts offsets from anchor boxes

┌─────────────────────────────────┐
│         Image Grid              │
│  ┌───┬───┬───┬───┬───┐          │
│  │ ▢ │ ▢ │ ▢ │ ▢ │ ▢ │  ▢ = anchors │
│  ├───┼───┼───┼───┼───┤          │
│  │ ▢ │ ▢ │ ▢ │ ▢ │ ▢ │          │
│  └───┴───┴───┴───┴───┘          │
│                                 │
│  Each cell has multiple anchors │
│  of different aspect ratios     │
└─────────────────────────────────┘
```

### Anchor Generation

Common anchor configurations:

```python
# YOLOv5 default anchors (width, height in pixels)
anchors = [
    [(10,13), (16,30), (33,23)],      # P3/8 (small)
    [(30,61), (62,45), (59,119)],     # P4/16 (medium)
    [(116,90), (156,198), (373,326)]  # P5/32 (large)
]

# Aspect ratios commonly used:
# 1:1 (square)
# 1:2, 2:1 (vertical/horizontal)
# 1:3, 3:1 (extreme ratios for specific objects)
```

### K-Means for Anchor Clustering

Find optimal anchors for your dataset:

```python
# Pseudo-code for anchor clustering
def cluster_anchors(boxes, k=9):
    """
    boxes: All ground truth boxes [N, 2] (width, height)
    k: Number of anchors to generate
    """
    # Use k-means with IoU distance
    centroids = kmeans(boxes, k, distance=iou_distance)
    return sorted(centroids, key=area)
```

### Anchor-Free Approaches

YOLOv8+ uses anchor-free detection:

```
Anchor-based prediction:
tx, ty = offset from anchor center
tw, th = scale relative to anchor

Anchor-free prediction:
x, y = direct center coordinates
w, h = direct width/height (or learned from distribution)

Advantages:
- No hyperparameters (anchor sizes)
- Better generalization
- Simpler training
```

---

## Popular Detection Models

### YOLO Family

| Model | mAP (COCO) | Speed (ms) | Best For |
|-------|------------|------------|----------|
| YOLOv8n | 37.3 | 1.2 | Edge devices |
| YOLOv8s | 44.9 | 2.1 | Mobile |
| YOLOv8m | 50.2 | 4.9 | Balanced |
| YOLOv8l | 52.9 | 7.8 | Accuracy |
| YOLOv8x | 53.9 | 13.0 | Best accuracy |

### SSD

**Single Shot MultiBox Detector:**
- Multi-scale feature maps
- Fixed anchor boxes
- Faster than two-stage, less accurate than YOLO

### EfficientDet

- EfficientNet backbone
- BiFPN neck
- Compound scaling (like EfficientNet)
- Good accuracy/efficiency tradeoff

| Model | mAP | Parameters |
|-------|-----|------------|
| D0 | 33.8 | 3.9M |
| D1 | 39.6 | 6.6M |
| D2 | 43.0 | 8.1M |
| D4 | 49.4 | 21M |
| D7 | 52.2 | 52M |

### DETR (Transformer-based)

**Detection Transformer:**
- No anchors, no NMS
- Set prediction with Hungarian matching
- End-to-end training
- Slower but elegant

### Model Comparison

| Model | mAP | FPS (V100) | Parameters | Use Case |
|-------|-----|------------|------------|----------|
| YOLOv8n | 37.3 | 850 | 3.2M | Real-time edge |
| YOLOv8s | 44.9 | 470 | 11.2M | Real-time mobile |
| YOLOv8m | 50.2 | 200 | 25.9M | Balanced |
| EfficientDet-D0 | 33.8 | 100 | 3.9M | Edge |
| Faster R-CNN | 42.0 | 25 | 41M | High accuracy |
| DETR | 42.0 | 28 | 41M | Research |

---

## Label Databases for Detection

### COCO (80 Classes)

**Common Objects in Context** - Most widely used detection dataset.

```swift
let cocoLabels = [
    // People and accessories (0-10)
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    
    // Animals (14-23)
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    
    // Accessories (24-28)
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    
    // Sports (29-38)
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
    
    // Kitchen (39-45)
    "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl",
    
    // Food (46-55)
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    
    // Furniture (56-62)
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv",
    
    // Electronics (63-67)
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    
    // Appliances (68-72)
    "microwave", "oven", "toaster", "sink", "refrigerator",
    
    // Other (73-79)
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
// Total: 80 classes
```

### Pascal VOC (20 Classes)

Older, smaller dataset:

```swift
let vocLabels = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
```

### Open Images

- 600 object classes
- 1.7M training images
- Hierarchical labels

### Custom Datasets

```swift
// Define custom labels
let customLabels = ["product_a", "product_b", "defect", "label"]

// Use with SwiftPixelUtils
let result = try DetectionOutput.process(
    output: modelOutput,
    labels: .custom(customLabels),
    config: config
)
```

---

## SwiftPixelUtils Detection API

### Basic Usage

```swift
import SwiftPixelUtils

// Process YOLO output
let result = try DetectionOutput.processYOLO(
    outputData: modelOutput,
    inputWidth: 640,
    inputHeight: 640,
    originalWidth: image.size.width,
    originalHeight: image.size.height,
    labels: .coco,
    config: .default
)

// Access detections
for detection in result.detections {
    print("Class: \(detection.label)")
    print("Confidence: \(detection.confidence)")
    print("Box: \(detection.boundingBox)")
}
```

### Configuration Options

```swift
let config = DetectionConfig(
    // Filtering
    confidenceThreshold: 0.25,
    nmsThreshold: 0.45,
    maxDetections: 100,
    
    // NMS behavior
    nmsType: .perClass,  // or .global
    
    // Box format
    inputBoxFormat: .centerWidthHeight,  // cx, cy, w, h
    outputBoxFormat: .corners,           // x1, y1, x2, y2
    
    // Coordinate system
    normalizeOutput: false,  // Output in pixel coords
    
    // Model specifics
    hasObjectness: false,  // YOLOv8 style
    transposed: true       // [1, 84, 8400] format
)
```

### Detection Result Structure

```swift
struct DetectionResult {
    // All detections after NMS
    let detections: [Detection]
    
    // Metadata
    let numDetections: Int
    let processingTimeMs: Double
    
    // Raw data (if needed)
    let rawPredictions: Int  // Before NMS
}

struct Detection {
    let classIndex: Int
    let label: String
    let confidence: Float
    let boundingBox: BoundingBox
}

struct BoundingBox {
    let x1, y1, x2, y2: Float  // Corners in pixels
    
    var center: (x: Float, y: Float)
    var size: (width: Float, height: Float)
    var area: Float
}
```

---

## Complete Implementation Examples

### Example 1: YOLOv8 with TFLite

```swift
import SwiftPixelUtils
import TensorFlowLite

class YOLOv8Detector {
    private let interpreter: Interpreter
    private let inputWidth = 640
    private let inputHeight = 640
    
    init(modelPath: String) throws {
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
    }
    
    func detect(image: UIImage) async throws -> DetectionResult {
        // 1. Preprocess
        let input = try await PixelExtractor.getModelInput(
            source: .uiImage(image),
            framework: .tfliteFloat,
            width: inputWidth,
            height: inputHeight,
            resizeMode: .letterbox(padding: .gray)
        )
        
        // 2. Run inference
        try interpreter.copy(input.data, toInputAt: 0)
        try interpreter.invoke()
        
        // 3. Get output
        let outputTensor = try interpreter.output(at: 0)
        
        // 4. Process detections
        return try DetectionOutput.processYOLO(
            outputData: outputTensor.data,
            inputWidth: inputWidth,
            inputHeight: inputHeight,
            originalWidth: Int(image.size.width),
            originalHeight: Int(image.size.height),
            labels: .coco,
            config: DetectionConfig(
                confidenceThreshold: 0.25,
                nmsThreshold: 0.45,
                transposed: true  // YOLOv8 format
            ),
            letterboxInfo: input.letterboxInfo
        )
    }
}

// Usage
let detector = try YOLOv8Detector(modelPath: "yolov8n.tflite")
let result = try await detector.detect(image: myImage)

for detection in result.detections {
    print("\(detection.label): \(detection.confidence * 100)%")
    print("  Box: \(detection.boundingBox)")
}
```

### Example 2: Drawing Detection Results

```swift
extension UIImage {
    func drawDetections(_ detections: [Detection]) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: size)
        
        return renderer.image { context in
            // Draw original image
            draw(at: .zero)
            
            let ctx = context.cgContext
            
            for detection in detections {
                // Box color based on class (using hash for consistency)
                let hue = CGFloat(detection.classIndex % 20) / 20.0
                let color = UIColor(hue: hue, saturation: 0.8, 
                                   brightness: 0.9, alpha: 1.0)
                
                // Draw bounding box
                let box = detection.boundingBox
                let rect = CGRect(
                    x: CGFloat(box.x1),
                    y: CGFloat(box.y1),
                    width: CGFloat(box.x2 - box.x1),
                    height: CGFloat(box.y2 - box.y1)
                )
                
                ctx.setStrokeColor(color.cgColor)
                ctx.setLineWidth(3)
                ctx.stroke(rect)
                
                // Draw label background
                let label = "\(detection.label) \(Int(detection.confidence * 100))%"
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 14),
                    .foregroundColor: UIColor.white
                ]
                let labelSize = label.size(withAttributes: attrs)
                
                let labelRect = CGRect(
                    x: rect.minX,
                    y: rect.minY - labelSize.height - 4,
                    width: labelSize.width + 8,
                    height: labelSize.height + 4
                )
                
                ctx.setFillColor(color.cgColor)
                ctx.fill(labelRect)
                
                // Draw label text
                label.draw(
                    at: CGPoint(x: labelRect.minX + 4, y: labelRect.minY + 2),
                    withAttributes: attrs
                )
            }
        }
    }
}
```

---

## Post-Processing Techniques

### Letterbox Coordinate Correction

When using letterbox padding, coordinates need adjustment. SwiftPixelUtils now automatically captures letterbox transform metadata:

```swift
// Model sees letterboxed image:
// ┌─────────────────────┐
// │░░░░░░░░░░░░░░░░░░░░░│ ← padding
// │░┌─────────────────┐░│
// │░│                 │░│
// │░│  Actual image   │░│
// │░│                 │░│
// │░└─────────────────┘░│
// │░░░░░░░░░░░░░░░░░░░░░│ ← padding
// └─────────────────────┘

// Automatic letterbox info with getPixelData:
let result = try await PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: PixelDataOptions(
        resize: ResizeOptions(width: 640, height: 640, strategy: .letterbox),
        colorFormat: .rgb,
        normalization: .scale,
        dataLayout: .nchw
    )
)

// Transform info is automatically captured
if let info = result.letterboxInfo {
    // info.scale: scale factor applied
    // info.offset: padding offset (x, y)
    // info.originalSize: original image dimensions
    // info.letterboxedSize: final padded dimensions
}

// Manual correction formula (if needed):
func correctLetterbox(
    box: BoundingBox,
    letterboxInfo: LetterboxInfo,
    originalSize: CGSize
) -> BoundingBox {
    let x1 = (box.x1 - Float(letterboxInfo.offset.x)) / letterboxInfo.scale
    let y1 = (box.y1 - Float(letterboxInfo.offset.y)) / letterboxInfo.scale
    let x2 = (box.x2 - Float(letterboxInfo.offset.x)) / letterboxInfo.scale
    let y2 = (box.y2 - Float(letterboxInfo.offset.y)) / letterboxInfo.scale
    
    // Clip to image bounds
    return BoundingBox(
        x1: max(0, min(x1, Float(originalSize.width))),
        y1: max(0, min(y1, Float(originalSize.height))),
        x2: max(0, min(x2, Float(originalSize.width))),
        y2: max(0, min(y2, Float(originalSize.height)))
    )
}
```

### Tracking Across Frames

```swift
class SimpleTracker {
    private var previousDetections: [Detection] = []
    private var trackIds: [Int] = []
    private var nextId = 0
    
    func update(detections: [Detection]) -> [(Detection, trackId: Int)] {
        var results: [(Detection, Int)] = []
        var usedPrevious = Set<Int>()
        
        for detection in detections {
            // Find best matching previous detection
            var bestMatch: Int? = nil
            var bestIoU: Float = 0.3  // Minimum IoU threshold
            
            for (i, prev) in previousDetections.enumerated() {
                if usedPrevious.contains(i) { continue }
                if prev.classIndex != detection.classIndex { continue }
                
                let iou = calculateIoU(detection.boundingBox, prev.boundingBox)
                if iou > bestIoU {
                    bestIoU = iou
                    bestMatch = i
                }
            }
            
            let trackId: Int
            if let match = bestMatch {
                trackId = trackIds[match]
                usedPrevious.insert(match)
            } else {
                trackId = nextId
                nextId += 1
            }
            
            results.append((detection, trackId))
        }
        
        previousDetections = detections
        trackIds = results.map { $0.1 }
        
        return results
    }
}
```

---

## Performance Optimization

### Efficient NMS Implementation

```swift
// Use Accelerate for faster IoU computation
import Accelerate

func batchedIoU(boxes1: [[Float]], boxes2: [[Float]]) -> [[Float]] {
    // Vectorized IoU calculation using Accelerate
    // Much faster than nested loops for many boxes
    // ...
}
```

### Model Optimization

| Technique | Speedup | Accuracy Impact |
|-----------|---------|-----------------|
| INT8 quantization | 2-4× | -0.5 to -2% mAP |
| FP16 inference | 1.5-2× | Minimal |
| Input size reduction | 2-4× | -2 to -5% mAP |
| TensorRT/CoreML | 2-3× | None |
| Pruning | 1.5-2× | -1 to -3% mAP |

---

## Evaluation Metrics

### Mean Average Precision (mAP)

$$mAP = \frac{1}{|C|} \sum_{c \in C} AP_c$$

Where $AP_c$ is Average Precision for class $c$.

### IoU Thresholds

| Metric | IoU Threshold | Description |
|--------|---------------|-------------|
| mAP@0.5 | 0.5 | Loose matching |
| mAP@0.75 | 0.75 | Strict matching |
| mAP@[.5:.95] | 0.5 to 0.95 | COCO primary metric |

---

## Troubleshooting

### Problem: No Detections

**Checklist:**
1. ✓ Confidence threshold too high?
2. ✓ Correct input preprocessing?
3. ✓ Output shape/format correct?
4. ✓ Labels match model?
5. ✓ Model loaded correctly?

### Problem: Wrong Coordinates

**Common causes:**
- Letterbox correction not applied
- Wrong coordinate format (center vs corner)
- Normalized vs pixel confusion
- Transposed output tensor

### Problem: Duplicate Detections

**Solutions:**
- Lower NMS IoU threshold
- Check if NMS is being applied
- Verify class-specific vs global NMS

---

## Mathematical Foundations

### Box Regression

YOLO predicts offsets:
$$b_x = \sigma(t_x) + c_x$$
$$b_y = \sigma(t_y) + c_y$$
$$b_w = p_w \cdot e^{t_w}$$
$$b_h = p_h \cdot e^{t_h}$$

Where $(c_x, c_y)$ is grid cell offset and $(p_w, p_h)$ is anchor size.

### Loss Function

Combined detection loss:
$$L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{cls} L_{cls}$$

- $L_{box}$: CIoU/DIoU loss for localization
- $L_{obj}$: Binary cross-entropy for objectness
- $L_{cls}$: Cross-entropy for classification

