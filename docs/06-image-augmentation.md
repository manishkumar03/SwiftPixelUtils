# Image Augmentation: Theory and Practice

A comprehensive reference for data augmentation techniques, their theory, implementation, and best practices for training robust ML models using SwiftPixelUtils.

## Table of Contents

- [Introduction](#introduction)
- [Why Augmentation Matters](#why-augmentation-matters)
  - [The Overfitting Problem](#the-overfitting-problem)
  - [Regularization Through Data](#regularization-through-data)
  - [Domain Shift Robustness](#domain-shift-robustness)
- [Categories of Augmentation](#categories-of-augmentation)
  - [Geometric Transformations](#geometric-transformations)
  - [Photometric Transformations](#photometric-transformations)
  - [Noise and Blur](#noise-and-blur)
  - [Occlusion Methods](#occlusion-methods)
  - [Advanced Augmentations](#advanced-augmentations)
- [Geometric Transformations Deep Dive](#geometric-transformations-deep-dive)
  - [Rotation](#rotation)
  - [Scaling and Cropping](#scaling-and-cropping)
  - [Flipping](#flipping)
  - [Translation](#translation)
  - [Shearing](#shearing)
  - [Perspective Transform](#perspective-transform)
  - [Elastic Deformation](#elastic-deformation)
- [Photometric Transformations Deep Dive](#photometric-transformations-deep-dive)
  - [Brightness Adjustment](#brightness-adjustment)
  - [Contrast Adjustment](#contrast-adjustment)
  - [Saturation Adjustment](#saturation-adjustment)
  - [Hue Shift](#hue-shift)
  - [Color Jitter](#color-jitter)
  - [Gamma Correction](#gamma-correction)
  - [Histogram Equalization](#histogram-equalization)
- [Noise and Blur Techniques](#noise-and-blur-techniques)
  - [Gaussian Noise](#gaussian-noise)
  - [Salt and Pepper Noise](#salt-and-pepper-noise)
  - [Speckle Noise](#speckle-noise)
  - [Gaussian Blur](#gaussian-blur)
  - [Motion Blur](#motion-blur)
  - [Defocus Blur](#defocus-blur)
- [Occlusion and Dropout](#occlusion-and-dropout)
  - [Random Erasing (Cutout)](#random-erasing-cutout)
  - [GridMask](#gridmask)
  - [CoarseDropout](#coarsedropout)
- [Advanced Augmentation Methods](#advanced-augmentation-methods)
  - [MixUp](#mixup)
  - [CutMix](#cutmix)
  - [Mosaic](#mosaic)
  - [AutoAugment](#autoaugment)
  - [RandAugment](#randaugment)
  - [TrivialAugment](#trivialaugment)
  - [AugMax](#augmax)
- [Task-Specific Augmentation](#task-specific-augmentation)
  - [Classification Augmentation](#classification-augmentation)
  - [Detection Augmentation](#detection-augmentation)
  - [Segmentation Augmentation](#segmentation-augmentation)
  - [Medical Imaging](#medical-imaging)
- [Invariance vs Equivariance](#invariance-vs-equivariance)
- [Augmentation Policies](#augmentation-policies)
  - [Random Application](#random-application)
  - [Sequential Pipelines](#sequential-pipelines)
  - [Composition Strategies](#composition-strategies)
- [Decision Guide: Which Augmentations to Use](#decision-guide-which-augmentations-to-use)
- [SwiftPixelUtils Augmentation API](#swiftpixelutils-augmentation-api)
  - [Basic Usage](#basic-usage)
  - [Available Augmentations](#available-augmentations)
  - [Pipeline Construction](#pipeline-construction)
  - [Real-Time Augmentation](#real-time-augmentation)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Best Practices](#best-practices)
- [Common Mistakes](#common-mistakes)
- [Performance Considerations](#performance-considerations)
- [Augmentation Theory Notes](#augmentation-theory-notes)
- [Mathematical Foundations](#mathematical-foundations)

---

## Introduction

Data augmentation is a regularization technique that artificially expands the training dataset by applying transformations to existing images. This guide covers the theory, implementation, and best practices for effective augmentation using SwiftPixelUtils.

---

## Why Augmentation Matters

### The Overfitting Problem

Without augmentation, models memorize training data instead of learning generalizable features:

```
Training without augmentation:
┌─────────────────────────────────────────────┐
│                                             │
│  Training Accuracy: 99.5%  ✓                │
│  Validation Accuracy: 72.3%  ✗              │
│                                             │
│  The model memorized specific pixel values, │
│  not robust features!                       │
│                                             │
└─────────────────────────────────────────────┘

Training with augmentation:
┌─────────────────────────────────────────────┐
│                                             │
│  Training Accuracy: 94.2%                   │
│  Validation Accuracy: 91.8%  ✓              │
│                                             │
│  Model learned features that generalize     │
│  across variations                          │
│                                             │
└─────────────────────────────────────────────┘
```

### Regularization Through Data

Augmentation acts as implicit regularization:

```
Original image:        Augmented variations:
┌────────────┐         ┌────────────┐ ┌────────────┐ ┌────────────┐
│            │         │ rotated    │ │ flipped    │ │ cropped    │
│   🐕       │    →    │   🐕      │ │   🐕       │ │   🐕       │
│            │         │            │ │            │ │            │
└────────────┘         └────────────┘ └────────────┘ └────────────┘
                       ┌────────────┐ ┌────────────┐ ┌────────────┐
                       │ brightened │ │ noisy      │ │ blurred    │
                       │   🐕       │ │   🐕       │ │   🐕       │
                       │            │ │            │ │            │
                       └────────────┘ └────────────┘ └────────────┘

1 image → many variations, same label
Model must learn "dog-ness", not specific pixels
```

**Regularization effect:**
- Increases effective training set size
- Prevents memorization
- Encourages learning invariant features
- Reduces variance of model

### Domain Shift Robustness

Augmentation helps with distribution shift:

```
Training images:               Real-world images:
┌────────────────┐             ┌────────────────┐
│ Perfect light  │             │ Various light  │
│ Centered       │      vs     │ Off-center     │
│ No noise       │             │ Camera noise   │
│ Standard angle │             │ Unusual angles │
└────────────────┘             └────────────────┘

Without augmentation: Model fails on real-world images
With augmentation: Model robust to these variations
```

---

## Categories of Augmentation

### Geometric Transformations

Modify spatial arrangement of pixels:

```
Original     Rotate      Flip        Crop        Scale
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ AB   │    │ CA   │    │   BA │    │ B    │    │ AB   │
│ CD   │ →  │ DB   │    │   DC │    │ D    │    │ CD   │
└──────┘    └──────┘    └──────┘    └──────┘    └──────┘
```

**Types:**
| Transform | Effect | Parameters |
|-----------|--------|------------|
| Rotation | Rotate by angle | angle, center |
| Flip | Mirror horizontally/vertically | axis |
| Crop | Extract region | x, y, width, height |
| Scale | Resize | factor, method |
| Translate | Shift position | dx, dy |
| Shear | Skew shape | shear_x, shear_y |
| Perspective | 3D viewpoint change | corner points |

### Photometric Transformations

Modify pixel values without changing geometry:

```
Original     Bright      Contrast    Saturate    Hue Shift
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ 🟢🔵 │    │ 🟢🔵 │   │ 🟢🔵 │   │ 🟢🔵 │    │ 🔴🟡 │
│  🔴  │ →  │  🔴  │   │  🔴  │    │  🔴  │    │  🔵  │
└──────┘    └──────┘    └──────┘    └──────┘    └──────┘
            brighter    higher      vivid       shifted
                        contrast    colors      hue
```

**Types:**
### Geometric Transformations and Affine Theory

Geometric augmentations map pixels from source $(x, y)$ to target $(x', y')$. Most can be represented as a $3 \times 3$ **Affine Matrix**:

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} =
\begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

- **Translation**: $a=1, d=1, b=c=0$.
- **Scaling**: $a=s_x, d=s_y$.
- **Rotation**: $a=\cos\theta, b=-\sin\theta, c=\sin\theta, d=\cos\theta$.
- **Shearing**: $b=\tan\phi_x$ or $c=\tan\phi_y$.

**Interpolation Artifacts**:
When rotating by $45^\circ$, old integer coordinates $(10, 10)$ might land on $(14.14, 14.14)$.
- **Nearest Neighbor**: Fast, but creates jagged "staircase" edges (aliasing). Bad for training unless labels are masks.
- **Bilinear**: Weighted average of 4 pixels. Standard.
- **Bicubic**: Weighted average of 16 pixels. sharper, but slower.

**Boundary Conditions**:
What happens when pixels move "out of frame"?
- **Zero/Constant**: Pad with black. (Easy, but creates sharp edges which valid convolutions detect).
- **Reflect/Mirror**: Mirrors the image content. (Natural, avoids edge artifacts).
- **Replicate/Clamp**: Repeats the edge pixel.

### Photometric Theory & Invariance

Unlike geometric transforms, photometric transforms change pixel *values*, not positions.

**Invariance vs Equivariance**:
- **Invariant**: The label should NOT change.
    - Example: *Color Jitter* on a Car. A red car is still a car.
- **Equivariant**: The label SHOULD change (or is invalid).
    - Example: *Vertical Flip* on a "Stop Sign". An upside-down stop sign is not a standard road sign (or implies an accident).
    - Example: *Hue Shift* on "Orange" vs "Apple". If you shift hue too much, an orange looks like a lime.

### Decision Guide: Choosing Augmentations

| Pipeline Type | Safe Augmentations | Risk / Dangers |
| :--- | :--- | :--- |
| **Natural Scenes** (ImageNet, COCO) | Horizontal Flip, Crop, Color Jitter, MixUp, CutMix. | Vertical Flip (gravity matters). Inverting colors (skies aren't green). |
| **Faces** (Identification) | Slight Rotation ($\pm 10^\circ$), Brightness. | Shearing (distorts features), Occlusion (hides identity keypoints). |
| **Medical Imaging** (X-Ray/CT) | Rotation ($90^\circ$ steps), Flip, Elastic Deform. | **Contrast/Brightness** (density is clinically significant!). Non-rigid warping (creates fake tumors). |
| **Documents / OCR** | Perspective Transform, Blur, Noise, Binarization. | Flipping (mirror text is unreadable). Heavy rotation (text direction). |
| **Satellite / Aerial** | Rotation ($0-360^\circ$), Flip (H+V). | Perspective (view is always nadir). |

### Occlusion Methods

Simulate partial visibility:

```
Original     Cutout       GridMask     Random Erase
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ 🐕   │    │ 🐕   │   │ 🐕   │    │ 🐕   │
│ dog  │ →  │ d█g  │    │ d░g  │    │ ███  │
│ here │    │ here │    │ h░r░ │    │ here │
└──────┘    └──────┘    └──────┘    └──────┘
            single      grid        random
            rectangle   pattern     rectangles
```

### Advanced Augmentations

Combine images or use learned policies:

```
MixUp:                  CutMix:
Image A + Image B       Patch from B into A
┌──────┐ ┌──────┐       ┌──────┐ ┌──────┐
│ 🐕   │+│ 🐈   │       │ 🐕   │+│ 🐈   │
│      │ │      │       │      │ │      │
└──────┘ └──────┘       └──────┘ └──────┘
     ↓                       ↓
┌──────┐                ┌──────┐
│ 👻   │  Blended       │ 🐕🐈 │  Patched
│      │  (0.6 dog,     │      │  (area-based
└──────┘   0.4 cat)     └──────┘   label mixing)
```

---

## Geometric Transformations Deep Dive

### Rotation

Rotate image around a center point:

```
Rotation by θ degrees:

┌─────────────┐         ┌─────────────┐
│     A       │         │      C      │
│             │   →     │   A     D   │
│  C     D    │         │      B      │
│     B       │         │             │
└─────────────┘         └─────────────┘

Transformation matrix:
┌                   ┐
│ cos(θ)  -sin(θ)  │
│ sin(θ)   cos(θ)  │
└                   ┘
```

**Implementation considerations:**
```swift
// Rotation creates empty corners - handle with:
enum RotationFill {
    case black          // Fill with black
    case reflect        // Mirror edges
    case wrap           // Wrap around
    case crop           // Crop to fit
}

// Recommended: Use crop for classification
// For detection: adjust bounding boxes after rotation
```

**When to use:**
- Objects can appear at any angle
- Top-down views
- Document/OCR

**When to avoid:**
- Orientation matters (e.g., "6" vs "9")
- Gravity-dependent scenes

### Scaling and Cropping

**Random Resized Crop (most common):**
```
Original 256×256          Random crop          Resize to target
┌────────────────┐       ┌────────────────┐    ┌────────────┐
│                │       │   ┌───────┐    │    │ ┌───────┐  │
│   ┌───────┐    │       │   │ 🐕    │    │    │ │  🐕   │  │
│   │  🐕   │    │   →   │   │       │    │ →  │ │       │  │
│   │       │    │       │   └───────┘    │    │ └───────┘  │
│   └───────┘    │       │                │    │            │
│                │       │ Random area    │    │ 224×224    │
└────────────────┘       └────────────────┘    └────────────┘
```

```swift
// Standard ImageNet-style augmentation
let crop = RandomResizedCrop(
    targetSize: (224, 224),
    scale: (0.08, 1.0),    // Crop 8% to 100% of area
    ratio: (0.75, 1.33)    // Aspect ratio range
)
```

**Center Crop (for validation):**
```swift
let centerCrop = CenterCrop(
    size: (224, 224)
)
```

### Flipping

**Horizontal flip (most common):**
```
Original        Flipped
┌──────┐       ┌──────┐
│ →🐕  │   →   │  🐕← │
│      │       │      │
└──────┘       └──────┘
```

**Vertical flip (less common):**
```
Original        Flipped
┌──────┐       ┌──────┐
│  🐕  │   →   │      │
│  ↓   │       │  ↑   │
└──────┘       └──────┘
```

```swift
// Horizontal flip - almost always safe
let hFlip = RandomHorizontalFlip(probability: 0.5)

// Vertical flip - use carefully
let vFlip = RandomVerticalFlip(probability: 0.5)
// Only when orientation doesn't matter (satellite, microscopy)
```

**Important:** For detection, flip bounding box coordinates too!

### Translation

Shift image position:

```
Original        Translated (+10, +5)
┌──────────┐   ┌──────────┐
│ 🐕       │   │          │
│          │ → │    🐕    │
│          │   │          │
└──────────┘   └──────────┘
```

```swift
let translate = RandomTranslate(
    xRange: (-0.1, 0.1),  // ±10% of width
    yRange: (-0.1, 0.1),  // ±10% of height
    fillMode: .reflect
)
```

### Shearing

Skew the image:

```
Original        Shear X         Shear Y
┌────────┐     ╱────────╲      ┌────────┐
│  ████  │    ╱  ████    ╲     │╲ ████ ╱│
│  ████  │ →  ╲  ████    ╱  →  │ ╲████╱ │
│  ████  │     ╲────────╱      │  ╲██╱  │
└────────┘                     └────────┘
```

```swift
let shear = RandomShear(
    xRange: (-0.2, 0.2),  // ±20% shear
    yRange: (-0.2, 0.2)
)
```

### Perspective Transform

Simulate viewpoint change:

```
Original (frontal)      Perspective (angled)
┌──────────────┐       ╱──────────────╲
│              │      ╱                ╲
│    🖼️        │  →  │      🖼️         │
│              │      ╲                ╱
└──────────────┘       ╲──────────────╱
```

```swift
let perspective = RandomPerspective(
    distortionScale: 0.2,
    probability: 0.5
)
```

### Elastic Deformation

Non-linear warping (great for handwriting, medical):

```
Original        Elastic deformation
┌──────┐       ┌──────┐
│  5   │   →   │  5   │  (wavy deformation)
│      │       │ ~~~  │
└──────┘       └──────┘
```

```swift
let elastic = ElasticTransform(
    alpha: 50,      // Deformation intensity
    sigma: 5,       // Smoothness
    probability: 0.5
)
```

---

## Photometric Transformations Deep Dive

### Brightness Adjustment

```
Brightness formula:
output = input × brightness_factor

factor < 1: darker
factor = 1: unchanged
factor > 1: brighter

┌────────┐  factor=0.5  ┌────────┐  factor=1.5  ┌────────┐
│ ████   │     →       │ ▓▓▓▓   │      →      │ ████   │
│ bright │             │ darker │             │ lighter│
└────────┘             └────────┘             └────────┘
```

```swift
let brightness = RandomBrightness(
    range: (0.7, 1.3)  // ±30% brightness
)
```

### Contrast Adjustment

```
Contrast formula:
output = (input - mean) × contrast_factor + mean

factor < 1: less contrast (gray)
factor = 1: unchanged
factor > 1: more contrast (vivid)

┌────────┐  factor=0.5 ┌────────┐  factor=1.5 ┌────────┐
│ █░█░█  │     →       │ ▓░▓░▓  │      →      │ █ █ █  │
│ normal │             │ washed │             │ vivid  │
└────────┘             └────────┘             └────────┘
```

```swift
let contrast = RandomContrast(
    range: (0.8, 1.2)  // ±20% contrast
)
```

### Saturation Adjustment

```
Saturation formula (in HSV space):
S_new = S × saturation_factor

factor = 0: grayscale
factor = 1: unchanged  
factor > 1: more saturated

┌────────┐  factor=0   ┌────────┐  factor=2   ┌────────┐
│ 🔴🟢🔵 │     →       │ ⚪⚪⚪  │      →      │ 🔴🟢🔵 │
│ colors │             │ gray   │             │ vivid  │
└────────┘             └────────┘             └────────┘
```

```swift
let saturation = RandomSaturation(
    range: (0.5, 1.5)
)
```

### Hue Shift

```
Hue shift (in HSV space):
H_new = (H + shift) mod 360

Rotates all colors around the color wheel:

Original:     +60°:        +180°:
🔴 Red    →   🟡 Yellow  →  🔵 Blue (opposite)
```

```swift
let hue = RandomHue(
    range: (-0.1, 0.1)  // ±10% of color wheel
)
```

### Color Jitter

Combine all color transformations:

```swift
let colorJitter = ColorJitter(
    brightness: 0.2,    // ±20%
    contrast: 0.2,      // ±20%
    saturation: 0.2,    // ±20%
    hue: 0.1            // ±10%
)

// Applies all four in random order
```

### Gamma Correction

Non-linear brightness adjustment:

```
Gamma formula:
output = input^gamma

gamma < 1: brighten dark regions more
gamma = 1: unchanged
gamma > 1: darken bright regions more

Useful for simulating different camera responses
```

```swift
let gamma = RandomGamma(
    range: (0.8, 1.2)
)
```

### Histogram Equalization

Automatic contrast enhancement:

```
Before:                 After:
Histogram bunched      Histogram spread

│    ████              │ ██    ██
│    ████           →  │ ████████
│    ████              │ ████████
└──────────            └──────────
 dark                   full range
```

```swift
let clahe = CLAHE(
    clipLimit: 2.0,
    gridSize: (8, 8)
)
```

---

## Noise and Blur Techniques

### Gaussian Noise

```
Gaussian noise formula:
output = input + N(0, σ²)

Where N(0, σ²) is random Gaussian with variance σ²

Original        + Noise         Result
┌────────┐     ┌────────┐     ┌────────┐
│ ████   │  +  │ ░▒▓░▒▓ │  =  │ █▓█▓   │
│ smooth │     │ random │     │ noisy  │
└────────┘     └────────┘     └────────┘
```

```swift
let gaussianNoise = GaussianNoise(
    variance: 0.01,      // Noise strength
    perChannel: true     // Independent per RGB
)
```

### Salt and Pepper Noise

```
Salt and pepper:
Randomly set pixels to min (0) or max (255)

Original        Result
┌────────┐     ┌────────┐
│ ████   │  →  │ █○█●   │  ○ = white (salt)
│ ████   │     │ ●██○   │  ● = black (pepper)
└────────┘     └────────┘
```

```swift
let saltPepper = SaltAndPepperNoise(
    amount: 0.02,       // 2% of pixels affected
    saltRatio: 0.5      // Equal salt and pepper
)
```

### Speckle Noise

```
Speckle (multiplicative) noise:
output = input × (1 + N(0, σ²))

Common in radar, ultrasound images
```

```swift
let speckle = SpeckleNoise(
    variance: 0.05
)
```

### Gaussian Blur

```
Gaussian blur kernel (σ=1):
┌────────────────────┐
│ 1  2  1            │
│ 2  4  2  × (1/16)  │
│ 1  2  1            │
└────────────────────┘

Original        Blurred
┌────────┐     ┌────────┐
│ sharp  │  →  │ soft   │
│ edges  │     │ edges  │
└────────┘     └────────┘
```

```swift
let blur = GaussianBlur(
    kernelSize: (3, 7),   // Random size 3-7
    sigma: (0.1, 2.0)     // Random sigma
)
```

### Motion Blur

```
Motion blur simulates camera/object movement:

Kernel for horizontal motion:
┌──────────────────┐
│ 1 1 1 1 1 1 1    │  ← direction of motion
└──────────────────┘

Original        Motion blur
┌────────┐     ┌────────┐
│   🚗   │  →  │ ════🚗 │
│ still  │     │ moving │
└────────┘     └────────┘
```

```swift
let motionBlur = MotionBlur(
    kernelSize: (5, 15),
    angle: (-45, 45)      // Direction range
)
```

### Defocus Blur

```
Simulates out-of-focus camera:
Uses disk-shaped kernel (bokeh)

Original        Defocused
┌────────┐     ┌────────┐
│ ★★★    │  →  │ ◉◉◉    │
│ sharp  │     │ soft   │
└────────┘     └────────┘
```

```swift
let defocus = DefocusBlur(
    radius: (3, 7)
)
```

---

## Occlusion and Dropout

### Random Erasing (Cutout)

```
Randomly erase rectangular region:

Original        Cutout
┌──────────┐   ┌──────────┐
│    🐕    │   │    🐕    │
│   here   │ → │   ████   │
│          │   │          │
└──────────┘   └──────────┘

Forces network to use multiple features,
not just one discriminative part
```

```swift
let cutout = RandomErasing(
    probability: 0.5,
    scale: (0.02, 0.33),   // 2-33% of image area
    ratio: (0.3, 3.3),     // Aspect ratio
    value: 0               // Fill with black (or random)
)
```

### GridMask

```
Regular grid pattern of occlusion:

Original        GridMask
┌──────────┐   ┌──────────┐
│ 🐕🐕🐕🐕│   │ 🐕░🐕░🐕│
│ 🐕🐕🐕🐕│ → │ ░░░░░░░  │
│ 🐕🐕🐕🐕│   │ 🐕░🐕░🐕│
└──────────┘   └──────────┘

More structured than random erasing
```

```swift
let gridMask = GridMask(
    ratio: 0.5,           // 50% coverage
    gridSize: (4, 4)
)
```

### CoarseDropout

```
Multiple random rectangular regions:

Original        CoarseDropout
┌──────────┐   ┌──────────┐
│ 🐕       │   │ █🐕█     │
│   dog    │ → │   d█g    │
│          │   │    █     │
└──────────┘   └──────────┘
```

```swift
let coarse = CoarseDropout(
    maxHoles: 8,
    maxHeight: 32,
    maxWidth: 32,
    fillValue: 0
)
```

---

## Advanced Augmentation Methods

### MixUp

Mix two images and their labels:

```
MixUp formula:
x_mixed = λ × x_a + (1-λ) × x_b
y_mixed = λ × y_a + (1-λ) × y_b

Where λ ~ Beta(α, α), typically α=0.2

Image A (dog)   Image B (cat)   Mixed (λ=0.6)
┌────────┐     ┌────────┐      ┌────────┐
│  🐕    │ +   │  🐈    │  =   │  👻    │
│        │     │        │      │  ghost │
└────────┘     └────────┘      └────────┘
Label: [1, 0]  Label: [0, 1]   Label: [0.6, 0.4]
```

```swift
let mixup = MixUp(
    alpha: 0.2
)

// In training loop:
let (mixedImage, mixedLabel) = mixup.apply(imageA, labelA, imageB, labelB)
```

**Benefits:**
- Smooth decision boundaries
- Better calibration
- Reduced overfitting

### CutMix

Paste region from one image to another:

```
CutMix:
Cut rectangular region from B, paste into A
Mix labels by area ratio

Image A (dog)   Image B (cat)   CutMix
┌────────┐     ┌────────┐      ┌────────┐
│  🐕    │ +   │  🐈    │  =   │  🐕🐈 │
│        │     │        │      │        │
└────────┘     └────────┘      └────────┘
                               Label based on area:
                               If 30% is cat region:
                               [0.7, 0.3]
```

```swift
let cutmix = CutMix(
    alpha: 1.0
)
```

**Better than MixUp for:**
- Localization tasks
- Preserving natural image statistics

### Mosaic

Combine 4 images (used in YOLO):

```
┌─────────────────────────────────┐
│ Image 1      │      Image 2     │
│    🚗        │        🚌       │
│──────────────┼─────────────────│
│ Image 3      │      Image 4    │
│    🚶        │        🚲       │
└─────────────────────────────────┘

One training sample contains 4 images
Great for detection - sees multiple objects
```

```swift
let mosaic = Mosaic(
    imageSize: (640, 640)
)
```

### AutoAugment

**Learned augmentation policy:**

```
AutoAugment searches for optimal policy using RL:

Policy = [
    Sub-policy 1: [(op1, prob1, mag1), (op2, prob2, mag2)]
    Sub-policy 2: [(op3, prob3, mag3), (op4, prob4, mag4)]
    ...
]

Example ImageNet policy:
Sub-policy 1: Posterize(0.4, 8), Rotate(0.6, 9)
Sub-policy 2: Equalize(0.8, _), Equalize(0.6, _)
...
```

```swift
let autoAugment = AutoAugment(
    policy: .imagenet  // or .cifar10, .svhn
)
```

### RandAugment

**Simpler than AutoAugment:**

```
RandAugment:
- N: number of augmentations to apply
- M: magnitude (0-30, shared across all ops)

Apply N random augmentations at magnitude M

Much simpler to tune than AutoAugment!
```

```swift
let randAugment = RandAugment(
    n: 2,      // Apply 2 augmentations
    m: 9       // Magnitude 9 out of 30
)
```

### TrivialAugment

**Even simpler - no hyperparameters:**

```
TrivialAugment:
- Pick ONE random augmentation
- Pick random magnitude
- Apply it

That's it! Surprisingly effective.
```

```swift
let trivialAugment = TrivialAugment()
```

### AugMax

Adversarial-style augmentation:

```swift
let augmax = AugMax(
    numAugmentations: 3,
    severity: 3
)
// Applies multiple augmentations and takes the "hardest" one
```

---

## Task-Specific Augmentation

### Classification Augmentation

```swift
// Standard ImageNet augmentation
let trainAugmentation = Compose([
    RandomResizedCrop(size: (224, 224), scale: (0.08, 1.0)),
    RandomHorizontalFlip(p: 0.5),
    RandAugment(n: 2, m: 9),
    Normalize(mean: imagenetMean, std: imagenetStd)
])

// Validation - minimal augmentation
let valAugmentation = Compose([
    Resize(size: 256),
    CenterCrop(size: (224, 224)),
    Normalize(mean: imagenetMean, std: imagenetStd)
])
```

### Detection Augmentation

**Must transform boxes along with images!**

```swift
let detectionAugmentation = Compose([
    // Geometric - need to adjust boxes
    RandomHorizontalFlip(p: 0.5),       // Flip boxes too
    RandomResizedCrop(scale: (0.5, 1.0)), // Adjust box coords
    Mosaic(p: 0.5),                      // Combine box lists
    
    // Photometric - boxes unchanged
    ColorJitter(brightness: 0.2, contrast: 0.2),
    
    // YOLO-specific
    RandomPerspective(p: 0.5),
    MixUp(p: 0.1)                        // Mix box lists + labels
])

// Filter boxes:
// - Remove boxes outside crop
// - Remove boxes with small area after transform
// - Update box coordinates for flips/crops
```

### Segmentation Augmentation

**Must transform masks identically to images!**

```swift
let segmentationAugmentation = Compose([
    // Same transform to image AND mask
    RandomResizedCrop(size: (512, 512)),
    RandomHorizontalFlip(p: 0.5),
    RandomRotation(degrees: 10),
    
    // Only to image (mask unchanged)
    ColorJitter(brightness: 0.2, contrast: 0.2),
    GaussianBlur(kernel: 3)
])

// Apply:
let (augImage, augMask) = augmentation.apply(image, mask)

// IMPORTANT: Use NEAREST interpolation for mask!
// Mask contains class indices, not continuous values
```

### Medical Imaging

```swift
let medicalAugmentation = Compose([
    // Geometric
    RandomRotation(degrees: 15),
    RandomScale(scale: (0.9, 1.1)),
    ElasticTransform(alpha: 50),        // Great for tissue
    
    // Intensity
    RandomGamma(range: (0.8, 1.2)),
    RandomContrast(range: (0.9, 1.1)),
    
    // Medical-specific
    GaussianNoise(variance: 0.01),      // Simulate sensor noise
    
    // NO hue shift (color is diagnostic!)
    // Conservative transforms (preserve clinical validity)
])
```

---

## Augmentation Policies

### Random Application

```swift
// Each augmentation applied with probability p
let augmentation = RandomApply([
    GaussianBlur(kernel: 3),
    GaussianNoise(variance: 0.01)
], p: 0.5)

// 50% chance either is applied
```

### Sequential Pipelines

```swift
// Apply in order
let pipeline = Compose([
    RandomResizedCrop(size: (224, 224)),  // 1st
    RandomHorizontalFlip(p: 0.5),         // 2nd
    ColorJitter(...),                      // 3rd
    Normalize(...)                         // 4th (always last)
])
```

### Composition Strategies

```swift
// OneOf: Apply exactly one from list
let colorAug = OneOf([
    ColorJitter(brightness: 0.2),
    RandomGamma(range: (0.8, 1.2)),
    CLAHE(clipLimit: 2.0)
], p: [0.4, 0.3, 0.3])

// SomeOf: Apply k from list
let someAugs = SomeOf([
    HorizontalFlip(),
    Rotate(limit: 15),
    RandomCrop(size: 200),
    Blur(limit: 3)
], k: 2)  // Apply 2 random augmentations
```

---

## Decision Guide: Which Augmentations to Use

- **Classification**: flips, crops, color jitter, mild blur.
- **Detection**: scale/translate, mosaic, cutout (preserve box labels).
- **Segmentation**: elastic transforms, photometric jitter (preserve masks).
- **Medical/industrial**: conservative changes; avoid heavy color shifts.

If the model overfits, increase augmentation strength; if it underfits or loses small details, reduce geometric distortions.

---

## Invariance vs Equivariance

Augmentations teach the model to be **invariant** (output unchanged) or **equivariant** (output changes predictably).

- **Classification** aims for invariance to flips/crops.
- **Detection/Segmentation** require equivariance: boxes/masks must transform with the image.

If labels are not transformed consistently, training will degrade even if augmentation looks correct.

## SwiftPixelUtils Augmentation API

### Basic Usage

```swift
import SwiftPixelUtils

// Create augmentation
let augmentation = ImageAugmentation.colorJitter(
    brightness: 0.2,
    contrast: 0.2,
    saturation: 0.2,
    hue: 0.1
)

// Apply to image
let augmented = try augmentation.apply(to: image)
```

### Available Augmentations

```swift
// Geometric
ImageAugmentation.horizontalFlip()
ImageAugmentation.verticalFlip()
ImageAugmentation.rotate(degrees: (-15, 15))
ImageAugmentation.randomCrop(size: (224, 224))
ImageAugmentation.randomResizedCrop(size: (224, 224), scale: (0.08, 1.0))
ImageAugmentation.perspective(distortion: 0.2)

// Photometric
ImageAugmentation.brightness(range: (0.8, 1.2))
ImageAugmentation.contrast(range: (0.8, 1.2))
ImageAugmentation.saturation(range: (0.5, 1.5))
ImageAugmentation.hue(range: (-0.1, 0.1))
ImageAugmentation.colorJitter(...)
ImageAugmentation.grayscale(probability: 0.1)

// Noise/Blur
ImageAugmentation.gaussianNoise(variance: 0.01)
ImageAugmentation.gaussianBlur(kernel: (3, 7))
ImageAugmentation.motionBlur(kernel: (5, 15))

// Occlusion
ImageAugmentation.randomErasing(probability: 0.5)
ImageAugmentation.cutout(size: (32, 32), count: 1)

// Advanced
ImageAugmentation.mixup(alpha: 0.2)
ImageAugmentation.cutmix(alpha: 1.0)
ImageAugmentation.randAugment(n: 2, m: 9)
```

### Pipeline Construction

```swift
let pipeline = AugmentationPipeline([
    .randomResizedCrop(size: (224, 224), scale: (0.08, 1.0)),
    .horizontalFlip(probability: 0.5),
    .colorJitter(brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1),
    .normalize(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
])

let augmented = try pipeline.apply(to: image)
```

### Real-Time Augmentation

```swift
// For on-device training or preview
class AugmentationPreview {
    let pipeline: AugmentationPipeline
    
    func generateVariations(from image: UIImage, count: Int) -> [UIImage] {
        return (0..<count).compactMap { _ in
            try? pipeline.apply(to: image)
        }
    }
}
```

---

## Complete Implementation Examples

### Example 1: ImageNet-Style Training

```swift
// Training augmentation
let trainTransform = AugmentationPipeline([
    .randomResizedCrop(size: (224, 224), scale: (0.08, 1.0), ratio: (0.75, 1.33)),
    .horizontalFlip(probability: 0.5),
    .randAugment(n: 2, m: 9),
    .normalize(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
])

// Validation
let valTransform = AugmentationPipeline([
    .resize(size: 256),
    .centerCrop(size: (224, 224)),
    .normalize(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
])

// Test-Time Augmentation (TTA)
func predictWithTTA(image: UIImage, model: Classifier) throws -> [Float] {
    let augmentations = [
        identity,
        .horizontalFlip(),
        .rotate(degrees: -5),
        .rotate(degrees: 5)
    ]
    
    var predictions: [[Float]] = []
    
    for aug in augmentations {
        let augmented = try aug.apply(to: image)
        let pred = try model.predict(augmented)
        predictions.append(pred)
    }
    
    // Average predictions
    return zip(predictions[0].indices, predictions[0]).map { i, _ in
        predictions.map { $0[i] }.reduce(0, +) / Float(predictions.count)
    }
}
```

### Example 2: Detection Training

```swift
// YOLO-style augmentation
let detectionPipeline = DetectionAugmentationPipeline([
    .mosaic(probability: 0.5),
    .randomPerspective(degrees: 10, translate: 0.1, scale: 0.5),
    .horizontalFlip(probability: 0.5),
    .hsv(hGain: 0.015, sGain: 0.7, vGain: 0.4),
    .mixup(probability: 0.1)
])

// Apply with box transformation
let (augImage, augBoxes) = try detectionPipeline.apply(
    image: image,
    boxes: boxes  // [(classId, x, y, w, h)]
)
```

---

## Best Practices

### 1. Start Simple, Add Complexity

```swift
// Stage 1: Baseline
let basic = [.randomCrop(), .horizontalFlip()]

// Stage 2: Add color
let withColor = basic + [.colorJitter()]

// Stage 3: Add advanced
let advanced = withColor + [.randAugment(n: 2, m: 9)]

// Evaluate each stage!
```

### 2. Match Augmentation to Domain

| Domain | Recommended | Avoid |
|--------|-------------|-------|
| Natural images | All common augs | - |
| Documents | Perspective, noise | Color changes |
| Medical | Conservative, elastic | Heavy distortion |
| Satellite | Rotation, flip | Hue shift |
| Faces | Conservative color | Heavy geometric |

### 3. Augmentation Strength Schedule

```swift
// Start with weak augmentation, increase during training
func getAugStrength(epoch: Int, totalEpochs: Int) -> Float {
    let progress = Float(epoch) / Float(totalEpochs)
    return min(1.0, progress * 1.5)  // Ramp up
}

let strength = getAugStrength(epoch: currentEpoch, totalEpochs: 100)
let augmentation = RandAugment(n: 2, m: Int(9 * strength))
```

---

## Common Mistakes

### 1. Augmenting Validation Data

```swift
// WRONG
let valAug = Compose([
    RandomCrop(...),        // NO random augs!
    ColorJitter(...),       // NO!
])

// CORRECT
let valAug = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(...)
])
```

### 2. Inconsistent Image/Mask Augmentation

```swift
// WRONG
let augImage = randomCrop(image)
let augMask = randomCrop(mask)  // Different random crop!

// CORRECT
let (augImage, augMask) = randomCrop.apply(image, mask)  // Same transform
```

### 3. Wrong Interpolation for Masks

```swift
// WRONG
let resizedMask = resize(mask, interpolation: .bilinear)
// Creates invalid class values like 0.5, 1.7

// CORRECT
let resizedMask = resize(mask, interpolation: .nearest)
// Preserves class indices 0, 1, 2, 3...
```

---

## Performance Considerations

### CPU vs GPU Augmentation

```swift
// CPU augmentation (common, easy)
let augmented = cpuAugment(image)

// GPU augmentation (faster for batches)
let augmented = metalAugment(imageBatch)

// Recommendation:
// - Single images: CPU is fine
// - Batch training: Consider GPU
// - On-device: Use Accelerate framework
```

### Memory-Efficient Augmentation

```swift
// Process in-place when possible
func augmentInPlace(_ buffer: inout [UInt8]) {
    // Modify buffer directly
}

// Avoid creating copies
let result = autoreleasepool {
    return heavyAugmentation(image)
}
```

---

## Augmentation Theory Notes

Augmentation improves **generalization** by teaching invariances (e.g., rotation, brightness) and reducing overfitting to spurious cues.

**Bias‑variance intuition:**
- Augmentation can reduce variance by expanding the effective dataset.
- Too‑aggressive transforms can introduce bias if they distort the task‑relevant signal.

**Distribution shift:**
Apply transforms that reflect real‑world variations at inference time. If the deployment domain is fixed (e.g., medical imaging), use conservative augmentations.

**Interpolation artifacts:**
Repeated resampling can blur details; prefer composing transforms and resampling once.

## Mathematical Foundations

### Affine Transformation Matrix

General 2D affine transform:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

| Transform | Matrix |
|-----------|--------|
| Rotation | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ |
| Scale | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$ |
| Shear | $\begin{bmatrix} 1 & sh_x \\ sh_y & 1 \end{bmatrix}$ |
| Translation | $t_x, t_y$ terms |

### Color Space Conversions

RGB to HSV:
$$V = \max(R, G, B)$$
$$S = \frac{V - \min(R,G,B)}{V}$$
$$H = \text{depends on which color is max}$$

### MixUp Label Smoothing

$$\tilde{y} = \lambda y_a + (1-\lambda) y_b$$

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$

