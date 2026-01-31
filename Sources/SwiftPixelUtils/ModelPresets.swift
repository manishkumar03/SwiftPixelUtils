//
//  ModelPresets.swift
//  SwiftPixelUtils
//
//  Pre-configured settings for common ML models
//

import Foundation
import CoreGraphics

/// Pre-configured preprocessing settings for popular ML models.
///
/// ## Overview
///
/// Different ML models expect different input preprocessing. Using incorrect settings
/// causes silent accuracy degradation or outright failures. These presets match the
/// exact preprocessing used during model training.
///
/// ## Usage
///
/// ```swift
/// // For YOLO object detection
/// let result = try await PixelExtractor.getPixelData(
///     from: .cgImage(image),
///     options: ModelPresets.yolo
/// )
///
/// // For CLIP embeddings
/// let result = try await PixelExtractor.getPixelData(
///     from: .cgImage(image),
///     options: ModelPresets.clip
/// )
/// ```
///
/// ## Why Settings Differ
///
/// | Model Family | Input Size | Normalization | Layout | Reason |
/// |-------------|------------|---------------|--------|--------|
/// | YOLO | 640×640 | Scale [0,1] | NCHW | Trained on COCO with letterboxing |
/// | MobileNet | 224×224 | ImageNet | NHWC | TensorFlow convention |
/// | ResNet | 224×224 | ImageNet | NCHW | PyTorch convention |
/// | CLIP | 224×224 | Custom | NCHW | OpenAI's LAION statistics |
/// | SAM | 1024×1024 | ImageNet | NCHW | High-res segmentation |
///
/// ## Creating Custom Presets
///
/// For models not listed, check the model's documentation for:
/// 1. **Input size**: Usually in model architecture or config
/// 2. **Normalization**: Mean/std values from training code
/// 3. **Data layout**: NCHW (PyTorch) vs NHWC (TensorFlow)
/// 4. **Channel order**: RGB (most) vs BGR (OpenCV-based)
public enum ModelPresets {
    
    // MARK: - YOLO Models
    
    /// YOLO (You Only Look Once) object detection preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 640×640**: YOLO's default input size, balances speed and accuracy.
    ///   Larger sizes (1280) improve small object detection but are slower.
    ///
    /// - **Letterbox resize**: Preserves aspect ratio by adding gray padding (114, 114, 114).
    ///   Critical for accurate bounding box predictions. Stretching would distort
    ///   learned aspect ratio priors.
    ///
    /// - **Scale normalization [0, 1]**: YOLO uses simple 0-1 scaling, not ImageNet stats.
    ///   This matches the original Darknet training code.
    ///
    /// - **NCHW layout**: YOLO implementations (Ultralytics, ONNX exports) use
    ///   channels-first format from PyTorch training.
    ///
    /// ## Why Letterbox?
    ///
    /// ```
    /// Original (16:9):        Letterboxed (1:1):
    /// ┌────────────────┐      ┌────────────────┐
    /// │                │      │░░░░░░░░░░░░░░░░│
    /// │     image      │  →   │     image      │
    /// │                │      │░░░░░░░░░░░░░░░░│
    /// └────────────────┘      └────────────────┘
    ///
    /// Gray padding (114) chosen to minimize edge artifacts at detection boundaries.
    /// ```
    ///
    /// ## Post-Processing Note
    ///
    /// Remember to scale bounding box outputs back to original image coordinates:
    /// ```swift
    /// let boxes = BoundingBox.scale(rawBoxes, from: CGSize(width: 640, height: 640), to: originalSize, format: .cxcywh)
    /// ```
    public static let yolo = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 640, height: 640, strategy: .letterbox),
        normalization: .scale,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// YOLOv8 uses the same preprocessing as YOLO base.
    public static let yolov8 = yolo
    
    // MARK: - MobileNet Models
    
    /// MobileNet family preprocessing (v1, v2, v3).
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ImageNet classification input size.
    ///   MobileNet was designed for efficient inference at this resolution.
    ///
    /// - **Cover resize**: Crops to fill the target, maximizing image content.
    ///   Classification assumes the subject is roughly centered.
    ///
    /// - **ImageNet normalization**: MobileNet trained on ImageNet uses the standard
    ///   mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
    ///
    /// - **NHWC layout**: TensorFlow convention (MobileNet originated at Google).
    ///   Use NCHW if using PyTorch ports.
    ///
    /// ## Performance Notes
    ///
    /// MobileNet uses depthwise separable convolutions for efficiency:
    /// - v1: ~4.2M parameters, ~569M MACs
    /// - v2: ~3.4M parameters, ~300M MACs (inverted residuals)
    /// - v3: ~5.4M parameters, ~219M MACs (neural architecture search)
    public static let mobilenet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nhwc,
        outputFormat: .float32Array
    )
    
    /// MobileNetV2 uses the same preprocessing as MobileNet base.
    public static let mobilenet_v2 = mobilenet
    
    /// MobileNetV3 uses the same preprocessing as MobileNet base.
    public static let mobilenet_v3 = mobilenet
    
    // MARK: - EfficientNet
    
    /// EfficientNet preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: B0 variant size. Larger variants use bigger inputs:
    ///   - B1: 240×240, B2: 260×260, B3: 300×300, B4: 380×380, etc.
    ///
    /// - **ImageNet normalization**: Standard ImageNet statistics.
    ///
    /// - **NHWC layout**: TensorFlow convention (EfficientNet is from Google Brain).
    ///
    /// ## Scaling Note
    ///
    /// EfficientNet uses compound scaling (depth, width, resolution together).
    /// For larger variants, modify the resize dimensions accordingly.
    public static let efficientnet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nhwc,
        outputFormat: .float32Array
    )
    
    // MARK: - ResNet Models
    
    /// ResNet family preprocessing (18, 34, 50, 101, 152).
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ImageNet classification input size.
    ///
    /// - **ImageNet normalization**: ResNet trained on ImageNet uses the standard
    ///   mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
    ///
    /// - **NCHW layout**: PyTorch convention (ResNet reference implementation).
    ///
    /// ## Architecture Note
    ///
    /// ResNet's skip connections enable training very deep networks without
    /// vanishing gradients. The identity mappings allow gradients to flow
    /// directly through the network during backpropagation.
    public static let resnet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// ResNet-50 uses the same preprocessing as ResNet base.
    public static let resnet50 = resnet
    
    // MARK: - Vision Transformer
    
    /// Vision Transformer (ViT) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ViT-B/16 input size (16×16 patches = 14×14 grid).
    ///   ViT-L uses 224×224 or 384×384 depending on variant.
    ///
    /// - **ImageNet normalization**: ViT models trained on ImageNet-21k or ImageNet-1k
    ///   use standard statistics.
    ///
    /// - **NCHW layout**: PyTorch/JAX convention for most ViT implementations.
    ///
    /// ## Patch Embedding
    ///
    /// ViT divides the image into fixed-size patches (typically 16×16 or 32×32),
    /// then linearly embeds each patch. A 224×224 image with 16×16 patches yields
    /// a sequence of 196 tokens (14 × 14), plus a [CLS] token for classification.
    public static let vit = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - CLIP
    
    /// CLIP (Contrastive Language-Image Pre-training) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: CLIP ViT-B/32 and ViT-B/16 input size.
    ///   CLIP ViT-L/14 uses 224×224, ViT-L/14@336px uses 336×336.
    ///
    /// - **Custom normalization**: CLIP uses statistics from LAION-400M dataset,
    ///   not ImageNet. These values differ slightly:
    ///   - Mean: [0.48145466, 0.4578275, 0.40821073]
    ///   - Std:  [0.26862954, 0.26130258, 0.27577711]
    ///
    /// - **NCHW layout**: PyTorch convention (OpenAI's implementation).
    ///
    /// ## Why Different Statistics?
    ///
    /// CLIP was trained on 400M image-text pairs from the internet, which has
    /// different color distributions than ImageNet's curated 1.2M images.
    /// The LAION dataset includes more diverse content (memes, photos, art),
    /// resulting in slightly different mean/std values.
    ///
    /// ## Zero-Shot Classification
    ///
    /// CLIP enables zero-shot classification by comparing image embeddings
    /// with text embeddings of class descriptions. No task-specific training required.
    public static let clip = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: Normalization(
            preset: .custom,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711]
        ),
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - SAM (Segment Anything Model)
    
    /// SAM (Segment Anything Model) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 1024×1024**: SAM uses high resolution for precise segmentation.
    ///   The image encoder (ViT-H) can handle this large input.
    ///
    /// - **Contain resize**: Preserves aspect ratio without cropping, important
    ///   for segmentation where all image content matters.
    ///
    /// - **ImageNet normalization**: SAM uses standard ImageNet statistics.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Architecture Note
    ///
    /// SAM's image encoder runs once per image, then the mask decoder can
    /// generate multiple masks for different prompts (points, boxes, text).
    /// The high input resolution enables fine-grained boundary predictions.
    public static let sam = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 1024, height: 1024, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - DINO
    
    /// DINO (Self-Distillation with No Labels) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ViT input size for DINO.
    ///
    /// - **ImageNet normalization**: DINO uses standard ImageNet statistics.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Self-Supervised Learning
    ///
    /// DINO learns visual features without labels by having a student network
    /// match the output of a teacher network (momentum-updated copy).
    /// The learned features are remarkably good for k-NN classification,
    /// object discovery, and semantic segmentation.
    public static let dino = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - DETR
    
    /// DETR (DEtection TRansformer) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 800×800**: DETR uses larger input for object detection.
    ///   The original paper uses 800 max dimension with aspect ratio preserved.
    ///
    /// - **Contain resize**: Preserves aspect ratio, important for accurate
    ///   bounding box predictions.
    ///
    /// - **ImageNet normalization**: DETR uses standard ImageNet statistics
    ///   (backbone is typically ResNet-50 pretrained on ImageNet).
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Transformer Detection
    ///
    /// DETR treats detection as a set prediction problem, using a transformer
    /// encoder-decoder to directly output a fixed set of predictions.
    /// Hungarian matching during training assigns predictions to ground truth.
    public static let detr = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 800, height: 800, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
}
