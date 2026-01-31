//
//  LabelDatabase.swift
//  SwiftPixelUtils
//
//  Built-in label databases for common ML models with JSON resource loading
//

import Foundation

/// Built-in label databases for common ML object detection, classification, and segmentation models.
///
/// ## Overview
///
/// ML models output class indices (0, 1, 2, ...) rather than human-readable labels.
/// This database provides the mapping from indices to labels for popular datasets.
///
/// ## Supported Datasets
///
/// | Dataset | Classes | Domain | Common Models |
/// |---------|---------|--------|---------------|
/// | **ImageNet** | 1000 | Fine-grained classification | ResNet, EfficientNet, ViT, MobileNet |
/// | **COCO** | 80/91 | Object detection | YOLO, SSD, Faster R-CNN, DETR |
/// | **CIFAR-10** | 10 | Basic classification | ResNet, VGG, WideResNet |
/// | **CIFAR-100** | 100 | Fine-grained classification | ResNet, DenseNet |
/// | **VOC** | 21 | Detection/segmentation | Older detectors, DeepLab |
/// | **Places365** | 365 | Scene recognition | ResNet-Places, VGG-Places |
/// | **ADE20K** | 150 | Semantic segmentation | SegFormer, PSPNet, DeepLab |
///
/// ## Architecture
///
/// Labels are stored in two ways for optimal performance:
/// 1. **Embedded labels**: Small datasets (COCO, CIFAR-10, VOC) are compiled into the binary
/// 2. **JSON resources**: Large datasets (ImageNet-1K, Places365) are loaded from bundled JSON
///
/// This hybrid approach balances memory efficiency with startup performance.
///
/// ## Usage
///
/// ```swift
/// // Single label lookup
/// let label = LabelDatabase.getLabel(0, dataset: .imagenet)  // "tench"
///
/// // Top-K predictions from model output
/// let predictions = LabelDatabase.getTopLabels(
///     scores: modelOutput,
///     dataset: .imagenet,
///     k: 5,
///     minConfidence: 0.1
/// )
/// for pred in predictions {
///     print("\(pred.label): \(String(format: "%.1f", pred.confidence * 100))%")
/// }
///
/// // Load custom labels from your own JSON file
/// try LabelDatabase.loadCustomLabels(
///     from: customURL,
///     as: "my_custom_dataset"
/// )
/// ```
///
/// ## JSON Format
///
/// Custom label files should follow this format:
/// ```json
/// {
///   "dataset": "my_dataset",
///   "num_classes": 10,
///   "labels": ["class0", "class1", ...]
/// }
/// ```
public enum LabelDatabase {
    
    // MARK: - Cache for Loaded Labels
    
    /// Thread-safe cache for lazily loaded label arrays
    /// 
    /// All datasets are loaded from bundled JSON resources on first access.
    /// This approach provides:
    /// - Consistent data format across all datasets
    /// - Easy updates without code changes
    /// - Memory efficiency through lazy loading
    private static var labelCache: [LabelDataset: [String]] = [:]
    private static var customLabelCache: [String: [String]] = [:]
    private static let cacheLock = NSLock()
    
    // MARK: - Public API
    
    /// Get a label by index for a standard dataset.
    ///
    /// - Parameters:
    ///   - index: Class index from model output (0-based)
    ///   - dataset: Dataset to use for label lookup
    /// - Returns: Human-readable label string, or nil if index is out of bounds
    ///
    /// ## Example
    /// ```swift
    /// let label = LabelDatabase.getLabel(0, dataset: .imagenet)
    /// print(label)  // "tench"
    ///
    /// let cocoLabel = LabelDatabase.getLabel(0, dataset: .coco)
    /// print(cocoLabel)  // "person"
    /// ```
    public static func getLabel(_ index: Int, dataset: LabelDataset = .coco) -> String? {
        let labels = getLabels(for: dataset)
        guard index >= 0 && index < labels.count else {
            return nil
        }
        return labels[index]
    }
    
    /// Get a label by index from a custom loaded dataset.
    ///
    /// - Parameters:
    ///   - index: Class index from model output (0-based)
    ///   - customDataset: Name used when loading the custom dataset
    /// - Returns: Human-readable label string, or nil if not found
    public static func getCustomLabel(_ index: Int, customDataset: String) -> String? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        
        guard let labels = customLabelCache[customDataset],
              index >= 0 && index < labels.count else {
            return nil
        }
        return labels[index]
    }
    
    /// Get top-K labels from prediction scores.
    ///
    /// Sorts predictions by confidence and returns the top results with their labels.
    /// Useful for displaying classification results to users.
    ///
    /// - Parameters:
    ///   - scores: Raw prediction scores from model (one per class)
    ///   - dataset: Dataset for label lookup
    ///   - k: Maximum number of results to return (default: 5)
    ///   - minConfidence: Minimum confidence threshold (default: 0.0)
    /// - Returns: Array of tuples with label, confidence score, and original index
    ///
    /// ## Example
    /// ```swift
    /// // Assuming modelOutput is [Float] with 1000 scores from ImageNet model
    /// let top5 = LabelDatabase.getTopLabels(
    ///     scores: modelOutput,
    ///     dataset: .imagenet,
    ///     k: 5,
    ///     minConfidence: 0.01
    /// )
    ///
    /// for (label, confidence, index) in top5 {
    ///     print("\(index): \(label) - \(confidence * 100)%")
    /// }
    /// // Output:
    /// // 281: tabby cat - 89.2%
    /// // 282: tiger cat - 7.3%
    /// // 285: Egyptian Mau - 2.1%
    /// ```
    public static func getTopLabels(
        scores: [Float],
        dataset: LabelDataset = .coco,
        k: Int = 5,
        minConfidence: Float = 0.0
    ) -> [(label: String, confidence: Float, index: Int)] {
        let labels = getLabels(for: dataset)
        return computeTopLabels(scores: scores, labels: labels, k: k, minConfidence: minConfidence)
    }
    
    /// Get top-K labels from a custom dataset.
    ///
    /// - Parameters:
    ///   - scores: Raw prediction scores from model
    ///   - customDataset: Name of the custom dataset
    ///   - k: Maximum number of results
    ///   - minConfidence: Minimum confidence threshold
    /// - Returns: Array of tuples with label, confidence score, and original index
    public static func getTopCustomLabels(
        scores: [Float],
        customDataset: String,
        k: Int = 5,
        minConfidence: Float = 0.0
    ) -> [(label: String, confidence: Float, index: Int)] {
        cacheLock.lock()
        let labels = customLabelCache[customDataset] ?? []
        cacheLock.unlock()
        
        return computeTopLabels(scores: scores, labels: labels, k: k, minConfidence: minConfidence)
    }
    
    /// Get all labels for a dataset.
    ///
    /// - Parameter dataset: Dataset to retrieve labels for
    /// - Returns: Array of all class labels in index order
    ///
    /// ## Note
    /// For large datasets like ImageNet-1K, labels are loaded lazily from
    /// bundled JSON resources on first access.
    public static func getAllLabels(for dataset: LabelDataset) -> [String] {
        return getLabels(for: dataset)
    }
    
    /// Get dataset metadata and information.
    ///
    /// - Parameter dataset: Dataset to query
    /// - Returns: Struct containing dataset name, class count, and description
    public static func getDatasetInfo(for dataset: LabelDataset) -> DatasetInfo {
        let labels = getLabels(for: dataset)
        
        return DatasetInfo(
            name: dataset.rawValue,
            numClasses: labels.count,
            description: getDescription(for: dataset)
        )
    }
    
    /// Get all available built-in datasets.
    ///
    /// - Returns: Array of all supported dataset identifiers
    public static func getAvailableDatasets() -> [LabelDataset] {
        return LabelDataset.allCases
    }
    
    /// Get names of all loaded custom datasets.
    ///
    /// - Returns: Array of custom dataset names
    public static func getLoadedCustomDatasets() -> [String] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return Array(customLabelCache.keys)
    }
    
    // MARK: - Custom Label Loading
    
    /// Load custom labels from a JSON file.
    ///
    /// The JSON file should contain a "labels" array with string class names.
    /// Optional fields: "dataset", "num_classes", "description".
    ///
    /// - Parameters:
    ///   - url: URL to the JSON file
    ///   - name: Name to reference this dataset (used in getCustomLabel)
    /// - Throws: `PixelUtilsError.invalidInput` if file cannot be parsed
    ///
    /// ## JSON Format
    /// ```json
    /// {
    ///   "labels": ["class0", "class1", "class2", ...]
    /// }
    /// ```
    ///
    /// ## Example
    /// ```swift
    /// let customURL = Bundle.main.url(forResource: "my_labels", withExtension: "json")!
    /// try LabelDatabase.loadCustomLabels(from: customURL, as: "my_model")
    ///
    /// let label = LabelDatabase.getCustomLabel(0, customDataset: "my_model")
    /// ```
    public static func loadCustomLabels(from url: URL, as name: String) throws {
        let data = try Data(contentsOf: url)
        let labels = try parseLabelsJSON(data)
        
        cacheLock.lock()
        customLabelCache[name] = labels
        cacheLock.unlock()
    }
    
    /// Load custom labels from JSON data.
    ///
    /// - Parameters:
    ///   - data: JSON data containing labels array
    ///   - name: Name to reference this dataset
    /// - Throws: `PixelUtilsError.invalidInput` if data cannot be parsed
    public static func loadCustomLabels(from data: Data, as name: String) throws {
        let labels = try parseLabelsJSON(data)
        
        cacheLock.lock()
        customLabelCache[name] = labels
        cacheLock.unlock()
    }
    
    /// Unload a custom dataset to free memory.
    ///
    /// - Parameter name: Name of the custom dataset to unload
    public static func unloadCustomLabels(name: String) {
        cacheLock.lock()
        customLabelCache.removeValue(forKey: name)
        cacheLock.unlock()
    }
    
    /// Clear all cached labels to free memory.
    ///
    /// This removes both built-in cached labels (they'll be reloaded on next access)
    /// and custom labels (which must be reloaded manually).
    public static func clearCache() {
        cacheLock.lock()
        labelCache.removeAll()
        customLabelCache.removeAll()
        cacheLock.unlock()
    }
    
    // MARK: - Private Implementation
    
    private static func getLabels(for dataset: LabelDataset) -> [String] {
        // Check cache first
        cacheLock.lock()
        if let cached = labelCache[dataset] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()
        
        // Load labels from JSON based on dataset
        let labels: [String]
        switch dataset {
        case .coco:
            labels = loadJSONLabels(filename: "coco_labels")
        case .coco91:
            // COCO-91 uses original category IDs with background at index 0
            labels = loadJSONLabels(filename: "coco91_labels")
        case .cifar10:
            labels = loadJSONLabels(filename: "cifar10_labels")
        case .voc:
            labels = loadJSONLabels(filename: "voc_labels")
        case .imagenet:
            labels = loadJSONLabels(filename: "imagenet_labels")
        case .imagenet21k:
            // ImageNet-21K WordNet synset labels (~21,843 classes)
            labels = loadJSONLabels(filename: "imagenet21k_labels")
        case .cifar100:
            labels = loadJSONLabels(filename: "cifar100_labels")
        case .places365:
            labels = loadJSONLabels(filename: "places365_labels")
        case .ade20k:
            labels = loadJSONLabels(filename: "ade20k_labels")
        }
        
        // Cache the result
        cacheLock.lock()
        labelCache[dataset] = labels
        cacheLock.unlock()
        
        return labels
    }
    
    private static func loadJSONLabels(filename: String) -> [String] {
        // Try to load from bundle resources
        guard let url = Bundle.module.url(forResource: filename, withExtension: "json") else {
            print("Warning: Could not find \(filename).json in bundle resources")
            return []
        }
        
        do {
            let data = try Data(contentsOf: url)
            return try parseLabelsJSON(data)
        } catch {
            print("Warning: Failed to load \(filename).json: \(error)")
            return []
        }
    }
    
    private static func parseLabelsJSON(_ data: Data) throws -> [String] {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let labels = json["labels"] as? [String] else {
            throw PixelUtilsError.invalidOptions("Invalid JSON format: expected object with 'labels' array")
        }
        return labels
    }
    
    private static func computeTopLabels(
        scores: [Float],
        labels: [String],
        k: Int,
        minConfidence: Float
    ) -> [(label: String, confidence: Float, index: Int)] {
        // Create indexed scores
        let indexedScores = scores.enumerated().map { ($0.offset, $0.element) }
        
        // Filter by confidence and sort
        let filtered = indexedScores
            .filter { $0.1 >= minConfidence }
            .sorted { $0.1 > $1.1 }
            .prefix(k)
        
        // Map to results
        return filtered.map { index, confidence in
            let label = index < labels.count ? labels[index] : "unknown_\(index)"
            return (label: label, confidence: confidence, index: index)
        }
    }
    
    private static func getDescription(for dataset: LabelDataset) -> String {
        switch dataset {
        case .coco:
            return "COCO 2017 object detection labels (80 classes)"
        case .coco91:
            return "COCO original labels with background (91 classes)"
        case .imagenet:
            return "ImageNet ILSVRC 2012 classification (1000 classes)"
        case .imagenet21k:
            return "ImageNet-21K full classification (21841 classes) - load externally"
        case .voc:
            return "PASCAL VOC with background (21 classes)"
        case .cifar10:
            return "CIFAR-10 classification (10 classes)"
        case .cifar100:
            return "CIFAR-100 fine-grained classification (100 classes)"
        case .places365:
            return "Places365 scene recognition (365 classes)"
        case .ade20k:
            return "ADE20K semantic segmentation (150 classes)"
        }
    }
}

// MARK: - Supporting Types

/// Supported label datasets.
///
/// Each dataset corresponds to a well-known ML benchmark with predefined class labels.
public enum LabelDataset: String, CaseIterable, Sendable {
    /// COCO 2017 with 80 object detection classes
    case coco = "coco"
    /// COCO original format with 91 classes (includes background)
    case coco91 = "coco91"
    /// ImageNet ILSVRC 2012 with 1000 classification classes
    case imagenet = "imagenet"
    /// ImageNet-21K with 21841 classes (must load externally)
    case imagenet21k = "imagenet21k"
    /// PASCAL VOC with 21 classes (includes background)
    case voc = "voc"
    /// CIFAR-10 with 10 basic classes
    case cifar10 = "cifar10"
    /// CIFAR-100 with 100 fine-grained classes
    case cifar100 = "cifar100"
    /// Places365 with 365 scene categories
    case places365 = "places365"
    /// ADE20K with 150 semantic segmentation classes
    case ade20k = "ade20k"
}

/// Dataset metadata and information.
public struct DatasetInfo: Sendable {
    /// Dataset identifier name
    public let name: String
    /// Number of classes in the dataset
    public let numClasses: Int
    /// Human-readable description
    public let description: String
    
    public init(name: String, numClasses: Int, description: String) {
        self.name = name
        self.numClasses = numClasses
        self.description = description
    }
}

// MARK: - Convenience Extensions

extension LabelDatabase {
    /// Apply softmax to raw logits and get top-K labels.
    ///
    /// Many models output raw logits that need softmax normalization
    /// to convert to probabilities. This method handles both steps.
    ///
    /// ## Softmax Formula
    /// $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
    ///
    /// - Parameters:
    ///   - logits: Raw model output logits
    ///   - dataset: Dataset for label lookup
    ///   - k: Maximum number of results
    ///   - minConfidence: Minimum probability threshold
    /// - Returns: Array of tuples with label, probability, and index
    public static func getTopLabelsWithSoftmax(
        logits: [Float],
        dataset: LabelDataset = .imagenet,
        k: Int = 5,
        minConfidence: Float = 0.0
    ) -> [(label: String, confidence: Float, index: Int)] {
        let probabilities = softmax(logits)
        return getTopLabels(scores: probabilities, dataset: dataset, k: k, minConfidence: minConfidence)
    }
    
    /// Compute softmax probabilities from logits.
    ///
    /// Uses the numerically stable version that subtracts the max value
    /// to prevent overflow: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
    ///
    /// - Parameter logits: Raw model output logits
    /// - Returns: Normalized probabilities that sum to 1.0
    public static func softmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        
        // Numerical stability: subtract max to prevent overflow
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        
        guard sumExp > 0 else {
            // Fallback to uniform distribution
            let uniform = 1.0 / Float(logits.count)
            return Array(repeating: uniform, count: logits.count)
        }
        
        return expValues.map { $0 / sumExp }
    }
}
