"""
Sliding Window Detector for Viola-Jones

Applies trained classifier to full images using sliding window approach.
Multi-scale detection via image pyramid.

Assignment: ELL715 Assignment 5 - Part 2 (40 marks)
AI Usage: Algorithm structure and docstrings assisted by Claude Code
"""

import numpy as np
from typing import List, Tuple

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("✓ Numba successfully imported - JIT optimization available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ Numba not available - using Python fallback implementations")
    print("  To enable JIT optimization: pip install numba")

    # Define dummy decorators when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)

try:
    from ..features.integral_image import compute_ii_fast
    from ..features.haar_features import generate_haar_features
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from features.integral_image import compute_ii_fast
    from features.haar_features import generate_haar_features


class Detection:
    """Single detection result"""

    def __init__(self, x, y, width, height, confidence):
        """
        Args:
            x, y: Top-left corner
            width, height: Bounding box size
            confidence: Detection score
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def __repr__(self):
        return f"Detection(x={self.x}, y={self.y}, size={self.width}x{self.height}, conf={self.confidence:.2f})"


@jit(nopython=True, cache=True)
def rectangle_sum_numba(ii, x, y, width, height):
    """
    Numba-compiled rectangle sum using integral image
    """
    y1, x1 = y, x
    y2, x2 = y + height, x + width
    return ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]


@jit(nopython=True, cache=True)
def compute_haar_features_numba(ii, feature_types, feature_coords):
    """
    Numba-compiled Haar feature computation for all features

    Args:
        ii: Integral image (H+1, W+1)
        feature_types: Array of feature type strings encoded as ints
                       0='2h', 1='2v', 2='3h', 3='3v', 4='4d'
        feature_coords: Array of shape (n_features, 4) with [x, y, width, height]

    Returns:
        responses: Array of feature responses (n_features,)
    """
    n_features = len(feature_types)
    responses = np.zeros(n_features, dtype=np.float64)

    for i in prange(n_features):
        ftype = feature_types[i]
        x, y, width, height = feature_coords[i]

        if ftype == 0:  # '2h' - Two horizontal rectangles
            h_half = height // 2
            top = rectangle_sum_numba(ii, x, y, width, h_half)
            bottom = rectangle_sum_numba(ii, x, y + h_half, width, h_half)
            responses[i] = bottom - top

        elif ftype == 1:  # '2v' - Two vertical rectangles
            w_half = width // 2
            left = rectangle_sum_numba(ii, x, y, w_half, height)
            right = rectangle_sum_numba(ii, x + w_half, y, w_half, height)
            responses[i] = right - left

        elif ftype == 2:  # '3h' - Three horizontal rectangles
            h_third = height // 3
            top = rectangle_sum_numba(ii, x, y, width, h_third)
            middle = rectangle_sum_numba(ii, x, y + h_third, width, h_third)
            bottom = rectangle_sum_numba(ii, x, y + 2*h_third, width, h_third)
            responses[i] = top + bottom - 2 * middle

        elif ftype == 3:  # '3v' - Three vertical rectangles
            w_third = width // 3
            left = rectangle_sum_numba(ii, x, y, w_third, height)
            middle = rectangle_sum_numba(ii, x + w_third, y, w_third, height)
            right = rectangle_sum_numba(ii, x + 2*w_third, y, w_third, height)
            responses[i] = left + right - 2 * middle

        elif ftype == 4:  # '4d' - Four diagonal rectangles
            h_half = height // 2
            w_half = width // 2
            top_left = rectangle_sum_numba(ii, x, y, w_half, h_half)
            top_right = rectangle_sum_numba(ii, x + w_half, y, w_half, h_half)
            bottom_left = rectangle_sum_numba(ii, x, y + h_half, w_half, h_half)
            bottom_right = rectangle_sum_numba(ii, x + w_half, y + h_half, w_half, h_half)
            responses[i] = top_left + bottom_right - top_right - bottom_left

    return responses


@jit(nopython=True, cache=True)
def compute_integral_image_numba(image):
    """
    Numba-compiled integral image computation
    """
    H, W = image.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)

    for y in prange(1, H + 1):
        for x in prange(1, W + 1):
            ii[y, x] = ii[y-1, x] + ii[y, x-1] - ii[y-1, x-1] + image[y-1, x-1]

    return ii


@jit(nopython=True, cache=True)
def process_window_numba(window, feature_types, feature_coords, classifier_weights,
                        classifier_threshold, total_alpha, threshold):
    """
    Numba-compiled window processing

    Args:
        window: Image window (16x16)
        feature_types: Encoded feature types
        feature_coords: Feature coordinates
        classifier_weights: AdaBoost weights [feature_idx, alpha, threshold, polarity]
        classifier_threshold: AdaBoost threshold
        total_alpha: Sum of all alphas
        threshold: Detection threshold

    Returns:
        confidence: Detection confidence, or -1 if no detection
    """
    # Compute integral image
    ii = compute_integral_image_numba(window.astype(np.float64))

    # Compute all feature responses
    responses = compute_haar_features_numba(ii, feature_types, feature_coords)

    # AdaBoost prediction (same as predict_proba)
    score = 0.0
    for i in prange(len(classifier_weights)):
        feature_idx = int(classifier_weights[i, 0])  # First column is feature index
        alpha = classifier_weights[i, 1]  # Second column is alpha
        thresh = classifier_weights[i, 2]  # Third column is threshold
        polarity = classifier_weights[i, 3]  # Fourth column is polarity
        
        # Same logic as WeakClassifier.predict: polarity * feature < polarity * threshold
        if polarity * responses[feature_idx] < polarity * thresh:
            score += alpha

    # Normalize score to [0, 1] range (same as original predict_proba / total_alpha)
    confidence = score / total_alpha

    # Check threshold
    if confidence >= threshold:
        return confidence
    else:
        return -1.0


def prepare_features_for_numba(features):
    """
    Convert HaarFeature objects to numba-compatible arrays

    Returns:
        feature_types: int array of feature types
        feature_coords: float array of coordinates
    """
    type_map = {'2h': 0, '2v': 1, '3h': 2, '3v': 3, '4d': 4}

    n_features = len(features)
    feature_types = np.zeros(n_features, dtype=np.int32)
    feature_coords = np.zeros((n_features, 4), dtype=np.int32)

    for i, feat in enumerate(features):
        feature_types[i] = type_map[feat.type]
        feature_coords[i] = [feat.x, feat.y, feat.width, feat.height]

    return feature_types, feature_coords


def prepare_classifier_for_numba(classifier):
    """
    Convert AdaBoost classifier to numba-compatible arrays

    Returns:
        weights: Array with [feature_idx, alpha, threshold, polarity] for each weak classifier
        total_alpha: Sum of all alphas
    """
    n_weak = len(classifier.weak_classifiers)
    weights = np.zeros((n_weak, 4), dtype=np.float64)

    for i, (weak_clf, alpha) in enumerate(zip(classifier.weak_classifiers, classifier.alphas)):
        weights[i, 0] = weak_clf.feature_idx
        weights[i, 1] = alpha
        weights[i, 2] = weak_clf.threshold
        weights[i, 3] = weak_clf.polarity

    total_alpha = sum(classifier.alphas)
    return weights, total_alpha


def extract_window(image, x, y, window_size):
    """
    Extract window from image at position (x, y)

    Args:
        image: Grayscale image
        x, y: Top-left corner
        window_size: Size of window (assumes square)

    Returns:
        window: Extracted patch (window_size x window_size)
        None if out of bounds
    """
    H, W = image.shape

    if x < 0 or y < 0 or x + window_size > W or y + window_size > H:
        return None

    return image[y:y+window_size, x:x+window_size]


def sliding_window(classifier, image, features, window_size=16,
                   step_size=2, threshold=0.5):
    """
    Apply classifier to image using sliding window

    Args:
        classifier: Trained classifier (AdaBoost or Cascade)
        image: Grayscale image (H, W)
        features: List of HaarFeature objects (same as training)
        window_size: Size of detection window (default 16x16)
        step_size: Stride for sliding window (default 2 pixels)
        threshold: Detection threshold (default 0.5)

    Returns:
        detections: List of Detection objects
    """
    H, W = image.shape
    detections = []

    print(f"Scanning image {W}x{H} with {window_size}x{window_size} window, step={step_size}")

    # Try to use numba-optimized version for AdaBoost
    if (NUMBA_AVAILABLE and hasattr(classifier, 'weak_classifiers') and
        hasattr(classifier, 'alphas') and len(classifier.weak_classifiers) > 0):

        print("✓ Using Numba-optimized sliding window (JIT compiled)...")
        print(f"  Numba available: {NUMBA_AVAILABLE}")
        print(f"  AdaBoost classifier: {len(classifier.weak_classifiers)} weak classifiers")

        # Prepare data for numba
        feature_types, feature_coords = prepare_features_for_numba(features)
        classifier_weights, total_alpha = prepare_classifier_for_numba(classifier)

        # Slide window across image
        for y in range(0, H - window_size + 1, step_size):
            for x in range(0, W - window_size + 1, step_size):
                # Extract window
                window = image[y:y+window_size, x:x+window_size]

                # Process window with numba
                confidence = process_window_numba(
                    window, feature_types, feature_coords, classifier_weights,
                    classifier.threshold, total_alpha, threshold
                )

                if confidence >= 0:
                    detections.append(Detection(x, y, window_size, window_size, confidence))

    else:
        # Fallback to original implementation
        print("⚠ Using standard sliding window implementation (fallback)")
        print(f"  Numba available: {NUMBA_AVAILABLE}")
        if hasattr(classifier, 'weak_classifiers'):
            print(f"  AdaBoost classifier: {len(classifier.weak_classifiers)} weak classifiers")
        else:
            print("  Cascade classifier detected")

        # Slide window across image
        for y in range(0, H - window_size + 1, step_size):
            for x in range(0, W - window_size + 1, step_size):
                # Extract window
                window = image[y:y+window_size, x:x+window_size]

                # Compute integral image
                ii = compute_ii_fast(window.astype(np.float64))

                # Compute feature responses
                responses = np.array([feat.compute(ii) for feat in features])

                # Predict
                if hasattr(classifier, 'predict_proba'):
                    # AdaBoost - get confidence score
                    score = classifier.predict_proba(responses.reshape(1, -1))[0]
                    total_alpha = sum(classifier.alphas)
                    # Normalize score to [0, 1] range
                    normalized_score = score / total_alpha

                    # Use normalized score for threshold comparison
                    if normalized_score >= threshold:
                        detections.append(Detection(x, y, window_size, window_size, normalized_score))
                else:
                    # Cascade - binary prediction
                    pred = classifier.predict(responses.reshape(1, -1))[0]
                    if pred == 1:
                        detections.append(Detection(x, y, window_size, window_size, 1.0))

    print(f"Found {len(detections)} detections")
    return detections


def sliding_window_parallel(classifier, image, features, window_size=16,
                            step_size=2, threshold=0.5, n_jobs=4):
    """
    Optimized parallel sliding window with batch processing

    Speed improvements:
    - Parallelizes window processing across cores
    - Batches feature computation
    - Reduces overhead from repeated function calls

    Args:
        classifier: Trained classifier (AdaBoost or Cascade)
        image: Grayscale image (H, W)
        features: List of HaarFeature objects
        window_size: Size of detection window (default 16x16)
        step_size: Stride for sliding window (default 2 pixels)
        threshold: Detection threshold (default 0.5)
        n_jobs: Number of parallel workers (default 4)

    Returns:
        detections: List of Detection objects
    """
    from joblib import Parallel, delayed

    H, W = image.shape
    print(f"Scanning image {W}x{H} (parallel, {n_jobs} cores)")

    # Generate all window positions
    positions = []
    for y in range(0, H - window_size + 1, step_size):
        for x in range(0, W - window_size + 1, step_size):
            positions.append((x, y))

    print(f"  Processing {len(positions)} windows...")

    # Try numba-optimized version
    if NUMBA_AVAILABLE and hasattr(classifier, 'weak_classifiers'):
        print("  ✓ Using Numba-optimized parallel processing...")
        print(f"    Numba available: {NUMBA_AVAILABLE}")
        print(f"    AdaBoost classifier: {len(classifier.weak_classifiers)} weak classifiers")

        # Prepare data for numba
        feature_types, feature_coords = prepare_features_for_numba(features)
        classifier_weights, total_alpha = prepare_classifier_for_numba(classifier)

        # Worker function using numba
        def process_window_numba_joblib(x, y):
            window = image[y:y+window_size, x:x+window_size]
            confidence = process_window_numba(
                window, feature_types, feature_coords, classifier_weights,
                classifier.threshold, total_alpha, threshold
            )
            if confidence >= 0:
                return Detection(x, y, window_size, window_size, confidence)
            return None

        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_window_numba_joblib)(x, y) for x, y in positions
        )

    else:
        print("  ⚠ Using standard parallel processing (fallback)")
        print(f"    Numba available: {NUMBA_AVAILABLE}")
        if hasattr(classifier, 'weak_classifiers'):
            print(f"    AdaBoost classifier: {len(classifier.weak_classifiers)} weak classifiers")
        else:
            print("    Cascade classifier detected")

        # Worker function to process a batch of windows
        def process_window(x, y):
            window = image[y:y+window_size, x:x+window_size]
            ii = compute_ii_fast(window.astype(np.float64))
            responses = np.array([feat.compute(ii) for feat in features])

            if hasattr(classifier, 'predict_proba'):
                score = classifier.predict_proba(responses.reshape(1, -1))[0]
                total_alpha = sum(classifier.alphas)
                confidence = score / total_alpha

                if confidence >= threshold:
                    return Detection(x, y, window_size, window_size, confidence)
            else:
                pred = classifier.predict(responses.reshape(1, -1))[0]
                if pred == 1:
                    return Detection(x, y, window_size, window_size, 1.0)

            return None

        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_window)(x, y) for x, y in positions
        )

    # Filter out None results
    detections = [det for det in results if det is not None]

    print(f"Found {len(detections)} detections")
    return detections


def create_image_pyramid(image, scale_factor=1.25, min_size=16):
    """
    Create multi-scale image pyramid

    Args:
        image: Input image
        scale_factor: Scaling between levels (default 1.25)
        min_size: Minimum dimension for smallest scale

    Returns:
        pyramid: List of (scale, scaled_image) tuples
    """
    pyramid = []
    current_scale = 1.0

    while True:
        # Compute scaled dimensions
        scaled_h = int(image.shape[0] / current_scale)
        scaled_w = int(image.shape[1] / current_scale)

        # Stop if too small
        if scaled_h < min_size or scaled_w < min_size:
            break

        # Resize image (simple nearest neighbor for now)
        # TODO: Use better interpolation (scipy.ndimage.zoom)
        scaled = image[::int(current_scale), ::int(current_scale)]

        pyramid.append((current_scale, scaled))
        current_scale *= scale_factor

    return pyramid


def multi_scale_detection(classifier, image, features, window_size=16,
                          step_size=2, scale_factor=1.25, threshold=0.5):
    """
    Detect faces at multiple scales using image pyramid

    Args:
        classifier: Trained classifier
        image: Grayscale image
        features: List of HaarFeature objects
        window_size: Detection window size
        step_size: Sliding window stride
        scale_factor: Pyramid scale factor
        threshold: Detection threshold

    Returns:
        all_detections: List of Detection objects (coordinates in original image space)
    """
    # Create pyramid
    pyramid = create_image_pyramid(image, scale_factor, window_size)
    print(f"Created pyramid with {len(pyramid)} scales")

    all_detections = []

    # Detect at each scale
    for scale, scaled_img in pyramid:
        print(f"\nScale {scale:.2f}: {scaled_img.shape}")

        # Use parallel sliding window for better performance
        detections = sliding_window_parallel(
            classifier, scaled_img, features,
            window_size, step_size, threshold, n_jobs=4
        )

        # Transform coordinates back to original image space
        for det in detections:
            det.x = int(det.x * scale)
            det.y = int(det.y * scale)
            det.width = int(det.width * scale)
            det.height = int(det.height * scale)
            all_detections.append(det)

    print(f"\nTotal detections across all scales: {len(all_detections)}")
    return all_detections


def multi_scale_detection_parallel(classifier, image, features, window_size=16,
                                   step_size=2, scale_factor=1.25,
                                   threshold=0.5, n_jobs=4):
    """
    Optimized multi-scale detection with parallelization

    Uses parallel sliding window for each scale level.
    Expected speedup: 3-4x on 4-core system

    Args:
        classifier: Trained classifier
        image: Grayscale image
        features: List of HaarFeature objects
        window_size: Detection window size
        step_size: Sliding window stride
        scale_factor: Pyramid scale factor
        threshold: Detection threshold
        n_jobs: Number of parallel workers

    Returns:
        all_detections: List of Detection objects (in original image space)
    """
    # Create pyramid
    pyramid = create_image_pyramid(image, scale_factor, window_size)
    print(f"Created pyramid with {len(pyramid)} scales (parallel mode)")

    all_detections = []

    # Detect at each scale
    for scale, scaled_img in pyramid:
        print(f"\nScale {scale:.2f}: {scaled_img.shape}")

        # Use parallel sliding window
        detections = sliding_window_parallel(
            classifier, scaled_img, features,
            window_size, step_size, threshold, n_jobs
        )

        # Transform coordinates back to original image space
        for det in detections:
            det.x = int(det.x * scale)
            det.y = int(det.y * scale)
            det.width = int(det.width * scale)
            det.height = int(det.height * scale)
            all_detections.append(det)

    print(f"\nTotal detections across all scales: {len(all_detections)}")
    return all_detections


def non_maximum_suppression(detections, overlap_threshold=0.3):
    """
    Remove overlapping detections using NMS

    Args:
        detections: List of Detection objects
        overlap_threshold: IoU threshold for suppression

    Returns:
        filtered: List of non-overlapping detections
    """
    if len(detections) == 0:
        return []

    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

    keep = []

    while len(detections) > 0:
        # Keep highest confidence detection
        best = detections.pop(0)
        keep.append(best)

        # Remove overlapping detections
        filtered = []
        for det in detections:
            iou = compute_iou(best, det)
            if iou < overlap_threshold:
                filtered.append(det)

        detections = filtered

    print(f"NMS: {len(keep)} detections after suppression")
    return keep


def compute_iou(det1, det2):
    """
    Compute Intersection over Union between two detections

    Args:
        det1, det2: Detection objects

    Returns:
        iou: Intersection over Union [0, 1]
    """
    # Intersection rectangle
    x1 = max(det1.x, det2.x)
    y1 = max(det1.y, det2.y)
    x2 = min(det1.x + det1.width, det2.x + det2.width)
    y2 = min(det1.y + det1.height, det2.y + det2.height)

    # Check for no overlap
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Intersection area
    intersection = (x2 - x1) * (y2 - y1)

    # Union area
    area1 = det1.width * det1.height
    area2 = det2.width * det2.height
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# Main execution for testing
if __name__ == '__main__':
    print("=" * 60)
    print("Sliding Window Detector")
    print("=" * 60)
    print("\nUsage:")
    print("1. Load trained classifier")
    print("2. Load image and convert to grayscale")
    print("3. Generate same features as training")
    print("4. Call multi_scale_detection()")
    print("5. Apply NMS to remove overlaps")
    print("\nSee detection notebook for complete example.")
    print("=" * 60)
