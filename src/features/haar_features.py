"""
Haar-like Features for Viola-Jones Face Detector

Implements rectangle features as described in Viola-Jones paper Section 2.
Features detect edges, lines, and other simple patterns.

Assignment: ELL715 Assignment 5 - Part 1 (20 marks)
AI Usage: Algorithm structure and docstrings assisted by Claude Code
"""

import numpy as np

# Handle imports for both module and standalone execution
try:
    from .integral_image import rectangle_sum
except ImportError:
    from integral_image import rectangle_sum


class HaarFeature:
    """
    Represents a single Haar-like rectangle feature.

    Feature types:
    - '2h': Two horizontal rectangles (top vs bottom)
    - '2v': Two vertical rectangles (left vs right)
    - '3h': Three horizontal rectangles (top-middle-bottom)
    - '3v': Three vertical rectangles (left-middle-right)
    - '4d': Four diagonal rectangles (checkerboard)

    Feature value = sum(white rectangles) - sum(black rectangles)
    """

    def __init__(self, feature_type, x, y, width, height):
        """
        Initialize Haar feature

        Args:
            feature_type: Type of feature ('2h', '2v', '3h', '3v', '4d')
            x, y: Top-left corner position
            width, height: Overall feature dimensions
        """
        self.type = feature_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute(self, integral_image):
        """
        Compute feature value using integral image

        Args:
            integral_image: Precomputed integral image

        Returns:
            Feature response (float)
        """
        if self.type == '2h':
            return self._compute_2h(integral_image)
        elif self.type == '2v':
            return self._compute_2v(integral_image)
        elif self.type == '3h':
            return self._compute_3h(integral_image)
        elif self.type == '3v':
            return self._compute_3v(integral_image)
        elif self.type == '4d':
            return self._compute_4d(integral_image)
        else:
            raise ValueError(f"Unknown feature type: {self.type}")

    def _compute_2h(self, ii):
        """Two horizontal rectangles (top white, bottom black)"""
        h_half = self.height // 2

        # Top rectangle (white)
        top = rectangle_sum(ii, self.x, self.y, self.width, h_half)

        # Bottom rectangle (black)
        bottom = rectangle_sum(ii, self.x, self.y + h_half, self.width, h_half)

        return bottom - top  # Positive if bottom brighter

    def _compute_2v(self, ii):
        """Two vertical rectangles (left white, right black)"""
        w_half = self.width // 2

        # Left rectangle (white)
        left = rectangle_sum(ii, self.x, self.y, w_half, self.height)

        # Right rectangle (black)
        right = rectangle_sum(ii, self.x + w_half, self.y, w_half, self.height)

        return right - left  # Positive if right brighter

    def _compute_3h(self, ii):
        """Three horizontal rectangles (top/bottom white, middle black)"""
        h_third = self.height // 3

        # Top rectangle (white)
        top = rectangle_sum(ii, self.x, self.y, self.width, h_third)

        # Middle rectangle (black)
        middle = rectangle_sum(ii, self.x, self.y + h_third, self.width, h_third)

        # Bottom rectangle (white)
        bottom = rectangle_sum(ii, self.x, self.y + 2*h_third, self.width, h_third)

        return (top + bottom) - middle  # Positive if edges brighter than center

    def _compute_3v(self, ii):
        """Three vertical rectangles (left/right white, middle black)"""
        w_third = self.width // 3

        # Left rectangle (white)
        left = rectangle_sum(ii, self.x, self.y, w_third, self.height)

        # Middle rectangle (black)
        middle = rectangle_sum(ii, self.x + w_third, self.y, w_third, self.height)

        # Right rectangle (white)
        right = rectangle_sum(ii, self.x + 2*w_third, self.y, w_third, self.height)

        return (left + right) - middle

    def _compute_4d(self, ii):
        """Four diagonal rectangles (checkerboard pattern)"""
        w_half = self.width // 2
        h_half = self.height // 2

        # Top-left (white)
        tl = rectangle_sum(ii, self.x, self.y, w_half, h_half)

        # Top-right (black)
        tr = rectangle_sum(ii, self.x + w_half, self.y, w_half, h_half)

        # Bottom-left (black)
        bl = rectangle_sum(ii, self.x, self.y + h_half, w_half, h_half)

        # Bottom-right (white)
        br = rectangle_sum(ii, self.x + w_half, self.y + h_half, w_half, h_half)

        return (tl + br) - (tr + bl)  # Diagonal difference

    def __repr__(self):
        return f"HaarFeature(type={self.type}, pos=({self.x},{self.y}), size={self.width}x{self.height})"


def generate_haar_features(window_size=16, max_features=None):
    """
    Generate all possible Haar features for given window size

    For 16×16 window, generates ~45k-60k features total.
    For V1 (simplified), can limit to first 10k features.

    Args:
        window_size: Size of detection window (default 16×16)
        max_features: Maximum number of features to generate (None = all)

    Returns:
        List of HaarFeature objects
    """
    features = []

    print(f"Generating Haar features for {window_size}×{window_size} window...")

    # 2-rectangle horizontal features
    print("  Generating 2-rectangle horizontal features...")
    for x in range(window_size):
        for y in range(window_size):
            for w in range(1, window_size - x + 1):
                for h in range(2, window_size - y + 1, 2):  # Even heights only
                    features.append(HaarFeature('2h', x, y, w, h))

                    if max_features and len(features) >= max_features:
                        return features

    # 2-rectangle vertical features
    print("  Generating 2-rectangle vertical features...")
    for x in range(window_size):
        for y in range(window_size):
            for w in range(2, window_size - x + 1, 2):  # Even widths only
                for h in range(1, window_size - y + 1):
                    features.append(HaarFeature('2v', x, y, w, h))

                    if max_features and len(features) >= max_features:
                        return features

    # 3-rectangle horizontal features
    print("  Generating 3-rectangle horizontal features...")
    for x in range(window_size):
        for y in range(window_size):
            for w in range(1, window_size - x + 1):
                for h in range(3, window_size - y + 1, 3):  # Multiple of 3
                    features.append(HaarFeature('3h', x, y, w, h))

                    if max_features and len(features) >= max_features:
                        return features

    # 3-rectangle vertical features
    print("  Generating 3-rectangle vertical features...")
    for x in range(window_size):
        for y in range(window_size):
            for w in range(3, window_size - x + 1, 3):  # Multiple of 3
                for h in range(1, window_size - y + 1):
                    features.append(HaarFeature('3v', x, y, w, h))

                    if max_features and len(features) >= max_features:
                        return features

    # 4-rectangle diagonal features
    print("  Generating 4-rectangle diagonal features...")
    for x in range(window_size):
        for y in range(window_size):
            for w in range(2, window_size - x + 1, 2):  # Even widths
                for h in range(2, window_size - y + 1, 2):  # Even heights
                    features.append(HaarFeature('4d', x, y, w, h))

                    if max_features and len(features) >= max_features:
                        return features

    print(f"  Generated {len(features)} features total")
    return features


def compute_feature_responses(features, patches, integral_images=None):
    """
    Compute feature responses for all patches

    Pre-computes integral images if not provided, then evaluates
    all features on all patches to create response matrix.

    Args:
        features: List of HaarFeature objects
        patches: Array of patches (N, H, W)
        integral_images: Pre-computed integral images (optional)

    Returns:
        response_matrix: (N_patches, N_features) array of feature values
    """
    try:
        from .integral_image import compute_ii_fast
    except ImportError:
        from integral_image import compute_ii_fast

    N = len(patches)
    M = len(features)

    print(f"Computing feature responses for {N} patches × {M} features...")

    # Pre-compute integral images if not provided
    if integral_images is None:
        print("  Computing integral images...")
        integral_images = []
        for patch in patches:
            ii = compute_ii_fast(patch.astype(np.float64))
            integral_images.append(ii)

    # Compute response matrix
    print("  Computing feature responses...")
    responses = np.zeros((N, M), dtype=np.float32)

    for i, ii in enumerate(integral_images):
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1}/{N} patches...")

        for j, feature in enumerate(features):
            responses[i, j] = feature.compute(ii)

    print(f"  Response matrix shape: {responses.shape}")
    return responses


def visualize_feature(feature, window_size=16):
    """
    Create visualization of Haar feature as image

    Args:
        feature: HaarFeature object
        window_size: Size of window

    Returns:
        Image array showing feature pattern (white=+1, black=-1, gray=0)
    """
    img = np.zeros((window_size, window_size), dtype=np.float32)

    x, y, w, h = feature.x, feature.y, feature.width, feature.height

    if feature.type == '2h':
        h_half = h // 2
        img[y:y+h_half, x:x+w] = 1  # Top white
        img[y+h_half:y+h, x:x+w] = -1  # Bottom black

    elif feature.type == '2v':
        w_half = w // 2
        img[y:y+h, x:x+w_half] = 1  # Left white
        img[y:y+h, x+w_half:x+w] = -1  # Right black

    elif feature.type == '3h':
        h_third = h // 3
        img[y:y+h_third, x:x+w] = 1  # Top white
        img[y+h_third:y+2*h_third, x:x+w] = -1  # Middle black
        img[y+2*h_third:y+h, x:x+w] = 1  # Bottom white

    elif feature.type == '3v':
        w_third = w // 3
        img[y:y+h, x:x+w_third] = 1  # Left white
        img[y:y+h, x+w_third:x+2*w_third] = -1  # Middle black
        img[y:y+h, x+2*w_third:x+w] = 1  # Right white

    elif feature.type == '4d':
        w_half = w // 2
        h_half = h // 2
        img[y:y+h_half, x:x+w_half] = 1  # Top-left white
        img[y:y+h_half, x+w_half:x+w] = -1  # Top-right black
        img[y+h_half:y+h, x:x+w_half] = -1  # Bottom-left black
        img[y+h_half:y+h, x+w_half:x+w] = 1  # Bottom-right white

    return img


# Main execution for testing
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Haar Features Implementation")
    print("=" * 60)

    # Generate features for 16×16 window
    # V1: Limit to 10k features
    features = generate_haar_features(window_size=16, max_features=10000)

    print(f"\n[OK] Generated {len(features)} Haar features")

    # Show first few features
    print("\nFirst 5 features:")
    for i, feat in enumerate(features[:5]):
        print(f"  {i+1}. {feat}")

    # Test feature computation
    print("\nTesting feature computation...")
    test_patch = np.random.rand(16, 16) * 100

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from integral_image import compute_ii_fast
    ii = compute_ii_fast(test_patch)

    response = features[0].compute(ii)
    print(f"  Feature 0 response: {response:.2f}")

    print("\n" + "=" * 60)
    print("Haar Features module ready for AdaBoost training!")
    print("=" * 60)
