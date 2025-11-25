"""
Integral Image Implementation for Viola-Jones Face Detector

Implements fast rectangle sum computation using integral image representation.
Based on Viola-Jones paper Section 2.1.

Assignment: ELL715 Assignment 5 - Part 1 (20 marks)
AI Usage: Algorithm structure and docstrings assisted by Claude Code
"""

import numpy as np


def compute_integral_image(image):
    """
    Compute integral image for fast rectangle sum queries.

    The integral image at location (x, y) contains the sum of pixels
    above and to the left of (x, y), inclusive.

    Formula from Viola-Jones paper Section 2.1:
        s(x,y) = s(x,y-1) + i(x,y)         (cumulative row sum)
        ii(x,y) = ii(x-1,y) + s(x,y)       (integral image)

    Boundary conditions: s(x,-1) = 0, ii(-1,y) = 0

    Args:
        image: 2D numpy array (H, W) with pixel intensities

    Returns:
        ii: Integral image of size (H+1, W+1)
            - Extra row and column for easier indexing
            - ii[0,:] = 0 and ii[:,0] = 0 (boundaries)
            - ii[y,x] = sum of all pixels in rectangle from (0,0) to (x-1,y-1)

    Time Complexity: O(H × W) for computation
    Query Complexity: O(1) for any rectangle sum
    """
    H, W = image.shape

    # Initialize integral image and cumulative row sum with extra borders
    # Dimensions: (H+1) × (W+1) for 1-indexed access
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    s = np.zeros((H + 1, W + 1), dtype=np.float64)

    # Compute integral image in one pass
    for y in range(1, H + 1):
        for x in range(1, W + 1):
            # s(x,y) = s(x,y-1) + i(x,y)
            s[y, x] = s[y, x - 1] + image[y - 1, x - 1]

            # ii(x,y) = ii(x-1,y) + s(x,y)
            ii[y, x] = ii[y - 1, x] + s[y, x]

    return ii


def compute_integral_image_vectorized(image):
    """
    Vectorized version of integral image computation (faster for large images).

    Uses numpy cumsum for optimized performance.

    Args:
        image: 2D numpy array (H, W)

    Returns:
        ii: Integral image of size (H+1, W+1)
    """
    H, W = image.shape

    # Pad image with zeros on top and left
    padded = np.pad(image.astype(np.float64), ((1, 0), (1, 0)), mode='constant')

    # Compute cumulative sum along both axes
    ii = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    return ii


def rectangle_sum(ii, x, y, width, height):
    """
    Compute sum of pixels in rectangle using integral image.

    Rectangle is defined by top-left corner (x, y) and dimensions (width, height).
    Uses 4 array references for O(1) computation.

    Formula:
        sum = ii[y+height, x+width] - ii[y, x+width]
              - ii[y+height, x] + ii[y, x]

    Explanation:
        ii[y2,x2] = sum of rectangle (0,0) to (x2,y2)
        ii[y1,x2] = sum of rectangle (0,0) to (x2,y1)
        ii[y2,x1] = sum of rectangle (0,0) to (x1,y2)
        ii[y1,x1] = sum of rectangle (0,0) to (x1,y1)

        sum = ii[y2,x2] - ii[y1,x2] - ii[y2,x1] + ii[y1,x1]

    Args:
        ii: Integral image (H+1, W+1)
        x: Left x-coordinate of rectangle (0-indexed)
        y: Top y-coordinate of rectangle (0-indexed)
        width: Width of rectangle
        height: Height of rectangle

    Returns:
        Sum of pixels in the rectangle

    Time Complexity: O(1)
    """
    x2 = x + width
    y2 = y + height

    # Use 4 array references (corners of rectangle)
    return ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x]


def compute_second_integral_image(image):
    """
    Compute integral image of squared pixel values.

    Used for variance normalization as described in Viola-Jones paper Section 5.

    Variance formula: σ² = E[x²] - E[x]²
    where E[x²] can be computed using integral image of squared pixels.

    Args:
        image: 2D numpy array (H, W)

    Returns:
        ii_squared: Integral image of squared pixels (H+1, W+1)
    """
    return compute_integral_image_vectorized(image.astype(np.float64) ** 2)


def compute_mean_and_variance(ii, ii_squared, x, y, width, height):
    """
    Compute mean and variance of pixels in rectangle using integral images.

    Args:
        ii: Integral image of pixels
        ii_squared: Integral image of squared pixels
        x, y: Top-left corner
        width, height: Rectangle dimensions

    Returns:
        mean: Mean pixel value
        variance: Variance of pixel values
    """
    n_pixels = width * height

    # Mean: E[x]
    sum_pixels = rectangle_sum(ii, x, y, width, height)
    mean = sum_pixels / n_pixels

    # Variance: E[x²] - E[x]²
    sum_squared = rectangle_sum(ii_squared, x, y, width, height)
    variance = (sum_squared / n_pixels) - (mean ** 2)

    return mean, variance


def normalize_patch_variance(patch):
    """
    Variance normalize a patch (as described in Viola-Jones paper).

    Normalization: (patch - mean) / std_dev

    Args:
        patch: 2D numpy array

    Returns:
        Normalized patch with mean=0, std=1
    """
    mean = patch.mean()
    std = patch.std()

    if std < 1e-10:  # Avoid division by zero
        return np.zeros_like(patch, dtype=np.float64)

    return (patch - mean) / std


# Convenience functions for common use cases
def compute_ii_fast(image):
    """
    Convenience function for fast integral image computation.

    Automatically uses vectorized version for better performance.

    Args:
        image: 2D numpy array

    Returns:
        ii: Integral image
    """
    return compute_integral_image_vectorized(image)


def verify_integral_image(image, ii):
    """
    Verify integral image correctness by comparing with brute force.

    Used for testing and validation.

    Args:
        image: Original image
        ii: Computed integral image

    Returns:
        True if correct, False otherwise
    """
    H, W = image.shape

    # Test random rectangles
    n_tests = 100
    for _ in range(n_tests):
        # Random rectangle
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        width = np.random.randint(1, W - x + 1)
        height = np.random.randint(1, H - y + 1)

        # Compute using integral image
        sum_ii = rectangle_sum(ii, x, y, width, height)

        # Compute using brute force
        sum_brute = image[y:y+height, x:x+width].sum()

        # Compare
        if not np.isclose(sum_ii, sum_brute, rtol=1e-5):
            print(f"Mismatch at ({x},{y}) {width}×{height}: {sum_ii} vs {sum_brute}")
            return False

    return True


# Main execution for testing
if __name__ == '__main__':
    # Test integral image
    print("Testing Integral Image Implementation...")

    # Simple test case
    test_image = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]], dtype=np.float64)

    print("\nTest Image:")
    print(test_image)

    # Compute integral image
    ii = compute_integral_image(test_image)

    print("\nIntegral Image:")
    print(ii)

    # Test rectangle sum
    print("\nTesting rectangle_sum():")
    # Full image (should be 45)
    full_sum = rectangle_sum(ii, 0, 0, 3, 3)
    print(f"Full image sum: {full_sum} (expected: 45)")

    # Top-left 2×2 (should be 1+2+4+5 = 12)
    tl_sum = rectangle_sum(ii, 0, 0, 2, 2)
    print(f"Top-left 2×2 sum: {tl_sum} (expected: 12)")

    # Bottom-right 2×2 (should be 5+6+8+9 = 28)
    br_sum = rectangle_sum(ii, 1, 1, 2, 2)
    print(f"Bottom-right 2×2 sum: {br_sum} (expected: 28)")

    # Verify with random image
    print("\n\nVerifying with random 16×16 image...")
    random_image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
    ii_random = compute_integral_image_vectorized(random_image)

    if verify_integral_image(random_image, ii_random):
        print("✓ Integral image implementation is CORRECT!")
    else:
        print("✗ Integral image implementation has ERRORS!")

    # Test variance computation
    print("\n\nTesting variance computation...")
    patch = np.random.rand(16, 16) * 100
    ii_patch = compute_ii_fast(patch)
    ii_sq_patch = compute_second_integral_image(patch)

    mean, var = compute_mean_and_variance(ii_patch, ii_sq_patch, 0, 0, 16, 16)
    print(f"Computed mean: {mean:.4f}, variance: {var:.4f}")
    print(f"NumPy mean: {patch.mean():.4f}, variance: {patch.var():.4f}")
    print(f"Match: {np.isclose(mean, patch.mean()) and np.isclose(var, patch.var())}")

    print("\n" + "=" * 60)
    print("Integral Image module ready for Haar feature extraction!")
    print("=" * 60)
