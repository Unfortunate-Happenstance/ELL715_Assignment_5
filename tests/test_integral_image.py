"""
Unit tests for integral image implementation

Assignment: ELL715 Assignment 5 - Part 1
AI Usage: Test structure assisted by Claude Code
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.integral_image import (
    compute_integral_image,
    compute_integral_image_vectorized,
    rectangle_sum,
    compute_second_integral_image,
    compute_mean_and_variance,
    verify_integral_image
)


class TestIntegralImage:
    """Test suite for integral image functions"""

    def test_simple_case(self):
        """Test with simple 3×3 matrix"""
        image = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float64)

        ii = compute_integral_image(image)

        # Check full image sum
        assert rectangle_sum(ii, 0, 0, 3, 3) == 45

        # Check top-left 2×2
        assert rectangle_sum(ii, 0, 0, 2, 2) == 12

        # Check bottom-right 2×2
        assert rectangle_sum(ii, 1, 1, 2, 2) == 28

    def test_vectorized_matches_loop(self):
        """Test that vectorized version matches loop version"""
        image = np.random.rand(16, 16) * 255

        ii_loop = compute_integral_image(image)
        ii_vec = compute_integral_image_vectorized(image)

        assert np.allclose(ii_loop, ii_vec)

    def test_rectangle_sum_full_image(self):
        """Test that full image sum equals total"""
        image = np.random.randint(0, 256, (16, 16))
        ii = compute_integral_image_vectorized(image)

        full_sum = rectangle_sum(ii, 0, 0, 16, 16)
        expected = image.sum()

        assert np.isclose(full_sum, expected)

    def test_rectangle_sum_single_pixel(self):
        """Test single pixel rectangles"""
        image = np.random.rand(8, 8) * 100
        ii = compute_integral_image_vectorized(image)

        for y in range(8):
            for x in range(8):
                rect_sum = rectangle_sum(ii, x, y, 1, 1)
                assert np.isclose(rect_sum, image[y, x])

    def test_random_rectangles(self):
        """Test random rectangles against brute force"""
        image = np.random.rand(16, 16) * 100
        ii = compute_integral_image_vectorized(image)

        assert verify_integral_image(image, ii)

    def test_zero_image(self):
        """Test with all-zero image"""
        image = np.zeros((10, 10))
        ii = compute_integral_image_vectorized(image)

        # All rectangle sums should be zero
        assert rectangle_sum(ii, 0, 0, 10, 10) == 0
        assert rectangle_sum(ii, 2, 3, 5, 4) == 0

    def test_ones_image(self):
        """Test with all-ones image"""
        image = np.ones((8, 8))
        ii = compute_integral_image_vectorized(image)

        # Rectangle sum should equal area
        assert rectangle_sum(ii, 0, 0, 8, 8) == 64
        assert rectangle_sum(ii, 2, 3, 4, 3) == 12

    def test_second_integral_image(self):
        """Test squared integral image"""
        image = np.array([[1, 2], [3, 4]], dtype=np.float64)
        ii_sq = compute_second_integral_image(image)

        # Sum of squares: 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
        sum_sq = rectangle_sum(ii_sq, 0, 0, 2, 2)
        assert np.isclose(sum_sq, 30)

    def test_mean_variance_computation(self):
        """Test mean and variance computation"""
        image = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float64)

        ii = compute_integral_image_vectorized(image)
        ii_sq = compute_second_integral_image(image)

        mean, var = compute_mean_and_variance(ii, ii_sq, 0, 0, 3, 3)

        expected_mean = image.mean()
        expected_var = image.var()

        assert np.isclose(mean, expected_mean)
        assert np.isclose(var, expected_var)

    def test_boundary_conditions(self):
        """Test boundary conditions (s(x,-1)=0, ii(-1,y)=0)"""
        image = np.random.rand(5, 5)
        ii = compute_integral_image(image)

        # First row and column should be zero
        assert np.all(ii[0, :] == 0)
        assert np.all(ii[:, 0] == 0)

    def test_dimensions(self):
        """Test that integral image has correct dimensions"""
        for h in [5, 10, 16]:
            for w in [5, 10, 16]:
                image = np.random.rand(h, w)
                ii = compute_integral_image_vectorized(image)

                assert ii.shape == (h + 1, w + 1)


def test_run_all():
    """Run all tests"""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    test_run_all()
