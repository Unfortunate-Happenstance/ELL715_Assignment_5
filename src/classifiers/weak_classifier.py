"""
Weak Classifier for AdaBoost

Single-feature threshold classifier as described in Viola-Jones paper.
Each weak classifier uses one Haar feature and a threshold.

Assignment: ELL715 Assignment 5 - Part 1 (part of 40 marks for AdaBoost)
AI Usage: Algorithm structure and docstrings assisted by Claude Code
"""

import numpy as np


class WeakClassifier:
    """
    Threshold-based weak classifier using single Haar feature

    Formula from paper Section 3:
        h(x) = 1 if p*f(x) < p*theta else 0

    where:
        f(x) = feature value
        theta = threshold
        p = polarity (+1 or -1) indicating inequality direction
    """

    def __init__(self, feature_idx, threshold, polarity, error):
        """
        Initialize weak classifier

        Args:
            feature_idx: Index of Haar feature to use
            threshold: Decision threshold
            polarity: +1 or -1 (direction of inequality)
            error: Weighted error on training data
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity
        self.error = error

    def predict(self, feature_values):
        """
        Predict class labels for given feature values

        Args:
            feature_values: Array of feature responses (N,)

        Returns:
            predictions: Binary array (N,) with 0/1 labels
        """
        # Apply threshold with polarity
        predictions = (self.polarity * feature_values < self.polarity * self.threshold).astype(int)
        return predictions

    def __repr__(self):
        return f"WeakClassifier(feature={self.feature_idx}, thresh={self.threshold:.2f}, pol={self.polarity}, err={self.error:.4f})"


def find_best_threshold(feature_responses, labels, weights):
    """
    Find optimal threshold for a single feature

    Tries all possible thresholds (sorted unique feature values)
    and selects the one with minimum weighted error.

    Args:
        feature_responses: Feature values for all samples (N,)
        labels: Ground truth labels (N,) with 0/1
        weights: Sample weights (N,) - must sum to 1

    Returns:
        best_threshold: Optimal threshold value
        best_polarity: +1 or -1
        best_error: Minimum weighted error
    """
    # Get sorted unique feature values as candidate thresholds
    sorted_indices = np.argsort(feature_responses)
    sorted_features = feature_responses[sorted_indices]
    sorted_labels = labels[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Total positive and negative weights
    total_pos = np.sum(sorted_weights[sorted_labels == 1])
    total_neg = np.sum(sorted_weights[sorted_labels == 0])

    # Initialize cumulative weights
    cum_pos = 0.0
    cum_neg = 0.0

    best_error = float('inf')
    best_threshold = 0.0
    best_polarity = 1

    # Try each threshold
    for i in range(len(sorted_features)):
        # Update cumulative weights
        if sorted_labels[i] == 1:
            cum_pos += sorted_weights[i]
        else:
            cum_neg += sorted_weights[i]

        # Skip if next feature value is the same
        if i < len(sorted_features) - 1 and sorted_features[i] == sorted_features[i + 1]:
            continue

        # Threshold between current and next feature
        if i < len(sorted_features) - 1:
            threshold = (sorted_features[i] + sorted_features[i + 1]) / 2
        else:
            threshold = sorted_features[i] + 1

        # Error for polarity = +1 (predict 1 if feature < threshold)
        # Misclassifies: negatives below threshold + positives above threshold
        error_pos = cum_neg + (total_pos - cum_pos)

        # Error for polarity = -1 (predict 1 if feature > threshold)
        # Misclassifies: positives below threshold + negatives above threshold
        error_neg = cum_pos + (total_neg - cum_neg)

        # Choose best polarity
        if error_pos < error_neg:
            error = error_pos
            polarity = 1
        else:
            error = error_neg
            polarity = -1

        # Update best
        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_polarity = polarity

    return best_threshold, best_polarity, best_error


def train_weak_classifier(feature_responses, labels, weights):
    """
    Train weak classifier on single feature

    Finds optimal threshold and polarity to minimize weighted error.

    Args:
        feature_responses: Feature values (N,)
        labels: Ground truth (N,) with 0/1
        weights: Sample weights (N,) - must sum to 1

    Returns:
        WeakClassifier object
    """
    threshold, polarity, error = find_best_threshold(feature_responses, labels, weights)

    # Note: feature_idx set to -1 here, should be set by caller
    return WeakClassifier(
        feature_idx=-1,
        threshold=threshold,
        polarity=polarity,
        error=error
    )


def select_best_feature(feature_response_matrix, labels, weights, features):
    """
    Select best feature from pool by training weak classifier on each

    Args:
        feature_response_matrix: (N_samples, N_features) response matrix
        labels: Ground truth (N,)
        weights: Sample weights (N,) - must sum to 1
        features: List of HaarFeature objects (for logging)

    Returns:
        best_classifier: WeakClassifier with lowest error
    """
    N_samples, N_features = feature_response_matrix.shape

    best_error = float('inf')
    best_classifier = None

    print(f"  Searching {N_features} features for best weak classifier...")

    # Try each feature
    for feature_idx in range(N_features):
        if (feature_idx + 1) % 1000 == 0:
            print(f"    Evaluated {feature_idx + 1}/{N_features} features...")

        # Get feature responses
        feature_values = feature_response_matrix[:, feature_idx]

        # Train weak classifier
        classifier = train_weak_classifier(feature_values, labels, weights)
        classifier.feature_idx = feature_idx

        # Update best
        if classifier.error < best_error:
            best_error = classifier.error
            best_classifier = classifier

    print(f"  Best feature: {best_classifier.feature_idx}, error: {best_error:.4f}")

    return best_classifier


# Main execution for testing
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Weak Classifier")
    print("=" * 60)

    # Simple linearly separable data
    np.random.seed(42)
    N = 100

    # Feature values: class 0 < 50, class 1 > 50
    feature_responses = np.concatenate([
        np.random.randn(N//2) * 10 + 30,  # Class 0
        np.random.randn(N//2) * 10 + 70   # Class 1
    ])
    labels = np.array([0] * (N//2) + [1] * (N//2))

    # Uniform weights
    weights = np.ones(N) / N

    print("\nFinding optimal threshold...")
    threshold, polarity, error = find_best_threshold(feature_responses, labels, weights)

    print(f"  Threshold: {threshold:.2f}")
    print(f"  Polarity: {polarity}")
    print(f"  Error: {error:.4f}")

    # Train weak classifier
    classifier = train_weak_classifier(feature_responses, labels, weights)
    classifier.feature_idx = 0

    print(f"\n{classifier}")

    # Test predictions
    predictions = classifier.predict(feature_responses)
    accuracy = np.mean(predictions == labels)
    print(f"\nAccuracy: {accuracy:.2%}")

    print("\n" + "=" * 60)
    print("Weak Classifier ready for AdaBoost!")
    print("=" * 60)
