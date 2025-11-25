"""
AdaBoost Classifier for Viola-Jones Face Detector

Implements AdaBoost algorithm from Viola-Jones paper Table 1 (Appendix).
Selects T weak classifiers and combines them into a strong classifier.

Assignment: ELL715 Assignment 5 - Part 1 (40 marks)
AI Usage: Algorithm structure and docstrings assisted by Claude Code
"""

import numpy as np
import pickle
from pathlib import Path

try:
    from .weak_classifier import select_best_feature, WeakClassifier
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from weak_classifier import select_best_feature, WeakClassifier


class AdaBoostClassifier:
    """
    Strong classifier composed of T weak classifiers

    Formula from paper:
        h(x) = 1 if sum(alpha_t * h_t(x)) >= 0.5 * sum(alpha_t) else 0

    where:
        h_t(x) = weak classifier t
        alpha_t = log(1/beta_t) = confidence weight
    """

    def __init__(self):
        """Initialize AdaBoost classifier"""
        self.weak_classifiers = []
        self.alphas = []
        self.threshold = 0.0

    def predict(self, feature_response_matrix):
        """
        Predict labels for samples

        Args:
            feature_response_matrix: (N, M) feature responses

        Returns:
            predictions: (N,) binary labels
        """
        N = feature_response_matrix.shape[0]
        scores = np.zeros(N)

        # Weighted sum of weak classifier predictions
        for weak_clf, alpha in zip(self.weak_classifiers, self.alphas):
            feature_values = feature_response_matrix[:, weak_clf.feature_idx]
            predictions = weak_clf.predict(feature_values)
            scores += alpha * predictions

        # Threshold at half of total alpha
        total_alpha = sum(self.alphas)
        predictions = (scores >= self.threshold * total_alpha).astype(int)

        return predictions

    def predict_proba(self, feature_response_matrix):
        """
        Get confidence scores (before thresholding)

        Args:
            feature_response_matrix: (N, M) feature responses

        Returns:
            scores: (N,) confidence scores
        """
        N = feature_response_matrix.shape[0]
        scores = np.zeros(N)

        for weak_clf, alpha in zip(self.weak_classifiers, self.alphas):
            feature_values = feature_response_matrix[:, weak_clf.feature_idx]
            predictions = weak_clf.predict(feature_values)
            scores += alpha * predictions

        return scores

    def save(self, filepath):
        """Save trained classifier"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved classifier to {filepath}")

    @staticmethod
    def load(filepath):
        """Load trained classifier"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        return f"AdaBoostClassifier(n_weak={len(self.weak_classifiers)}, threshold={self.threshold:.3f})"


def train_adaboost(feature_response_matrix, labels, features, T=50,
                   validation_data=None, verbose=True):
    """
    Train AdaBoost classifier with optional validation tracking

    Algorithm from Viola-Jones paper Table 1:
    1. Initialize weights: w_1,i = 1/(2m) for negatives, 1/(2l) for positives
    2. For t = 1 to T:
       a) Normalize weights
       b) Select best weak classifier with minimum error
       c) Calculate beta_t = epsilon_t / (1 - epsilon_t)
       d) Update weights: w_{t+1,i} = w_{t,i} * beta_t^{1-e_i}
    3. Final classifier: h(x) = 1 if sum(alpha_t * h_t(x)) >= 0.5 * sum(alpha_t)

    Args:
        feature_response_matrix: (N, M) precomputed feature responses
        labels: (N,) ground truth labels (0/1)
        features: List of HaarFeature objects
        T: Number of weak classifiers to select
        validation_data: Tuple of (val_responses, val_labels) for tracking (optional)
        verbose: Print progress

    Returns:
        If validation_data is None: AdaBoostClassifier
        If validation_data provided: Tuple of (AdaBoostClassifier, history)
            where history = {'train_acc': [...], 'val_acc': [...]}
    """
    N = len(labels)
    m = np.sum(labels == 0)  # Number of negatives
    l = np.sum(labels == 1)  # Number of positives

    if verbose:
        print("=" * 60)
        print(f"Training AdaBoost with T={T} rounds")
        print(f"  Samples: {N} (negatives: {m}, positives: {l})")
        print("=" * 60)

    # Step 1: Initialize weights (Viola-Jones formula)
    weights = np.zeros(N, dtype=np.float64)
    weights[labels == 0] = 1.0 / (2 * m)
    weights[labels == 1] = 1.0 / (2 * l)

    if verbose:
        print(f"\nInitial weights sum: {weights.sum():.6f}")
        print(f"  Negative samples: {m} x {1.0/(2*m):.6f} = {m * 1.0/(2*m):.6f}")
        print(f"  Positive samples: {l} x {1.0/(2*l):.6f} = {l * 1.0/(2*l):.6f}")

    # Storage for selected classifiers
    weak_classifiers = []
    alphas = []

    # History tracking for validation
    history = None
    if validation_data is not None:
        val_responses, val_labels = validation_data
        history = {'train_acc': [], 'val_acc': []}
        if verbose:
            print(f"  Validation set: {len(val_labels)} samples")

    # Training loop
    for t in range(T):
        if verbose:
            print(f"\n--- Round {t+1}/{T} ---")

        # Step 2a: Normalize weights
        weights = weights / np.sum(weights)

        if verbose and t < 3:  # Show first few rounds
            print(f"  Normalized weights sum: {weights.sum():.6f}")

        # Step 2b: Select best weak classifier
        weak_clf = select_best_feature(
            feature_response_matrix,
            labels,
            weights,
            features
        )

        # Calculate epsilon (weighted error)
        epsilon = weak_clf.error

        # Prevent division by zero
        if epsilon >= 0.5:
            if verbose:
                print(f"  Warning: epsilon >= 0.5 ({epsilon:.4f}), stopping early")
            break

        if epsilon < 1e-10:
            epsilon = 1e-10

        # Step 2c: Calculate beta
        beta = epsilon / (1.0 - epsilon)

        # Calculate alpha for final classifier
        alpha = np.log(1.0 / beta)

        if verbose:
            print(f"  Epsilon: {epsilon:.6f}")
            print(f"  Beta: {beta:.6f}")
            print(f"  Alpha: {alpha:.6f}")

        # Step 2d: Update weights
        # Get predictions for this classifier
        feature_values = feature_response_matrix[:, weak_clf.feature_idx]
        predictions = weak_clf.predict(feature_values)

        # e_i = 0 if correct, 1 if incorrect
        errors = (predictions != labels).astype(int)

        # w_{t+1,i} = w_{t,i} * beta^{1-e_i}
        # If correct (e_i=0): multiply by beta (reduce weight)
        # If incorrect (e_i=1): multiply by 1 (keep weight)
        weights = weights * (beta ** (1 - errors))

        # Store classifier
        weak_classifiers.append(weak_clf)
        alphas.append(alpha)

        # Track accuracy history (every round if validation, every 10 rounds otherwise)
        should_evaluate = (validation_data is not None) or ((t + 1) % 10 == 0)

        if should_evaluate:
            # Temporary classifier to evaluate
            temp_clf = AdaBoostClassifier()
            temp_clf.weak_classifiers = weak_classifiers
            temp_clf.alphas = alphas
            temp_clf.threshold = 0.5

            train_preds = temp_clf.predict(feature_response_matrix)
            train_acc = np.mean(train_preds == labels)

            if validation_data is not None:
                # Evaluate on validation set
                val_preds = temp_clf.predict(val_responses)
                val_acc = np.mean(val_preds == val_labels)

                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)

                if verbose and (t + 1) % 10 == 0:
                    print(f"  Train acc: {train_acc:.2%}  |  Val acc: {val_acc:.2%}")
            else:
                if verbose:
                    print(f"  Training accuracy: {train_acc:.2%}")

    # Create final classifier
    classifier = AdaBoostClassifier()
    classifier.weak_classifiers = weak_classifiers
    classifier.alphas = alphas
    classifier.threshold = 0.5  # Default threshold

    if verbose:
        print("\n" + "=" * 60)
        print("AdaBoost Training Complete!")
        print(f"  Selected {len(weak_classifiers)} weak classifiers")
        print(f"  Total alpha: {sum(alphas):.4f}")

        if validation_data is not None:
            best_val_idx = np.argmax(history['val_acc'])
            best_val_acc = history['val_acc'][best_val_idx]
            print(f"  Best validation accuracy: {best_val_acc:.2%} at round {best_val_idx+1}")

        print("=" * 60)

    # Return classifier with history if validation was used
    if validation_data is not None:
        return classifier, history
    else:
        return classifier


def evaluate_classifier(classifier, feature_response_matrix, labels, verbose=True):
    """
    Evaluate classifier performance

    Args:
        classifier: Trained AdaBoostClassifier
        feature_response_matrix: (N, M) feature responses
        labels: (N,) ground truth
        verbose: Print metrics

    Returns:
        metrics: Dict with accuracy, precision, recall, f1
    """
    predictions = classifier.predict(feature_response_matrix)

    # Calculate metrics
    accuracy = np.mean(predictions == labels)

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Evaluation Metrics")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:5d}  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  TN: {tn:5d}")
        print("=" * 60)

    return metrics


# Main execution for testing
if __name__ == '__main__':
    print("AdaBoost implementation ready!")
    print("\nTo train AdaBoost:")
    print("1. Load dataset patches")
    print("2. Generate Haar features")
    print("3. Compute feature response matrix")
    print("4. Call train_adaboost()")
    print("\nSee training notebook for complete example.")
