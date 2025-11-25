"""
Cascade of Classifiers for Viola-Jones Face Detector

Implements cascaded AdaBoost stages as described in Viola-Jones paper Section 4.
Each stage rejects non-faces quickly while passing faces to next stage.

Assignment: ELL715 Assignment 5 - Part 1 (20 marks)
AI Usage: Algorithm structure and docstrings assisted by Claude Code
"""

import numpy as np
import pickle
from pathlib import Path

try:
    from .adaboost import train_adaboost, AdaBoostClassifier
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from adaboost import train_adaboost, AdaBoostClassifier


class CascadeStage:
    """
    Single stage in cascade

    Each stage is an AdaBoost classifier with adjusted threshold
    to achieve target false positive and detection rates.
    """

    def __init__(self, classifier, threshold, stage_num):
        """
        Initialize cascade stage

        Args:
            classifier: Trained AdaBoostClassifier
            threshold: Decision threshold (adjusted for target rates)
            stage_num: Stage number in cascade
        """
        self.classifier = classifier
        self.threshold = threshold
        self.stage_num = stage_num

    def predict(self, feature_response_matrix):
        """
        Predict with this stage

        Args:
            feature_response_matrix: (N, M) feature responses

        Returns:
            predictions: (N,) binary labels
        """
        # Get confidence scores
        scores = self.classifier.predict_proba(feature_response_matrix)
        total_alpha = sum(self.classifier.alphas)

        # Apply threshold
        predictions = (scores >= self.threshold * total_alpha).astype(int)
        return predictions

    def __repr__(self):
        return f"CascadeStage(stage={self.stage_num}, n_weak={len(self.classifier.weak_classifiers)}, thresh={self.threshold:.3f})"


class CascadeClassifier:
    """
    Cascade of AdaBoost classifiers

    Processes samples through sequential stages. Each stage acts as filter:
    - Non-faces rejected early (fast)
    - Faces pass through all stages

    Detection formula:
        h(x) = 1 if all stages pass, else 0
    """

    def __init__(self):
        """Initialize cascade"""
        self.stages = []

    def add_stage(self, stage):
        """Add stage to cascade"""
        self.stages.append(stage)

    def predict(self, feature_response_matrix, return_stage_info=False):
        """
        Predict with cascade

        Args:
            feature_response_matrix: (N, M) feature responses
            return_stage_info: If True, return dict with per-stage statistics

        Returns:
            predictions: (N,) binary labels
            stage_info: (optional) Dict with rejection statistics per stage
        """
        N = feature_response_matrix.shape[0]
        active_mask = np.ones(N, dtype=bool)  # Samples still being evaluated
        predictions = np.zeros(N, dtype=int)

        stage_info = {
            'samples_per_stage': [],
            'rejections_per_stage': []
        }

        for i, stage in enumerate(self.stages):
            if not np.any(active_mask):
                break

            # Evaluate only active samples
            active_indices = np.where(active_mask)[0]
            active_responses = feature_response_matrix[active_indices]

            stage_predictions = stage.predict(active_responses)

            # Track statistics
            stage_info['samples_per_stage'].append(len(active_indices))
            stage_info['rejections_per_stage'].append(np.sum(stage_predictions == 0))

            # Update predictions for active samples
            predictions[active_indices] = stage_predictions

            # Deactivate rejected samples
            rejected_indices = active_indices[stage_predictions == 0]
            active_mask[rejected_indices] = False

        if return_stage_info:
            return predictions, stage_info
        return predictions

    def save(self, filepath):
        """Save cascade to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved cascade to {filepath}")

    @staticmethod
    def load(filepath):
        """Load cascade from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        return f"CascadeClassifier(n_stages={len(self.stages)})"


def adjust_threshold(classifier, feature_response_matrix, labels,
                     target_fpr=0.5, target_tpr=0.995):
    """
    Adjust threshold to achieve target false positive and true positive rates

    For cascade stages, we typically want:
    - High TPR (>99%) to avoid missing faces
    - Moderate FPR (30-50%) to reduce non-faces for next stage

    Args:
        classifier: Trained AdaBoostClassifier
        feature_response_matrix: (N, M) feature responses
        labels: (N,) ground truth
        target_fpr: Target false positive rate (default 0.5 = 50%)
        target_tpr: Target true positive rate (default 0.995 = 99.5%)

    Returns:
        best_threshold: Adjusted threshold value (as fraction of total_alpha)
    """
    # Get confidence scores
    scores = classifier.predict_proba(feature_response_matrix)
    total_alpha = sum(classifier.alphas)

    # Normalize scores
    normalized_scores = scores / total_alpha

    # Get sorted unique thresholds
    thresholds = np.sort(np.unique(normalized_scores))

    best_threshold = 0.5
    best_distance = float('inf')

    for thresh in thresholds:
        preds = (normalized_scores >= thresh).astype(int)

        # Calculate rates
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Distance from target
        # Prioritize TPR (must exceed target)
        if tpr >= target_tpr:
            distance = abs(fpr - target_fpr)
            if distance < best_distance:
                best_distance = distance
                best_threshold = thresh

    return best_threshold


def train_cascade(feature_response_matrix, labels, features,
                 stage_configs, verbose=True):
    """
    Train cascade of classifiers

    Algorithm (simplified from Viola-Jones Section 4):
    1. Train Stage 1 on full dataset
    2. Adjust threshold to achieve target detection/FP rates
    3. Collect false positives that passed Stage 1
    4. Train Stage 2 on faces + FPs from Stage 1
    5. Repeat for additional stages

    Args:
        feature_response_matrix: (N, M) feature responses for full dataset
        labels: (N,) ground truth labels
        features: List of HaarFeature objects
        stage_configs: List of dicts with stage parameters
            Example: [
                {'T': 10, 'target_fpr': 0.5, 'target_tpr': 0.995},
                {'T': 40, 'target_fpr': 0.01, 'target_tpr': 0.99}
            ]
        verbose: Print progress

    Returns:
        CascadeClassifier
    """
    cascade = CascadeClassifier()

    if verbose:
        print("=" * 60)
        print(f"Training Cascade with {len(stage_configs)} stages")
        print("=" * 60)

    # Start with full dataset
    current_responses = feature_response_matrix
    current_labels = labels

    for stage_num, config in enumerate(stage_configs, 1):
        T = config['T']
        target_fpr = config.get('target_fpr', 0.5)
        target_tpr = config.get('target_tpr', 0.995)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Stage {stage_num}/{len(stage_configs)}")
            print(f"  T={T} weak classifiers")
            print(f"  Target FPR: {target_fpr:.1%}, Target TPR: {target_tpr:.1%}")
            print(f"  Training samples: {len(current_labels)}")
            print(f"    Faces: {np.sum(current_labels == 1)}")
            print(f"    Non-faces: {np.sum(current_labels == 0)}")
            print("=" * 60)

        # Train AdaBoost for this stage
        classifier = train_adaboost(
            current_responses,
            current_labels,
            features,
            T=T,
            verbose=verbose
        )

        # Adjust threshold
        if verbose:
            print(f"\nAdjusting threshold for Stage {stage_num}...")

        threshold = adjust_threshold(
            classifier,
            current_responses,
            current_labels,
            target_fpr=target_fpr,
            target_tpr=target_tpr
        )

        if verbose:
            print(f"  Adjusted threshold: {threshold:.4f}")

        # Create and add stage
        stage = CascadeStage(classifier, threshold, stage_num)
        cascade.add_stage(stage)

        # Evaluate stage
        stage_preds = stage.predict(current_responses)
        tp = np.sum((stage_preds == 1) & (current_labels == 1))
        fp = np.sum((stage_preds == 1) & (current_labels == 0))
        tn = np.sum((stage_preds == 0) & (current_labels == 0))
        fn = np.sum((stage_preds == 0) & (current_labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        if verbose:
            print(f"\nStage {stage_num} Performance:")
            print(f"  TPR (Detection Rate): {tpr:.2%}")
            print(f"  FPR (False Positive Rate): {fpr:.2%}")
            print(f"  Faces passing: {tp}/{tp+fn}")
            print(f"  Non-faces rejected: {tn}/{fp+tn}")

        # Prepare data for next stage
        # Keep all faces + false positives from this stage
        if stage_num < len(stage_configs):
            passing_mask = stage_preds == 1
            current_responses = current_responses[passing_mask]
            current_labels = current_labels[passing_mask]

            if verbose:
                print(f"\nPreparing for Stage {stage_num + 1}:")
                print(f"  Samples passing to next stage: {len(current_labels)}")
                print(f"    Faces: {np.sum(current_labels == 1)}")
                print(f"    Non-faces (FPs): {np.sum(current_labels == 0)}")

    if verbose:
        print("\n" + "=" * 60)
        print("Cascade Training Complete!")
        print(f"  Total stages: {len(cascade.stages)}")
        print("=" * 60)

    return cascade


def evaluate_cascade(cascade, feature_response_matrix, labels, verbose=True):
    """
    Evaluate cascade performance

    Args:
        cascade: Trained CascadeClassifier
        feature_response_matrix: (N, M) feature responses
        labels: (N,) ground truth
        verbose: Print detailed metrics

    Returns:
        metrics: Dict with performance metrics
        stage_stats: List of dicts with per-stage statistics
    """
    predictions, stage_info = cascade.predict(
        feature_response_matrix,
        return_stage_info=True
    )

    # Overall metrics
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

    # Compute detailed stage statistics
    stage_stats = []
    active_mask = np.ones(len(labels), dtype=bool)
    
    for i, stage in enumerate(cascade.stages):
        if not np.any(active_mask):
            break

        active_indices = np.where(active_mask)[0]
        active_labels = labels[active_indices]
        active_responses = feature_response_matrix[active_indices]
        
        stage_predictions = stage.predict(active_responses)
        
        # Count faces and nonfaces at input
        faces_in = np.sum(active_labels == 1)
        nonfaces_in = np.sum(active_labels == 0)
        
        # Count what passes and gets rejected
        passed_mask = stage_predictions == 1
        rejected_mask = stage_predictions == 0
        
        faces_passed = np.sum((active_labels == 1) & passed_mask)
        nonfaces_passed = np.sum((active_labels == 0) & passed_mask)
        faces_rejected = np.sum((active_labels == 1) & rejected_mask)
        nonfaces_rejected = np.sum((active_labels == 0) & rejected_mask)
        
        stage_stat = {
            'faces_in': faces_in,
            'nonfaces_in': nonfaces_in,
            'faces_passed': faces_passed,
            'nonfaces_passed': nonfaces_passed,
            'faces_rejected': faces_rejected,
            'nonfaces_rejected': nonfaces_rejected
        }
        stage_stats.append(stage_stat)
        
        # Update active mask
        rejected_indices = active_indices[rejected_mask]
        active_mask[rejected_indices] = False

    if verbose:
        print("\n" + "=" * 60)
        print("Cascade Evaluation Metrics")
        print("=" * 60)
        print(f"Overall Performance:")
        print(f"  Accuracy:  {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1 Score:  {f1:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:5d}  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  TN: {tn:5d}")

        print(f"\nCascade Stage Statistics:")
        for i, (n_samples, n_rejected) in enumerate(zip(
            stage_info['samples_per_stage'],
            stage_info['rejections_per_stage']
        ), 1):
            rejection_rate = n_rejected / n_samples if n_samples > 0 else 0
            print(f"  Stage {i}: {n_samples:5d} samples, {n_rejected:5d} rejected ({rejection_rate:.1%})")

        print("=" * 60)

    return metrics, stage_stats


# Main execution for testing
if __name__ == '__main__':
    print("=" * 60)
    print("Cascade Classifier Implementation")
    print("=" * 60)
    print("\nTo train cascade:")
    print("1. Load dataset and compute feature responses")
    print("2. Define stage configurations:")
    print("   stage_configs = [")
    print("       {'T': 10, 'target_fpr': 0.5, 'target_tpr': 0.995},")
    print("       {'T': 40, 'target_fpr': 0.01, 'target_tpr': 0.99}")
    print("   ]")
    print("3. Call train_cascade()")
    print("\nSee training notebook for complete example.")
    print("=" * 60)
