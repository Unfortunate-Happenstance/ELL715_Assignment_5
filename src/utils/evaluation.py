"""
Evaluation Utilities for Face Detection Models

Provides comprehensive evaluation metrics and visualizations:
- ROC curve with AUC score
- Precision-Recall curve
- Training/validation curves
- Feature importance analysis
- Confusion matrix visualization

Assignment: ELL715 Assignment 5 - V2 Enhancements
AI Usage: Implementation assisted by Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix


def plot_roc_curve(y_true, y_scores, label='Model', ax=None):
    """
    Plot ROC curve with AUC score

    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        label: Label for the curve
        ax: Matplotlib axis (creates new if None)

    Returns:
        Tuple of (fpr, tpr, auc_score, ax)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fpr, tpr, roc_auc, ax


def plot_precision_recall_curve(y_true, y_scores, label='Model', ax=None):
    """
    Plot Precision-Recall curve with average precision

    Important for imbalanced datasets (face detection has rare positives)

    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        label: Label for the curve
        ax: Matplotlib axis (creates new if None)

    Returns:
        Tuple of (precision, recall, avg_precision, ax)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, linewidth=2, label=f'{label} (AP = {avg_precision:.3f})')

    # Baseline (proportion of positives)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Baseline ({baseline:.3f})')

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return precision, recall, avg_precision, ax


def plot_training_curves(train_history, val_history=None, metric='Accuracy', ax=None):
    """
    Plot training and validation curves over AdaBoost rounds

    Helps identify overfitting and optimal stopping point

    Args:
        train_history: List of training metric values per round
        val_history: List of validation metric values per round (optional)
        metric: Name of metric being plotted
        ax: Matplotlib axis (creates new if None)

    Returns:
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    rounds = np.arange(1, len(train_history) + 1)

    ax.plot(rounds, train_history, linewidth=2, marker='o', markersize=4,
            label=f'Training {metric}', color='blue')

    if val_history is not None:
        ax.plot(rounds, val_history, linewidth=2, marker='s', markersize=4,
                label=f'Validation {metric}', color='orange')

        # Mark best validation point
        best_idx = np.argmax(val_history)
        best_val = val_history[best_idx]
        ax.axvline(x=best_idx+1, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.plot(best_idx+1, best_val, 'r*', markersize=15,
                label=f'Best ({best_idx+1} rounds: {best_val:.3f})')

    ax.set_xlabel('AdaBoost Round (T)', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} vs Training Rounds', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def analyze_feature_importance(classifier, features, top_k=20):
    """
    Analyze and visualize feature importance by alpha weights

    Shows which Haar features contribute most to classification

    Args:
        classifier: Trained AdaBoostClassifier
        features: List of HaarFeature objects
        top_k: Number of top features to display

    Returns:
        Sorted list of (feature_idx, alpha, feature) tuples
    """
    # Get alpha weights and feature indices from weak classifiers
    feature_importance = []
    for weak_clf, alpha in zip(classifier.weak_classifiers, classifier.alphas):
        feat_idx = weak_clf.feature_idx
        feature_importance.append((feat_idx, alpha, features[feat_idx]))

    # Sort by alpha (descending)
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Display top features
    print(f"\nTop {top_k} Most Important Features:")
    print("=" * 70)
    print(f"{'Rank':<6} {'Alpha':<10} {'Type':<8} {'Position':<15} {'Size':<10}")
    print("-" * 70)

    for rank, (feat_idx, alpha, feat) in enumerate(feature_importance[:top_k], 1):
        print(f"{rank:<6} {alpha:<10.4f} {feat.type:<8} "
              f"({feat.x},{feat.y}){'':>7} {feat.width}x{feat.height}")

    print("=" * 70)

    # Count feature types
    type_counts = {}
    type_alphas = {}
    for feat_idx, alpha, feat in feature_importance:
        ftype = feat.type
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
        type_alphas[ftype] = type_alphas.get(ftype, 0) + alpha

    print("\nFeature Type Distribution:")
    print("-" * 40)
    for ftype in sorted(type_counts.keys()):
        count = type_counts[ftype]
        total_alpha = type_alphas[ftype]
        print(f"  {ftype}: {count} features (total alpha: {total_alpha:.2f})")
    print("-" * 40)

    return feature_importance


def visualize_top_features(feature_importance, top_k=20, window_size=16):
    """
    Visualize top features as image patches

    Args:
        feature_importance: Sorted list from analyze_feature_importance()
        top_k: Number of features to visualize
        window_size: Size of detection window

    Returns:
        Figure object
    """
    try:
        from ..features.haar_features import visualize_feature
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'features'))
        from haar_features import visualize_feature

    # Create grid
    n_cols = 5
    n_rows = (top_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = axes.flatten() if top_k > 1 else [axes]

    for i in range(top_k):
        feat_idx, alpha, feat = feature_importance[i]
        feat_img = visualize_feature(feat, window_size)

        axes[i].imshow(feat_img, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i].set_title(f"Rank {i+1}: {feat.type}\nalpha={alpha:.3f}", fontsize=9)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(top_k, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Top {top_k} Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def confusion_matrix_detailed(y_true, y_pred, labels=['Non-Face', 'Face'], normalize=False):
    """
    Create detailed confusion matrix visualization

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to show percentages

    Returns:
        Tuple of (cm, fig, ax)
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Percentage' if normalize else 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Add metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = f"Accuracy: {accuracy:.3f}  |  Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1: {f1:.3f}"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    return cm, fig, ax


def compare_models_table(models_dict):
    """
    Create comparison table for multiple models

    Args:
        models_dict: Dictionary of {model_name: metrics_dict}
                    where metrics_dict contains 'accuracy', 'precision', 'recall', etc.

    Returns:
        DataFrame with comparison
    """
    import pandas as pd

    df = pd.DataFrame(models_dict).T

    # Format as percentages
    for col in df.columns:
        if col in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            df[col] = df[col].apply(lambda x: f"{x:.2%}")

    print("\nModel Comparison:")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)

    return df


# Main test
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Evaluation Utilities")
    print("=" * 60)

    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_scores = y_true + np.random.randn(n_samples) * 0.3
    y_pred = (y_scores > 0.5).astype(int)

    # Test ROC curve
    print("\nTesting ROC curve...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, roc_auc, ax = plot_roc_curve(y_true, y_scores, label='Test Model', ax=ax)
    print(f"  AUC: {roc_auc:.3f}")
    plt.close()

    # Test PR curve
    print("\nTesting Precision-Recall curve...")
    fig, ax = plt.subplots(figsize=(8, 6))
    precision, recall, avg_precision, ax = plot_precision_recall_curve(y_true, y_scores, label='Test Model', ax=ax)
    print(f"  Average Precision: {avg_precision:.3f}")
    plt.close()

    # Test training curves
    print("\nTesting training curves...")
    train_history = [0.6, 0.7, 0.75, 0.8, 0.83, 0.85, 0.86, 0.87]
    val_history = [0.58, 0.68, 0.73, 0.77, 0.79, 0.78, 0.77, 0.76]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = plot_training_curves(train_history, val_history, metric='Accuracy', ax=ax)
    plt.close()

    # Test confusion matrix
    print("\nTesting confusion matrix...")
    cm, fig, ax = confusion_matrix_detailed(y_true, y_pred)
    print(f"  Confusion matrix shape: {cm.shape}")
    plt.close()

    print("\n" + "=" * 60)
    print("[OK] Evaluation utilities module ready!")
    print("=" * 60)
