# Implementation Details

Technical architecture and algorithm documentation for V1 implementation.

## Architecture

**Data Flow**:
```
Images → Dataset Gen → Patches (16x16)
                ↓
         Integral Images
                ↓
         Haar Features (10k) → Feature Responses (NxM matrix)
                                        ↓
                                  AdaBoost Training
                                   ↓          ↓
                            Stage 1 (T=10)  Stage 2 (T=40)
                                   ↓          ↓
                               Cascade Classifier
```

**Component Dependencies**:
- `weak_classifier.py` ← uses → `numpy`
- `adaboost.py` ← uses → `weak_classifier.py`
- `cascade.py` ← uses → `adaboost.py`
- `haar_features.py` ← uses → `integral_image.py`

## Core Modules

### 1. Integral Image (`src/features/integral_image.py`)

**Algorithm**: Viola-Jones §2.1

Cumulative sum table for O(1) rectangle sum computation.

**Key formulas**:
```python
s(x,y) = s(x,y-1) + i(x,y)        # Row cumulative sum
ii(x,y) = ii(x-1,y) + s(x,y)      # Integral image
sum = ii[y2,x2] - ii[y1,x2] - ii[y2,x1] + ii[y1,x1]  # Rectangle sum
```

**Implementation** (`integral_image.py:15-30`):
- Dimensions: (H+1) x (W+1) with zero-padding
- Uses cumulative row sums for efficiency
- `compute_ii_fast()`: Numpy vectorized version
- `rectangle_sum()`: 4-point lookup

**Tests**: 12 unit tests (edge cases, random patches, brute force validation)

### 2. Haar Features (`src/features/haar_features.py`)

**Feature types** (5 total):
| Type | Pattern | White Regions | Black Regions |
|------|---------|---------------|---------------|
| 2h | Horizontal | Top | Bottom |
| 2v | Vertical | Left | Right |
| 3h | Horizontal | Top + Bottom | Middle |
| 3v | Vertical | Left + Right | Middle |
| 4d | Diagonal | TL + BR | TR + BL |

**Generation** (`haar_features.py:149-223`):
- All positions (x,y) in 16x16 window
- All valid sizes (even widths/heights for splits)
- V1: Limited to first 10,000 features (from ~45-60k total)

**Feature computation**:
```python
value = sum(white_rectangles) - sum(black_rectangles)
# Uses integral image: O(1) per rectangle, O(k) per feature (k=2-4 rects)
```

**Example** (2h feature):
```python
def _compute_2h(self, ii):
    h_half = self.height // 2
    top = rectangle_sum(ii, self.x, self.y, self.width, h_half)
    bottom = rectangle_sum(ii, self.x, self.y + h_half, self.width, h_half)
    return bottom - top  # Positive if bottom brighter
```

### 3. Weak Classifier (`src/classifiers/weak_classifier.py`)

**Algorithm**: Single-feature threshold classifier

**Formula**:
```
h(x) = 1 if p*f(x) < p*theta else 0
  where p ∈ {+1, -1} is polarity
```

**Training** (`weak_classifier.py:60-135`):
- Sort feature values
- Try all unique values as thresholds
- Test both polarities (+1, -1)
- Select threshold + polarity with minimum weighted error

**Optimization**: Cumulative weight sums for O(N) threshold search

**Key function**:
```python
def find_best_threshold(feature_responses, labels, weights):
    # Sort samples by feature value
    # Accumulate positive/negative weights
    # Error = cum_neg + (total_pos - cum_pos)  # for polarity +1
    # Return: threshold, polarity, error
```

### 4. AdaBoost (`src/classifiers/adaboost.py`)

**Algorithm**: Viola-Jones Table 1 (exact implementation)

**Pseudocode**:
```
1. Initialize weights:
   w[i] = 1/(2m) for negatives
   w[i] = 1/(2l) for positives

2. For t = 1 to T:
   a) Normalize: w ← w / sum(w)
   b) Select best feature j with minimum error ε
   c) Compute: β = ε / (1 - ε)
   d) Update: w[i] ← w[i] * β^(1 - e[i])
      where e[i] = 0 if correct, 1 if incorrect

3. Final classifier:
   h(x) = 1 if sum(α_t * h_t(x)) >= 0.5 * sum(α_t)
   where α_t = log(1/β_t)
```

**Implementation** (`adaboost.py:102-233`):
- Pre-compute feature response matrix (N x M) for efficiency
- Each round: search all M features for minimum weighted error
- Store selected weak classifiers + alpha weights
- Default threshold: 0.5 (adjustable for cascade)

**Critical details**:
- Weight init: `1/(2*m)` for negatives, `1/(2*l)` for positives (handles imbalance)
- Normalization: MUST normalize weights each round
- Early stop if ε >= 0.5 (no discriminative power)

**V1 config**: T=50 weak classifiers

### 5. Cascade (`src/classifiers/cascade.py`)

**Algorithm**: Viola-Jones §4 (multi-stage filtering)

**Concept**:
```
Sample → Stage 1 → reject 50% non-faces
              ↓
         Stage 2 → final classification
```

**Training procedure** (`cascade.py:148-248`):
```
1. Train Stage 1 on full dataset (T=10)
2. Adjust threshold to achieve target TPR=99.5%, FPR=50%
3. Collect false positives from Stage 1
4. Train Stage 2 on faces + FPs (T=40)
5. Adjust threshold for final TPR=99%, FPR=1%
```

**Threshold adjustment** (`cascade.py:128-146`):
- Sweep all possible thresholds
- Find threshold closest to target FPR while maintaining TPR >= target
- Prioritize TPR (avoid missing faces)

**V1 config**:
| Stage | T | Target TPR | Target FPR |
|-------|---|------------|------------|
| 1 | 10 | 99.5% | 50% |
| 2 | 40 | 99% | 1% |

**Efficiency**: Most non-faces rejected at Stage 1 (cheap 10 features)

## Key Algorithms (Code)

### Feature Response Matrix

**Purpose**: Avoid recomputing features during AdaBoost training

```python
# src/features/haar_features.py:226-271
def compute_feature_responses(features, patches):
    N, M = len(patches), len(features)
    responses = np.zeros((N, M), dtype=np.float32)

    # Pre-compute integral images
    integral_images = [compute_ii_fast(p) for p in patches]

    # Evaluate all features on all patches
    for i, ii in enumerate(integral_images):
        for j, feature in enumerate(features):
            responses[i, j] = feature.compute(ii)

    return responses  # (N_patches, N_features)
```

**Storage**: Cached to disk (183MB train, 518MB test for 10k features)

### AdaBoost Weight Update

```python
# src/classifiers/adaboost.py:191-202
# Get predictions for selected weak classifier
feature_values = feature_response_matrix[:, weak_clf.feature_idx]
predictions = weak_clf.predict(feature_values)

# e_i = 0 if correct, 1 if incorrect
errors = (predictions != labels).astype(int)

# Weight update: w_{t+1,i} = w_{t,i} * beta^{1-e_i}
# Correct samples: multiply by beta (reduce weight)
# Incorrect samples: multiply by 1 (keep weight)
weights = weights * (beta ** (1 - errors))
```

### Cascade Prediction

```python
# src/classifiers/cascade.py:45-76
def predict(self, feature_response_matrix):
    N = feature_response_matrix.shape[0]
    active_mask = np.ones(N, dtype=bool)
    predictions = np.zeros(N, dtype=int)

    for stage in self.stages:
        # Evaluate only active (not yet rejected) samples
        active_indices = np.where(active_mask)[0]
        stage_preds = stage.predict(feature_response_matrix[active_indices])

        # Update predictions and deactivate rejected samples
        predictions[active_indices] = stage_preds
        rejected = active_indices[stage_preds == 0]
        active_mask[rejected] = False

    return predictions
```

## Performance Optimizations

**Bottleneck**: 10k features x 4.8k training samples = 47.9M evaluations

**Solutions**:
1. **Pre-compute responses**: One-time cost, cache to disk (`.npy` files)
2. **float32**: Half memory vs float64 (183MB vs 366MB)
3. **Vectorization**: Numpy operations on entire arrays
4. **Progress tracking**: `tqdm` for long operations

**Memory profile**:
- Train responses: 4,794 x 10,000 x 4 bytes = 183MB
- Test responses: 13,560 x 10,000 x 4 bytes = 518MB
- Total cached: ~700MB

**Compute time** (approximate):
- Feature response computation: ~3-5 min (cached)
- AdaBoost T=50: ~5-8 min
- Cascade 2-stage: ~10-15 min total

## Testing

**Unit tests**: `tests/test_integral_image.py` (12 tests)

Coverage:
- Edge cases (1x1, 2x2 images)
- Random patches (compare to brute force)
- Zero padding boundary conditions
- Rectangle sum at all positions

**Validation approach**:
- Separate test set (male/ folder, unseen during training)
- Metrics: accuracy, precision, recall, F1, confusion matrix
- Per-stage analysis for cascade

**Success criteria** (V1):
- Minimum: 70% test accuracy
- Good: 75-80%
- Excellent: >85%

## Critical Implementation Details

**From Viola-Jones paper**:
1. Integral image dimensions: (H+1) x (W+1) with s(x,-1)=0, ii(-1,y)=0
2. Weight normalization: MUST happen before each AdaBoost round
3. Feature count: ~45-60k for 16x16 (V1 uses first 10k)
4. Cascade thresholds: Adjust per stage for target TPR/FPR, not overall accuracy
5. False positive bootstrapping: Use FPs from stage i to train stage i+1

**Import handling** (module vs script):
```python
try:
    from .integral_image import rectangle_sum  # Module import
except ImportError:
    from integral_image import rectangle_sum   # Script import
```

## File Reference

| File | Key Functions | Lines |
|------|---------------|-------|
| `integral_image.py` | `compute_ii_fast()`, `rectangle_sum()` | 15-30, 40-55 |
| `haar_features.py` | `generate_haar_features()`, `compute_feature_responses()` | 149-223, 226-271 |
| `weak_classifier.py` | `find_best_threshold()`, `select_best_feature()` | 60-135, 163-202 |
| `adaboost.py` | `train_adaboost()`, `evaluate_classifier()` | 102-233, 236-287 |
| `cascade.py` | `train_cascade()`, `adjust_threshold()` | 148-248, 128-146 |
