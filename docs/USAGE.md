# Usage Guide

Step-by-step guide to train and use the Viola-Jones face detector.

## Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone/navigate to project
cd ELL715_Assignment_5

# Install dependencies
uv sync

# Verify installation
uv run python -c "import numpy; print('OK')"
```

**Dependencies**: numpy, scipy, scikit-image, matplotlib, seaborn, tqdm, jupyter, pytest

## Training Workflow

### Step 1: Explore Dataset

```bash
uv run jupyter notebook notebooks/01_dataset_exploration.ipynb
```

**What it does**:
- Loads 799 training faces, 3,995 non-faces
- Visualizes sample patches
- Shows intensity/variance distributions
- Computes average face pattern

**Output**: Figures in `results/figures/`

### Step 2: (Optional) Visualize Haar Features

```bash
uv run jupyter notebook notebooks/02_haar_features_demo.ipynb
```

**What it does**:
- Generates and visualizes Haar feature patterns
- Shows feature responses on sample faces

### Step 3: Train AdaBoost

```bash
uv run jupyter notebook notebooks/03_adaboost_training.ipynb
```

**Duration**: ~5-10 minutes

**What it does**:
1. Loads dataset (799 faces, 3,995 non-faces)
2. Generates 10,000 Haar features
3. Computes feature responses (47.9M evaluations, cached to `data/processed/`)
4. Trains AdaBoost with T=50 rounds
5. Evaluates on train/test sets
6. Saves model to `data/models/adaboost_v1_T50.pkl`

**Expected output**:
- Training accuracy: ~95-99%
- Test accuracy: ~70-85%

**Files created**:
- `data/processed/train_responses_10k.npy` (183MB)
- `data/processed/test_responses_10k.npy` (518MB)
- `data/models/adaboost_v1_T50.pkl` (~1-2MB)
- `results/figures/selected_features.png`

### Step 4: Train Cascade

```bash
uv run jupyter notebook notebooks/04_cascade_training.ipynb
```

**Duration**: ~10-15 minutes

**Prerequisites**: Step 3 completed (needs cached feature responses)

**What it does**:
1. Loads pre-computed feature responses
2. Trains Stage 1 (T=10) on full dataset
3. Adjusts threshold for TPR=99.5%, FPR=50%
4. Trains Stage 2 (T=40) on faces + FPs from Stage 1
5. Evaluates cascade performance
6. Compares with single AdaBoost
7. Saves model to `data/models/cascade_v1_2stage.pkl`

**Expected output**:
- Cascade test accuracy: ~70-85%
- Stage 1 rejects ~50% non-faces
- Improved precision vs single AdaBoost

## Using Trained Models

### Load and Predict

```python
import numpy as np
import pickle
from src.classifiers.adaboost import AdaBoostClassifier
from src.classifiers.cascade import CascadeClassifier

# Load model
model = AdaBoostClassifier.load('data/models/adaboost_v1_T50.pkl')
# Or: model = CascadeClassifier.load('data/models/cascade_v1_2stage.pkl')

# Load pre-computed feature responses
test_responses = np.load('data/processed/test_responses_10k.npy')

# Predict
predictions = model.predict(test_responses)

# Get confidence scores (AdaBoost only)
scores = model.predict_proba(test_responses)
```

### Extract Features from New Image

```python
from src.features.integral_image import compute_ii_fast
from src.features.haar_features import generate_haar_features

# Load/prepare 16x16 grayscale patch
patch = ...  # shape (16, 16), dtype uint8 or float

# Generate same features as training
features = generate_haar_features(window_size=16, max_features=10000)

# Compute integral image
ii = compute_ii_fast(patch.astype(np.float64))

# Compute feature responses
responses = np.array([f.compute(ii) for f in features])  # shape (10000,)

# Predict (need to reshape for model)
prediction = model.predict(responses.reshape(1, -1))  # shape (1,)
```

## Model Files

**Location**: `data/models/`

| File | Size | Description |
|------|------|-------------|
| `adaboost_v1_T50.pkl` | ~1-2MB | Single AdaBoost (50 weak classifiers) |
| `cascade_v1_2stage.pkl` | ~2-3MB | 2-stage cascade (Stage 1: T=10, Stage 2: T=40) |

**Contents**:
- List of WeakClassifier objects (feature_idx, threshold, polarity)
- Alpha weights for AdaBoost
- Stage thresholds for cascade

**Note**: Models depend on feature generation order - must use same `max_features=10000` setting

## Common Issues

### 1. UnicodeEncodeError (Windows)

**Error**: `'charmap' codec can't encode character '\u2713'`

**Fix**: Code uses 'x' for multiplication and '[OK]' instead of Unicode checkmarks

### 2. Long Computation Time

**Symptom**: Feature response computation takes >10 minutes

**Solutions**:
- First run: Pre-computation is normal (~3-5 min)
- Subsequent runs: Loads from cache (`.npy` files)
- If re-running: Delete `data/processed/*_responses*.npy` to force recompute

### 3. Out of Memory

**Symptom**: Crashes during feature response computation

**Solutions**:
- Reduce features: `max_features=5000` instead of 10000
- Use smaller batch size in `compute_feature_responses()`
- Close other applications

### 4. Import Errors

**Error**: `ImportError: attempted relative import with no known parent package`

**Fix**: Code handles this with try/except blocks - should work as both module and script

### 5. Model Not Found

**Error**: `FileNotFoundError: data/models/adaboost_v1_T50.pkl`

**Solution**: Run notebook 03 first to train and save AdaBoost model

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_integral_image.py -v

# Expected: 12 tests passed
```

## Next Steps

### V2: Scale Up (Better Accuracy)

**Changes**:
1. Increase features: `max_features=50000` in `generate_haar_features()`
2. Increase AdaBoost rounds: `T=200` in `train_adaboost()`
3. Add cascade stages: 3-5 total stages
4. More training data: Augment with rotations/scaling

**Expected**: 80-85% test accuracy

**Cost**: 5x longer training (~1 hour total)

### Part 2: Multi-Face Detection

**Implementation**:
1. Sliding window over full image
2. Multi-scale image pyramid (scale factor ~1.25)
3. Apply cascade at each window
4. Non-maximum suppression for overlapping detections

**New files**:
- `src/detector/sliding_window.py`
- `notebooks/05_detection_demo.ipynb`

## Troubleshooting

**Q**: Training accuracy >99% but test accuracy <75%?

**A**: Overfitting. Solutions:
- Reduce T (try T=30 instead of T=50)
- Add more non-face diversity
- Use cross-validation

**Q**: Cascade worse than single AdaBoost?

**A**: Threshold adjustment issues. Check:
- Stage 1 TPR should be >99% (check notebook output)
- Stage 2 should receive enough non-faces from Stage 1
- Try adjusting target_fpr/target_tpr in stage configs

**Q**: Want to change dataset split?

**A**: Modify `src/data/dataset_generator.py`:
```python
train_folders = ['female', 'malestaff']
test_folders = ['male']
```

## Performance Benchmarks

**Hardware**: Typical laptop (Intel i5/i7, 8-16GB RAM)

| Operation | Time | Memory |
|-----------|------|--------|
| Generate 10k features | <1 sec | <100MB |
| Compute train responses | 3-5 min | 183MB |
| Compute test responses | 5-8 min | 518MB |
| Train AdaBoost T=50 | 5-8 min | <500MB |
| Train cascade 2-stage | 10-15 min | <500MB |

**Faster**: Use SSD, close background apps, increase Python process priority
