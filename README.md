# Viola-Jones Face Detector

From-scratch implementation of the Viola-Jones face detection algorithm for ELL715 Assignment 5. Complete codebase with reproducible results.

## Installation

```bash
# Clone repository
git clone git@github.com:Unfortunate-Happenstance/ELL715_Assignment_5.git                      
cd ELL715_Assignment_5

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -r requirements.txt
```

## Quick Start

### Dataset Preparation
```bash
# Extract patches from Faces94 dataset
uv run python -m src.data.dataset_generator
```

### Training
```bash
# Run notebooks in sequence
uv run jupyter notebook notebooks/01_dataset_exploration.ipynb
uv run jupyter notebook notebooks/02_haar_features_demo.ipynb
uv run jupyter notebook notebooks/03_adaboost_training.ipynb  # ~5-10 min
uv run jupyter notebook notebooks/04_cascade_training.ipynb   # ~10-15 min
```

### Testing
```bash
# Run evaluation scripts
uv run python scripts/evaluate_adaboost.py
uv run python scripts/evaluate_cascade.py
```

## Reproducibility

- **Environment**: Python 3.9+, dependencies pinned in `uv.lock`/`requirements.txt`
- **Data**: Faces94 dataset (place in `faces94/` directory)
- **Random Seed**: Set in notebooks for consistent results
- **Caching**: Feature responses cached to `data/processed/` (183MB train, 518MB test)
- **Models**: Trained models saved to `data/models/` for reuse
- **Results**: All figures and metrics reproducible from notebooks

## Project Structure

```
src/
├── data/dataset_generator.py      # Patch extraction (16x16, 5:1 imbalance)
├── features/
│   ├── integral_image.py          # O(1) rectangle sums
│   └── haar_features.py           # 10k Haar features (5 types)
├── classifiers/
│   ├── weak_classifier.py         # Threshold optimization
│   ├── adaboost.py                # Ensemble learning (T=50)
│   └── cascade.py                 # Multi-stage cascade
└── detector/sliding_window.py     # Multi-scale detection

notebooks/                         # Training and evaluation
data/                              # Processed data and models
results/figures/                   # Output plots and metrics
```

## Status & Results

**V1 (Baseline)**: 10k features, T=50 AdaBoost, 2-stage cascade
- Training: 799 faces + 3995 non-faces
- Testing: 2260 faces + 11300 non-faces
- Accuracy: See notebook outputs (target >70%)

**V2 (Optimized)**: 32k features, T=200 AdaBoost, multi-stage cascade
- Enhanced feature set and longer training
- Target: 80-85% accuracy

## References

- Viola, P. & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features"
- Dataset: Faces94 (AT&T/Essex University)
