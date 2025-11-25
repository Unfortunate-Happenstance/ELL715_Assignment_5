# Viola-Jones Face Detector

From-scratch implementation of the Viola-Jones face detection algorithm (2001 paper) for ELL715 Assignment 5.

## Dataset

**Faces94**: Grayscale face images
- Training: 799 faces (female/ + malestaff/)
- Testing: 2260 faces (male/)
- Patches: 16x16 pixels, center crop = face, 5 random crops = non-face
- Training total: 799 faces + 3,995 non-faces = 4,794 patches
- Testing total: 2,260 faces + 11,300 non-faces = 13,560 patches

## Quick Start

```bash
# Install dependencies (using uv)
uv sync

# Run notebooks in order
uv run jupyter notebook notebooks/01_dataset_exploration.ipynb
uv run jupyter notebook notebooks/03_adaboost_training.ipynb  # ~5-10 min
uv run jupyter notebook notebooks/04_cascade_training.ipynb   # ~10-15 min
```

## Project Structure

```
src/
├── data/
│   └── dataset_generator.py          # Extract 16x16 patches from Faces94
├── features/
│   ├── integral_image.py              # O(1) rectangle sum (V-J §2.1)
│   └── haar_features.py               # Generate 10k Haar features (5 types)
├── classifiers/
│   ├── weak_classifier.py             # Threshold-based single-feature classifier
│   ├── adaboost.py                    # AdaBoost algorithm (V-J Table 1)
│   └── cascade.py                     # 2-stage cascade (V-J §4)
└── tests/
    └── test_integral_image.py         # 12 unit tests

notebooks/
├── 01_dataset_exploration.ipynb       # EDA, patch visualization
├── 02_haar_features_demo.ipynb        # Feature visualization
├── 03_adaboost_training.ipynb         # Train AdaBoost (T=50, 10k features)
└── 04_cascade_training.ipynb          # Train 2-stage cascade

data/
├── processed/
│   ├── train_faces.pkl, train_nonfaces.pkl
│   ├── test_faces.pkl, test_nonfaces.pkl
│   ├── train_responses_10k.npy        # Cached feature responses (183MB)
│   └── test_responses_10k.npy         # (518MB)
└── models/
    ├── adaboost_v1_T50.pkl            # Single AdaBoost classifier
    └── cascade_v1_2stage.pkl          # 2-stage cascade
```

## V1 Status (Complete)

**Implemented**:
- [x] Dataset generation (799 train, 2260 test faces)
- [x] Integral image with unit tests
- [x] Haar feature generation (10,000 features)
- [x] Weak classifier (threshold + polarity)
- [x] AdaBoost (T=50 rounds, V-J Table 1 algorithm)
- [x] Cascade (2 stages: T1=10, T2=40)
- [x] Training notebooks with evaluation

**Performance** (V1 target: >70% accuracy):
- AdaBoost T=50: See notebook 03 output
- Cascade 2-stage: See notebook 04 output

## Next Steps

**V2 (Scale-up)**:
- Increase to 50k Haar features
- AdaBoost T=200 rounds
- 3-5 cascade stages
- Target: 80-85% accuracy

**Part 2 (Detection)**:
- Sliding window over full images
- Multi-scale pyramid
- Non-maximum suppression

## Key Files

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Dataset | `src/data/dataset_generator.py` | 150 | Extract patches, 5:1 imbalance |
| Integral | `src/features/integral_image.py` | 120 | O(1) rectangle sum, cumulative sums |
| Features | `src/features/haar_features.py` | 355 | 2h/2v/3h/3v/4d patterns |
| Weak | `src/classifiers/weak_classifier.py` | 245 | Find optimal threshold per feature |
| AdaBoost | `src/classifiers/adaboost.py` | 300 | Weight update, feature selection |
| Cascade | `src/classifiers/cascade.py` | 400 | Multi-stage, threshold adjustment |

## References

- **Paper**: Viola, P. & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features"
- **Dataset**: Faces94 (AT&T/Essex)
- **Assignment**: ELL715 Assignment 5 (160 marks total)
  - Part 1: Implementation (120 marks: 20+20+20+40+20)
  - Part 2: Multi-face detection (40 marks)

## Technical Notes

- **AI Usage**: Algorithm structure and docstrings assisted by Claude Code
- **Dependencies**: numpy, scipy, scikit-image, matplotlib, jupyter (see `pyproject.toml`)
- **Compute**: Pre-compute feature responses to disk (47.9M training, 135.6M testing evaluations)
- **Memory**: float32 for responses (~700MB total)
- **Known issues**: Windows charmap → use 'x' for multiplication, avoid Unicode

## Documentation

- `docs/IMPLEMENTATION.md` - Algorithm details, architecture
- `docs/USAGE.md` - Step-by-step training guide
- `CLAUDE.md` - Project context for Claude Code
