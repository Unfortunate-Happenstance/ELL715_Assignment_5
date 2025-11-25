# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# For Claude: 
- Prioritise brevity and inquisitiveness in discussions with me
- Ask questions where necessary to both yourself and me - don't hesistate to take help if you need a resource

## Project Overview

This repository contains implementations for **ELL715 Assignment 5: Facial Image Analysis** from IIT Delhi. The assignment consists of two parts:

1. **Part 1 (Mandatory - 160 marks)**: Implement Viola-Jones face detector from scratch
2. **Part 2 (Bonus - 100 marks)**: Comparative analysis of face identification algorithms (EigenFaces vs. Wavelets)

**Dataset**: Faces94 dataset located in `Faces94/` directory with three folders:
- `malestaff/` - Training set for Part 1
- `female/` - Training set for Part 1
- `male/` - Testing set for Part 1

For Part 2, use 75% of images from `malestaff` and `female` as gallery, remaining 25% as probes.

## Implementation Requirements

### Part 1: Viola-Jones Face Detector (from scratch)

The implementation must include these components in order:

1. **Dataset Generation** (20 marks)
   - Extract 16×16 patches from training/test images
   - Center patch = 'face' class
   - 5 random patches per image = 'not-a-face' class

2. **Haar Features** (20 marks)
   - Horizontal, vertical, and diagonal Haar filters
   - Multiple scales for all filter types
   - Total feature set will be very large (paper mentions 180,000+ features for 24×24)

3. **Integral Image** (20 marks)
   - Fast Haar feature extraction using integral image representation
   - See `docs/Viola-Jones-Paper.md` Section 2.1 for formulas:
     - `s(x,y) = s(x,y-1) + i(x,y)`
     - `ii(x,y) = ii(x-1,y) + s(x,y)`

4. **AdaBoost Algorithm** (40 marks)
   - Implement from scratch (see `docs/Viola-Jones-Paper.md` Appendix/Table 1 for algorithm)
   - Each weak classifier selects single best feature
   - Weight update: `w_{t+1,i} = w_{t,i} * β_t^{1-e_i}` where `β_t = ε_t/(1-ε_t)`
   - Strong classifier: weighted combination of weak classifiers

5. **Cascade of Classifiers** (20 marks)
   - Arrange classifiers in cascade structure
   - Early stages reject most negative windows quickly
   - Later stages are more complex, process fewer candidates
   - See `docs/Viola-Jones-Paper.md` Section 4 for training procedure

### Part 2: Face Identification (Bonus)

1. **EigenFaces** (40 marks)
   - External PCA packages allowed

2. **Wavelets** (40 marks)
   - Open-ended: use any wavelet configuration (e.g., Gabor wavelets)
   - External packages allowed

3. **Comparison** (20 marks)
   - Compare identification performance
   - Plot t-SNE of gallery feature spaces for separability analysis

## Key Constraints

- **No direct usage of pre-built face detection functions/classes** (except image reading)
- External libraries allowed for sub-functionalities and matrix operations
- Python is recommended language
- Document where AI/agentic coding was used
- Report must follow ACM format

## Report Writing Guide

Your report **must follow ACM format**. This repository includes resources to help you:

### ACM Format Template
- Use `Report/sample-sigplan.tex` as a template example
- This shows the complete structure of an ACM SIGPLAN conference paper

### ACM Formatting Reference
Refer to `docs/ACM_guide.md` for detailed guidance on:

**Document Structure**:
- Use `\documentclass[sigplan]{acmart}` for the document class
- Required preamble elements (copyright, conference info, etc.)

**Authors and Affiliations** (Section 2.3):
- Proper `\author` and `\affiliation` formatting
- Each author gets separate `\author{}` command
- `\institution`, `\city`, and `\country` are mandatory

**Abstract and Keywords**:
- Abstract must come before `\maketitle`
- Keywords using `\keywords{}` command
- CCS concepts (optional for student assignments but good practice)

**Sections and Structure**:
- Standard LaTeX sectioning: `\section`, `\subsection`, `\subsubsection`
- Sections should be numbered (default behavior)

**Figures and Tables** (Section 2.8):
- **Figure captions go AFTER the figure body**
- **Table captions go BEFORE the table body**
- Use `\Description{}` for accessibility
- Scale images: `\textwidth` for single column, `\columnwidth` for two-column figures

**Bibliography**:
- Use BibTeX with `ACM-Reference-Format.bst` style
- Citation commands: `\cite`, `\citep`, `\citeauthor`
- See Section 2.14 in ACM guide for details

**Quick Start**:
```latex
\documentclass[sigplan]{acmart}
\title{Your Title}
\author{Your Name}
\affiliation{%
  \institution{IIT Delhi}
  \city{New Delhi}
  \country{India}}
\begin{abstract}
Your abstract...
\end{abstract}
\maketitle
```

## Critical Paper References

The Viola-Jones paper (`docs/Viola-Jones-Paper.md`) contains essential implementation details:

- **Integral Image computation**: Section 2.1 with recurrence formulas
- **Rectangle feature evaluation**: 4-9 array references depending on type
- **AdaBoost algorithm**: Appendix/Table 1 with complete pseudocode
- **Cascade training**: Section 4.1 - iteratively add stages until target detection/FP rates met
- **Feature normalization**: Variance normalize sub-windows using second integral image of squared values

## Grading Philosophy

**"Grading will entirely be on how much efforts you have put in. We will keep correctness secondary."**

This means:
- Document all implementation attempts and challenges
- Show exploratory analysis and experiments
- Detail the approach even if results aren't perfect
- Explain design decisions and trade-offs
- Include visualizations and intermediate results

## Expected Deliverables

1. Final test accuracy on `male/` folder
2. Face detection on multi-face images from internet
3. Well-documented codebase
4. Informal report in ACM format with all results

For Part 2 (bonus): Identification accuracy, t-SNE plots, comparative analysis
