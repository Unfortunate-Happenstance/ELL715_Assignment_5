# ELL715: Assignment-5
## Facial Image Analysis

**Instructor:** Prof. Monika Agrawal  
**Professor, C.A.R.E., Indian Institute of Technology Delhi**

**Total Marks:** 260 marks (160 + 100 bonus)  
**Deadline:** 21st November 2025  
**Date Issued:** October 29, 2025

---

## Instructions

- This assignment can be done in teams. Your team should comprise of ≤ smallest even natural number of members (i.e., maximum 2 members).

- You may use agents for partial coding and reference. You'll have to specify at what all places agentic coding has been employed.

- You'll have to submit the code and a report following the ACM format. Your code repository and reported results should be reproducible. You'll have to attach a README file which should have instructions to run the code.

- Both report and code has to be zipped inside a zip file. The zip file should be named as per the following scheme:  
  `<2022EE11436>-<NAKSHAT PANDEY>`
  

- Python is recommended language for implementation. You are free to utilize any open source package/library of your choice.

- If you have any queries, you can meet the TAs in their office hours.

- We will conduct demos for grading.

- We shall run moss for plagiarism check. Based upon the amount of plagiarism adequate penalty will be applied.

- **Grading will entirely be on how much efforts you have put in. We will keep correctness secondary.**

---

## Report Guide

For the report, follow the ACM format as specified. Refer to `ACM_guide.md` for detailed instructions on using the ACM LaTeX class.

**Known Issues:**
- Overfull hbox warnings: These occur when text is too wide for the line. To fix, you can break long lines, use smaller fonts like `\footnotesize` for code or paths, or adjust spacing. Document any such issues in your report.

---

## 1. Introduction

This assignment has two parts, the first one is mandatory and the second is bonus. In the first part you are required to implement Viola Jones algorithm from scratch for face detection. While in the second part, you are required to present a comparative analysis between two face identification algorithms.

**Dataset:** You should use Faces94 dataset.

**Viola Jones Paper:** `ELL715_Assignment_5\docs\Viola-Jones-Paper.md`

**Link to Faces94 Dataset:** Dataset

---

## 2. Part-1: Viola-Jones Face Detector (160 marks)

You are supposed to implement the famous Viola-Jones face detector from scratch. You can use external libraries for implementing sub-functionalities and matrix manipulations, but direct usage of functions/classes is strictly prohibited (except for reading the images). 

### Implementation Requirements:

### 1. Dataset Generation (20 marks)

There are three folders located in `Faces94` directory in the dataset, use the `maleStaff` and `female` for training Viola-Jones classifier, and `male` for testing. 

**For ground truths:**
- You can select 16 × 16 patch from the center of the image. This will serve as the 'face' class.
- Then, you can extract 5 other 16 × 16 random patches from the image. These will be tagged as 'not-a-face' class.
- Repeat this process for all the images in the training and testing set.

### 2. Haar Features (20 marks)

Next, Haar features are to be extracted. For this consider:
- Horizontal Haar filters
- Vertical Haar filters
- Diagonal Haar filters
- Multiple scales for all filter types

### 3. Integral Image (20 marks)

To extract Haar features quickly, you should implement Integral image based feature extraction. Refer the Viola Jones paper for more details.

### 4. Adaboost Algorithm (40 marks)

Next to classify an image as 'face' or 'not-a-face' implement Adaboost algorithm from scratch.

### 5. Cascade of Classifiers (20 marks)

Finally, arrange these classifiers in a cascade, as described in the original paper.

### Deliverables (40 marks):

1. Final test accuracy.

2. Face detection results on a couple of images with multiple faces. You can use images from internet.

3. A well documented codebase and an informal report. Put up all the results in the report.

---

## 3. Part-2 (Bonus): Face Identification Results (100 marks)

This is a bonus part. For this part, you should use initial 75% images for each subject in `maleStaff` & `female` folders as the gallery. Remaining 25% images per subject are to be used as probes. 

### After the data split, implement the following:

### 1. EigenFaces (40 marks)

Implement the EigenFaces algorithm. You are free to use any external package for PCA.

### 2. Wavelets (40 marks)

This part is open ended. Use different wavelets to construct a feature vector per image. You are free to use any wavelet configuration, and can implement them with any external package. An example of good texture based wavelets are Gabor wavelets.

### 3. Comparison and Analysis (20 marks)

Finally, compare the identification performance of the two methods. Try plotting tSNE plots of the gallery feature spaces so as to understand subject level separability.

---

## Final Note

**As mentioned in the instructions, grading will first account for efforts, and then correctness! Hence detail the efforts well in the report**

**All the best!**