# 🧪 Optical Chemical Structure Recognition (OCSR)

This repository contains the implementation of a system for recognizing and converting hand-drawn chemical structure images into their machine-readable equivalents. The approach integrates classic computer vision techniques with supervised machine learning for bond classification — tailored to handle the challenges of handwritten molecular diagrams.

## 🎯 Objective

The primary goal is to convert 2D chemical structure images into annotated formats by:
- Recognizing atomic symbols using template matching
- Detecting molecular bond structures
- Classifying bonds (single, double, triple, wedge, dashed) using HOG features + ML classifiers
- Mapping out the molecular graph of the compound

> Unlike existing OCSR systems that perform poorly on hand-drawn inputs, this system is tailored specifically for handwritten molecules.

---

## 📂 Dataset

- **Size:** 360 images  
- **Classes:** 9 unique molecules × 40 samples each  
- **Style:** Hand-drawn using black marker on plain paper  
- **Device:** Captured using iPhone 12 Pro  
- **Preprocessing:**
  - Binarization (threshold 40%)
  - Resized to 400×300 pixels
  - Converted to grayscale
  - No rotation or augmentation applied

---

## 🧠 Methodology

### Pipeline Overview

1. **Text Label Recognition**
   - Scale-invariant template matching for atom detection (O, H, OH, N, etc.)
   - Non-maximum suppression to reduce bounding box overlap

2. **Text Removal**
   - Gaussian smoothing and filtering

3. **Corner Detection**
   - Modified Harris Corner Detector
   - Douglas-Peucker algorithm for polygon simplification

4. **Bond Detection**
   - Hough Transform on sliding window regions between node pairs
   - Heuristics to reduce false positives (e.g., carbon valence limit, 180° bond exclusion)

5. **Bond Classification**
   - HOG features extracted from bond cross-sections
   - Classifiers: SVM (best performer), Logistic Regression, Decision Tree
   - Voting system across sliding windows

---

## 🔬 Tools & Technologies
- Python 3.10

- OpenCV

- Scikit-learn

- Numpy, Matplotlib

- HOG feature extraction

- SVM for bond classification



