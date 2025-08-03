# Face Detection System for Digital Cameras

## Project Overview

This repository contains a custom face detection system designed for embedded digital camera applications. The system uses a two-stage cascade approach combining traditional computer vision techniques with machine learning classifiers to efficiently detect faces in images while minimizing computational resources.

## Challenge

As a Data Scientist hired to develop a face detection system for digital cameras, the goal was to create a pipeline that would help technicians automatically optimize camera settings during selfies. The system needed to identify faces in images and return the coordinates of bounding boxes where faces are detected, or an empty list if no faces are present.

## Requirements

- **Input**: An image
- **Output**: A list of bounding box coordinates where faces are detected
- **Constraints**:
  - No pre-trained models allowed (must be trained from scratch using Scikit-learn)
  - Limited computational resources
  - Optimized for embedded systems with power constraints

## Solution Architecture

The implementation uses a two-stage cascade approach:

1. **First Stage (Fast Rejection)**: 
   - Local Binary Patterns (LBP) features
   - RandomForest classifier
   - Quickly filters out obvious non-face regions

2. **Second Stage (Precise Verification)**:
   - Histogram of Oriented Gradients (HOG) features
   - Linear SVM classifier with optional PCA
   - Accurately verifies face candidates that passed the first stage

3. **Post-processing**:
   - Non-Maximum Suppression (NMS) to reduce redundant detections
   - Minimum size filtering to eliminate small false positives

## Performance Highlights

- **Speed**: 2-4× faster than OpenCV's DNN-based face detector
- **Accuracy**: 
  - RandomForest: 88.7% precision, 86.6% recall
  - LinearSVC with PCA: 92.1% precision, 86.2% recall
- **Resource Usage**: ~10MB memory footprint (vs. ~100MB for DNN approaches)
- **Power Efficiency**: Estimated 60-70% reduction in power consumption

## Implementation Details

### Dataset Preparation
- Created from public face datasets
- Split into training (70%), validation (15%), and test (15%) sets
- Face patches extracted using OpenCV's DNN detector for initial labeling
- Multiple negative samples generated from non-face regions

### Feature Extraction
- **LBP**: Uniform patterns with 8 points and radius of 1
- **HOG**: 9 orientations, 8×8 pixel cells, 2×2 cell blocks
- **PCA**: Optional dimensionality reduction to 100 components

### Model Training
- Hyperparameter optimization using RandomizedSearchCV
- Hard-negative mining to improve robustness
- Cross-validation to ensure generalization

### Sliding Window Detection
- Multi-scale approach (70%, 50%, 35%, and 25% of minimum dimension)
- Adaptive step size based on window dimensions
- Optimized threshold combinations for precision/recall balance

## Usage

```python
# Load the trained models
rf_model = joblib.load('saved_models/randomforest_lbp_optimized.pkl')
svm_model = joblib.load('saved_models/linearsvc_pca_hardneg.pkl')

# Detect faces in an image
detections = detect_faces_sliding_window_cascade(
    image, 
    rf_model, 
    svm_model, 
    rf_threshold=0.6, 
    svm_threshold=0.8
)

# Result format: list of (x, y, width, height, confidence_score)
```

## Project Structure

- `face_detection.ipynb`: Main notebook with complete implementation
- `models/`: Folder containing OpenCV DNN model used for initial dataset creation
- `saved_models/`: Folder containing trained Scikit-learn models
- `Selfie_test/`, `Selfie_training/`, `Selfie_validation/`: Image datasets

## Technical Approach

The project followed these key steps:
1. Literature research on lightweight face detection approaches
2. Dataset collection and preparation
3. Feature extraction pipeline development
4. Two-stage classifier training and optimization
5. Sliding window implementation with cascade evaluation
6. Performance evaluation and threshold tuning

## Conclusions

This implementation demonstrates that traditional computer vision techniques, when carefully optimized and combined in a cascade architecture, can provide competitive performance for face detection tasks in embedded camera applications while meeting strict power and resource constraints.

The system achieves a good balance between:
- **Speed**: Fast enough for real-time applications on embedded hardware
- **Accuracy**: Comparable to more complex models in controlled environments
- **Resource usage**: Minimal memory and computational requirements
- **Flexibility**: Adjustable thresholds for different use cases

## References

1. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection.
2. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns.
3. Viola, P., & Jones, M. J. (2004). Robust real-time face detection.
4. Felzenszwalb, P. F., Girshick, R. B., McAllester, D., & Ramanan, D. (2009). Object detection with discriminatively trained part-based models.
5. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.