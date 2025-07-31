# Face Detection System for a Compact Digital Camera

## Project Context
This project is developed for **ProCam S.p.A.**, a company preparing to launch a new compact digital camera aimed at young photography enthusiasts. The system is intended to facilitate selfie-taking by automatically detecting faces in the frame and returning the coordinates of the bounding boxes surrounding the detected faces.

## Objective
To build a face detection pipeline from scratch (without pre-trained models), using only **Scikit-learn**. The system must:
- Accept an input image
- Return a list of bounding boxes where faces are detected
- Return an empty list if no faces are present

## Constraints
- Pre-trained models (e.g., OpenCV Haar cascades, deep learning models) are **not allowed**
- Only classic machine learning models trainable from scratch using **Scikit-learn**
- Limited computational resources (the model must be lightweight and fast)
- No dataset is provided: the developer must find and preprocess a suitable dataset
- The solution must be **well-documented**, with justification for all technical choices

## Dataset
The selected dataset is **WIDER FACE**, accessed via [Deeplake](https://activeloop.ai/docs/datasets/wider-face/).

- Includes `train`, `validation`, and `test` splits (~32,000 images, ~393,000 faces)
- Bounding boxes are provided for each image (no manual parsing needed)
- Each image will be converted into fixed-size patches (`64x64` pixels):
  - **Positive samples**: cropped face regions
  - **Negative samples**: random background regions with no faces

---

## Technical Pipeline

### 1. Dataset Construction
- Load the dataset via **Deeplake**
- Automatically extract `64x64` patches:
  - Positive: from face bounding boxes
  - Negative: random regions that do not overlap faces
- Build `X_train`, `y_train`, `X_val`, `y_val` as NumPy arrays

### 2. Feature Extraction
Use classic handcrafted feature extractors:
- **HOG (Histogram of Oriented Gradients)**  
  📖 [Dalal & Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
- **LBP (Local Binary Patterns)**  
  📖 [Ojala et al., 1996](https://ieeexplore.ieee.org/document/541204)
- Optionally: combine HOG + LBP
- Optional: PCA for dimensionality reduction  
  📖 [Jolliffe, 2002 – Principal Component Analysis](https://link.springer.com/book/10.1007/978-1-4757-1904-8)

### 3. Model Training
Candidate classifiers:
- **SVM (Support Vector Machine)**  
  📖 [Cortes & Vapnik, 1995](https://link.springer.com/article/10.1007/BF00994018)
- **Random Forest**  
  📖 [Breiman, 2001](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- **SGDClassifier (Stochastic Gradient Descent)**  
  📖 [Bottou, 2010](https://leon.bottou.org/papers/bottou-2010)

Task:
- Perform binary classification (face vs. non-face)
- Use cross-validation and F1-score to evaluate models

### 3.5 Hyperparameter Optimization (Randomized Search)
To tune the hyperparameters of the selected models, the pipeline will use [**RandomizedSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) from Scikit-learn. This method explores a randomized set of hyperparameter combinations, making it more efficient than exhaustive grid search.

- Search space will include parameters like:
  - `C`, `gamma`, `kernel` for SVM
  - `n_estimators`, `max_depth` for Random Forest
  - `alpha`, `penalty` for SGDClassifier
- Cross-validation (e.g., 5-fold) will be used to estimate performance
- The model with the highest validation F1-score will be selected

### 4. Image-Wide Detection
- Apply **sliding window** across the full image
- Use **multiscale analysis** (rescale image and repeat detection)
- Classify each patch; if predicted as a face, save bounding box

### 5. Post-processing
- Apply **Non-Maximum Suppression (NMS)** to remove overlapping boxes
- Return the final list of face bounding boxes

### 6. Validation and Testing
- Test the system on new images (with and without faces)
- Evaluate using metrics: **precision**, **recall**, **F1-score**, **IoU**
- Visualize bounding boxes overlaid on images

---

## Technical Notes
- Images will be converted to grayscale and resized to `64x64`
- HOG features via [`skimage.feature.hog`](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
- LBP via [`skimage.feature.local_binary_pattern`](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html)
- All classifiers implemented with [Scikit-learn](https://scikit-learn.org/stable/)
- Final model exportable with [`joblib`](https://scikit-learn.org/stable/modules/model_persistence.html)

---

## Cited Sources
- [Dalal & Triggs, 2005 – HOG](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
- [Ojala et al., 1996 – LBP](https://ieeexplore.ieee.org/document/541204)
- [Viola & Jones, 2001 – Haar features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- [Cortes & Vapnik, 1995 – SVM](https://link.springer.com/article/10.1007/BF00994018)
- [Breiman, 2001 – Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Bottou, 2010 – SGD](https://leon.bottou.org/papers/bottou-2010)
- [Jolliffe, 2002 – PCA](https://link.springer.com/book/10.1007/978-1-4757-1904-8)
- [WIDER FACE via Deeplake](https://activeloop.ai/docs/datasets/wider-face/)

---

## Next Steps
- Extract face and non-face patches from WIDER FACE via Deeplake
- Implement `extract_features(img)` using HOG (+ optional LBP)
- Train and optimize the classifier (RandomizedSearchCV)
- Build and evaluate the full detection pipeline (sliding window + NMS)
