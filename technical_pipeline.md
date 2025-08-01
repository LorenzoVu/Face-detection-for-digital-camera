# Project Execution – Face‑Detection Pipeline (LBP → HOG + SVM)

## Overview

The goal is to deliver a **from‑scratch, Scikit‑learn–only** face‑detection engine that:

1. **Detects faces in real time** on a Cortex‑A55–class SoC (< 1 W), providing automatic autofocus/exposure cues during selfies.
2. **Fits the embedded flash budget** (< 1 MB) and runs with minimal RAM.
3. Is **fully explainable** and legally distributable (no pre‑trained CNN weights or third‑party IP restrictions).

To meet these constraints we adopt a **two‑stage cascade**:

- **Stage 1** – *fast rejector*: Local Binary Patterns → Random Forest.
- **Stage 2** – *precise verifier*: Histogram of Oriented Gradients → Linear SVM.

**Why two stages?**\
The camera sensor produces tens of thousands of candidate windows per frame. Running high‑dimensional HOG features and an SVM on *all* of them would blow the real‑time budget. Instead, **Stage 1** applies the ultra‑cheap LBP histogram and a shallow Random Forest to reject \~98 % of pure‑background windows using only integer comparisons. The remaining \~2 % carry the highest probability of containing a face; **Stage 2** then spends the heavier HOG + SVM computation on this small subset, achieving high precision without exceeding the 1 W CPU envelope. This staged pruning is the core trick that makes the pipeline both *fast* and *accurate* on embedded hardware.


All training, validation and benchmarking rely **exclusively on the WIDER FACE dataset** (CC‑BY 4.0). The sections below document the exact data pipeline, feature engineering, model selection, and deployment details.

---

## Technical Pipeline – LBP → HOG + SVM (Two‑Stage Cascade)

> **Scope revision (2025‑08‑01):** External datasets (FDDB, COCO, AFW) have been removed per project directive; only WIDER FACE is used.

### 1 Dataset Construction

| Step             | Action                                                                                                                          | Detail                                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Download**     | `deeplake.load("activeloop/wider-face")` or click 👉 [Deeplake WIDER FACE docs](https://activeloop.ai/docs/datasets/wider-face) | Streams the official Train/Val split.                                                                                                           |
| **Patch mining** | Extract `64 × 64` grayscale windows                                                                                             | *Positives*: boxes with IoU ≥ 0.5 against a WIDER face‑box. *Negatives*: random windows ≥ 200 px away from every box **within the same image**. |
| **Splitting**    | Stratified `train / val / test = 70 / 15 / 15 %`                                                                                | Stratify by WIDER’s 61 event classes to preserve pose/scene diversity.                                                                          |
| **Storage**      | Save as NumPy `.npz`                                                                                                            | `X_train, y_train, X_val, y_val, X_test, y_test`.                                                                                               |

*Rationale:* WIDER covers frontal, profile and occluded faces; Table I of the [WIDER FACE paper](https://arxiv.org/abs/1511.06523) reports ≥ 90 % annotation coverage across poses.

---

### 2 Feature Extraction

```python
# Stage 1 – LBP (uniform, 59 bins)
lbp = skimage.feature.local_binary_pattern(win, P=8, R=1, method="uniform")
feat1, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)

# Stage 2 – HOG (3 780‑dim)
hog = skimage.feature.hog(win,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          orientations=9,
                          feature_vector=True)
```

*(Optional)* Concatenate `feat = np.hstack([feat1, hog])` and compress with **PCA (95 % var.)** if flash constraints tighten.

---

### 3 Model Training

| Stage | Estimator                  | Hyper‑parameter search (`RandomizedSearchCV`, 5‑fold)                         | Selection metric           |
| ----- | -------------------------- | ----------------------------------------------------------------------------- | -------------------------- |
| 1     | `RandomForestClassifier`   | `n_estimators ∈ [50,100,150]`, `max_depth ∈ [4,6]`, `class_weight='balanced'` | **Recall** ≥ 0.98          |
| 2     | `LinearSVC` (`dual=False`) | `C ∈ loguniform(1e‑4, 1)`                                                     | **F1‑score** on validation |

**Hard‑negative mining loop**: RF train → RF inference → collect top‑k FPs → augment negatives → SVM re‑train. Two iterations gave diminishing returns (≤ 0.2 pp F1 gain, log `EXP‑2025‑07‑28`).

---

### 4 Image‑Wide Detection

1. Build scale pyramid `[1.0, 0.8, 0.6, 0.4]`.
2. Slide `64 × 64` window every 16 px (≈ 11 k windows @ 640 × 480).
3. **Stage 1** → RF probability > `thr₁` (0.10) ? keep.
4. **Stage 2** → SVM score > `thr₂` (0.0) ? face candidate.

---

### 5 Post‑processing

- **Non‑Maximum Suppression** with IoU > 0.30.
- Filter detections < `15 × 15` px.

---

### 6 Validation & Testing (WIDER‑only)

|   |
| - |

## Cited Sources

1. **Dalal & Triggs 2005** – *Histograms of Oriented Gradients for Human Detection*. [[PDF\]](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
2. **Ojala et al. 1996** – *A comparative study of texture measures with classification based on featured distributions*. [[IEEE\]](https://ieeexplore.ieee.org/document/541204)
3. **Cortes & Vapnik 1995** – *Support‑Vector Networks*. [[SpringerLink\]](https://link.springer.com/article/10.1007/BF00994018)
4. **Breiman 2001** – *Random Forests*. [[PDF\]](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
5. **Jolliffe 2002** – *Principal Component Analysis*. [[Book\]](https://link.springer.com/book/10.1007/978-1-4757-1904-8)
6. **Yang et al. 2016** – *WIDER FACE: A Face Detection Benchmark*. [[arXiv\]](https://arxiv.org/abs/1511.06523)
7. **Deeplake WIDER FACE dataset card**. [[Web\]](https://activeloop.ai/docs/datasets/wider-face)
8. **Internal Benchmarks 2025‑07‑26 → 2025‑07‑29** – stored in repo `results/`.

