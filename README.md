 Lung X-Ray Classification Using SVM

 Overview
This project implements a machine learning pipeline to classify lung X-ray images
into multiple categories(e.g., Normal, Lung Opacity, Viral Pneumonia) using a **Support Vector Machine (SVM) classifier. 
>> image preprocessing, 
>> feature extraction (HOG + LBP),
>> PCA-based dimensionality reduction,
>> SVM training, 
>> evaluation,
>> visualization, and 
>> model saving.

---

## Dataset
- Source:https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease/data
- Structure:

Lung X-Ray Image/
├─ Lung_Opacity/
├─ Normal/
├─ Viral_Pneumonia/

````
- Images are loaded, converted to grayscale, resized to `128x128`, and normalized.

---

## Steps Implemented

1. Dataset Exploration: Count images per class and check dataset structure.
2. Image Loading & Preprocessing: Grayscale conversion, resizing, normalization.
3. Dataset Split: Train/test split with stratification.
4. Feature Extraction:
   - HOG (Histogram of Oriented Gradients)
   - LBP (Local Binary Patterns)
5. PCA: Dimensionality reduction to retain 95% variance.
6. SVM Training: RBF kernel SVM classifier.
7. Model Evaluation: 
   - Accuracy
   - Confusion matrix
   - Classification report
   - Per-class accuracy
8. Visualization:
   - Confusion matrix
   - PCA explained variance
   - ROC curves (optional, if probability=True)
9. Model Saving: SVM model, scaler, PCA, class names, test accuracy.

---

## Requirements
- Python 3.x
- Libraries:
  - `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`
  - `matplotlib`, `seaborn`
  - `joblib`

Install dependencies:
```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib seaborn joblib
````
---
## Usage

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. Update `dataset_path> https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease/data
3. Run the pipeline:

```bash
python lung_xray_svm.py
```

4. The trained model will be saved as `svm_lung_xray_model.pkl`.

---

## Results

* Test accuracy:  (varies based on dataset)
* Confusion matrix and PCA plots are generated automatically.
* Multi-class ROC curves are shown if `probability=True` in SVM.

---

> Author

1 Yohannes Shiferaw ............43829\14

2 Dawit Belete .................01551\14

