# ==============================================
# Lung X-Ray Classification Using SVM
# Steps 1–10 Combined
# ==============================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# -----------------------------
# Step 1: Dataset Exploration
# -----------------------------
dataset_path = '/content/drive/MyDrive/archive (2)/Lung X-Ray Image/Lung X-Ray Image'

classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
print(f"Found {len(classes)} classes: {classes}")
for class_name in classes:
    num_images = len([f for f in os.listdir(os.path.join(dataset_path, class_name)) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  {class_name}: {num_images} images")

# -----------------------------
# Step 2: Load and Preprocess Images
# -----------------------------
def load_images(base_path, img_size=(128,128), max_per_class=800):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    for idx, cls in enumerate(class_names):
        files = [f for f in os.listdir(os.path.join(base_path, cls)) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_per_class]
        for f in files:
            img = cv2.imread(os.path.join(base_path, cls, f))
            if img is None:
                continue
            if len(img.shape)==3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y), class_names

X, y, class_names = load_images(dataset_path)

# -----------------------------
# Step 3: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -----------------------------
# Step 4: Feature Extraction (HOG + LBP)
# -----------------------------
def extract_features(images):
    features = []
    for img in images:
        if img.ndim==3: img = img[:,:,0]
        img_uint8 = (img*255).astype('uint8')
        # HOG
        hog_feat = feature.hog(img_uint8, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')
        # LBP
        lbp = feature.local_binary_pattern(img_uint8, P=8, R=1, method='uniform')
        lbp_hist,_ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
        lbp_hist = lbp_hist / (lbp_hist.sum()+1e-6)
        features.append(np.hstack([hog_feat, lbp_hist]))
    return np.array(features)

X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)

# -----------------------------
# Step 5: PCA
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Original features:", X_train_feat.shape[1])
print("After PCA:", X_train_pca.shape[1])

# -----------------------------
# Step 6: Train SVM
# -----------------------------
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_pca, y_train)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Per-class accuracy
print("Per-Class Accuracy:")
for i, cls in enumerate(class_names):
    acc = cm[i,i]/cm[i].sum()
    print(f"  {cls}: {acc:.3f}")

# -----------------------------
# Step 8: PCA Explained Variance
# -----------------------------
component_importance = pca.explained_variance_ratio_
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.bar(range(1,len(component_importance)+1), component_importance)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Component Importance")
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(np.cumsum(component_importance), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 9: Optional ROC Curve (if probability=True)
# -----------------------------
if hasattr(svm, "predict_proba"):
    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
    y_score = svm.predict_proba(X_test_pca)
    plt.figure(figsize=(6,5))
    colors = ['blue','red','green']
    for i,color in zip(range(len(class_names)),colors):
        fpr,tpr,_ = roc_curve(y_test_bin[:,i], y_score[:,i])
        plt.plot(fpr,tpr,color=color,lw=2,label=f'{class_names[i]} (AUC={auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# -----------------------------
# Step 10: Save the Model
# -----------------------------
model_path = '/content/drive/MyDrive/svm_lung_xray_model.pkl'
joblib.dump({
    'svm_model': svm,
    'scaler': scaler,
    'pca': pca,
    'class_names': class_names,
    'test_accuracy': accuracy
}, model_path)
print(f"✅ Model saved to: {model_path}")
