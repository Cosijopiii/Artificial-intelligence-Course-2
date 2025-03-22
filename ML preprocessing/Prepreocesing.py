#%%
from datasets import load_dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#%%


def extract_features(images, radius=3, n_points=24, pca_components=28):
    features = []

    for img in images:
        # Ensure correct dtype and color space
        img = np.array(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Convert to grayscale properly
        if len(img.shape) == 3:  # If color image (H, W, C)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray = img.copy()

        # 1. Normalización CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)

        # 2. LBP
        lbp = local_binary_pattern(normalized, n_points, radius, method='uniform')
        hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

        # 3. Haralick
        glcm = graycomatrix(normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]

        # Combinar características
        features.append(np.concatenate([hist_lbp, [contrast, energy]]))

    # Aplicar PCA
    pca = PCA(n_components=pca_components)
    return pca.fit_transform(np.array(features))


#%%

dataset = load_dataset("marmal88/skin_cancer")  # 10,015 imágenes
def process_medical_images(batch):
    processed = []
    for img in batch["image"]:
        # Preprocesamiento específico para imágenes médicas
        img = np.array(img)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (128, 128))
        resized = np.array(resized)
        processed.append(resized)
    return {"image": processed, "label": batch["dx"]}

skin_dataset = dataset.map(process_medical_images, batched=True)

# Extraer características y entrenar
X_medical = extract_features(dataset["train"]["image"])
y_medical = skin_dataset["train"]["dx"]

X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(X_medical, y_medical, test_size=0.2)

mlp_med = MLPClassifier(hidden_layer_sizes=(50, 25))
mlp_med.fit(X_train_med, y_train_med)
print(classification_report(y_test_med, mlp_med.predict(X_test_med)))