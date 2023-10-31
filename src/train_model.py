import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def extract_hog_features_opencv(image, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Extract HOG features from an image using OpenCV.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    hog = cv2.HOGDescriptor(_winSize=(gray_image.shape[1] // cell_size[1] * cell_size[1],
                                      gray_image.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    features = hog.compute(gray_image)

    return features.flatten()

# ---- Dataset Preparation ----
TRAIN_PATH = r"C:\Users\91636\Documents\Sem5\CV\Project\train"
features_list = []
labels_list = []

for label in os.listdir(TRAIN_PATH):
    label_path = os.path.join(TRAIN_PATH, label)
    if os.path.isdir(label_path):
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path)
            features = extract_hog_features_opencv(image)
            features_list.append(features)
            labels_list.append(label)

features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)

# ---- Train the LinearSVC Classifier ----
svc_classifier = LinearSVC()
svc_classifier.fit(X_train, y_train)

# Evaluate on the validation set
y_pred = svc_classifier.predict(X_val)
print(classification_report(y_val, y_pred))

# Save the trained classifier
joblib.dump(svc_classifier, r"C:\Users\91636\Documents\Sem5\CV\Project\atom_classifier.pkl")
