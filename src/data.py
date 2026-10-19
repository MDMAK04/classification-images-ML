import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir='data/DB2C', img_size=(128, 128)):

    classes = ['obj_car', 'obj_ship']
    images = []
    labels = []

    print("Chargement des images...")
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Dossier introuvable : {class_dir}")
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label_idx) # 0 pour car, 1 pour ship

    X = np.array(images)
    y = np.array(labels)
    
    print(f"Total d'images chargées : {len(X)}")
    print(f"Car (0) : {np.sum(y == 0)} | Ship (1) : {np.sum(y == 1)}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test