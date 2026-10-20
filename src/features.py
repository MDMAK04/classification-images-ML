import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def color_moments(img):
    features = []
    for i in range(3):
        canal = img[:, :, i]
        features.extend([np.mean(canal), np.std(canal)])
    vecteur = np.array(features, dtype=float)
    norme = np.linalg.norm(vecteur)
    return vecteur / norme if norme > 0 else vecteur

def hsv_histogram(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_img],
                        [0, 1, 2], 
                        None, 
                        [8, 2, 2], 
                        [0, 180, 0, 256, 0, 256])
    hist_vector = hist.flatten()
    return hist_vector / np.linalg.norm(hist_vector)

def glcm_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_scaled = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(gray_scaled,
                        distances=[1], 
                        angles=[0], 
                        levels=8, 
                        symmetric=True, 
                        normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, correlation, energy, homogeneity])

def lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(7):
        hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]) + 1e-10)
    return hu_moments

def extract_features(img):
    return np.concatenate([
        color_moments(img),    # 6
        hsv_histogram(img),    # 32
        glcm_features(img),    # 4
        lbp_features(img),     # 10
        hu_moments(img)        # 7
    ])

def extract_features_batch(images):
    return np.array([extract_features(img) for img in images])
