# Car vs Ship — Supervised Image Classification with Classical Machine Learning

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-computer%20vision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-demo%20app-red)

An end-to-end supervised image classification pipeline that distinguishes **cars** from **ships** using **handcrafted features** (color, texture, shape) and compares **six classical machine learning models**. A **Streamlit web app** lets anyone upload an image and see the predictions of every model.

**Live demo:** _<paste your Streamlit URL here>_

> Academic project — Supervised Image Classification Workshop (Pr. Abdelouahed Sabri), Master in Machine Learning and Multimedia Intelligence, USMBA, Fès.

---

## Overview

The project covers the full machine learning workflow:

1. **Data preparation** — loading, resizing, stratified train/test split
2. **Feature extraction** — 59 handcrafted features per image
3. **Training** — seven models, each wrapped in a `StandardScaler` pipeline
4. **Evaluation** — confusion matrix, precision, recall, F1-score, accuracy
5. **Comparison** — all models on the same data and the same features
6. **Deployment** — interactive web app for live predictions

## Dataset

The project uses the **DB2C** image dataset (not included in this repository). Two classes:

| Label | Folder | Content |
|---|---|---|
| 0 | `obj_car` | Cars |
| 1 | `obj_ship` | Ships (**positive class** for precision/recall/F1) |

- Images are resized to **128 × 128** and converted to RGB once at load time (OpenCV reads BGR).
- Split: **80 % train / 20 % test**, stratified, `random_state=42`.
- See [`data/README.md`](data/README.md) for the expected folder layout.

## Features

Each image is converted into a **59-dimensional feature vector**:

| Group | Description | Dimensions |
|---|---|---|
| Color statistics | Mean and standard deviation of the R, G, B channels | 6 |
| HSV histogram | 8 × 2 × 2 bins over hue, saturation, value | 32 |
| GLCM texture | Contrast, correlation, energy, homogeneity | 4 |
| LBP texture | Uniform Local Binary Patterns histogram | 10 |
| Hu moments | 7 shape moments (log-scaled) | 7 |

In the feature matrix, **one row = one image** and **one column = one feature**.

## Models

All models are evaluated on identical features and an identical split. Each one is a scikit-learn `Pipeline` (`StandardScaler` → classifier).

| # | Model | Configuration |
|---|---|---|
| 1 | SVM | RBF kernel |
| 2 | SVM | Linear kernel |
| 3 | SVM | Sigmoid kernel |
| 4 | KNN | k = 3 |
| 5 | KNN | k = 5 |
| 6 | Decision Tree | default parameters |
| 7 | Random Forest | 100 trees |


## Model Comparison

Results on the held-out test set (Accuracy, Precision, Recall and F1-Score, weighted averages over both classes) for the 7 models:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **94.90%** | **95.20%** | **94.90%** | **94.56%** |
| SVM (Linear) | 93.88% | 93.88% | 93.88% | 93.88% |
| SVM (Sigmoid) | 93.88% | 93.83% | 93.88% | 93.57% |
| Decision Tree | 92.86% | 92.63% | 92.86% | 92.60% |
| KNN (k=5) | 91.84% | 91.55% | 91.84% | 91.43% |
| SVM (RBF) | 90.82% | 91.75% | 90.82% | 89.53% |
| KNN (k=3) | 89.80% | 89.35% | 89.80% | 88.96% |

**Random Forest** achieves the best performance in this run, with an accuracy of 94.90%.

## Project Structure

```text
.
├── app.py                 # Streamlit web application
├── requirements.txt
├── src/
│   ├── data.py            # loading, resizing, stratified split
│   ├── features.py        # RGB, HSV, GLCM, LBP, Hu feature extraction
│   ├── models.py          # the six model pipelines
│   └── train.py           # training, evaluation, figures, model saving
├── models/                # trained models (.joblib)
├── results/               # metrics, confusion matrices, reports, figures
├── notebooks/             # original workshop notebook
├── data/                  # dataset location (git-ignored)
└── report/                # LaTeX report
```

## Getting Started

### 1. Install

```bash
git clone https://github.com/MDMAK04/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the data

Place the dataset so that it looks like:

```text
DB2C/
├── obj_car/
└── obj_ship/
```

### 3. Train and evaluate

```bash
python -m src.train --data-dir path/to/DB2C
```

Optional: add `--no-scaler` to train without standardization (outputs go to `results/sans_scaler/`) and measure its effect.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Web Application

- Upload a `.jpg` / `.png` image
- Pick any of the six models
- See the predicted class, class probabilities and confidence
- Compare the predictions of all six models on the same image
- Inspect the extracted feature vector, grouped by feature type
- Browse the evaluation results in a dedicated tab

## Notes and Limitations

- Handcrafted features capture global color, texture and shape statistics; they are not as expressive as features learned by deep networks.
- Source images are very small (e.g. 120 × 80), so resizing to 128 × 128 slightly distorts the aspect ratio.
- With a small dataset, results depend on the train/test split; the fixed seed makes them reproducible.

## Author

**Mohammed El Makhloufi** — Master's student in Machine Learning and Multimedia Intelligence, USMBA, Fès, Morocco
[GitHub](https://github.com/MDMAK04) · [LinkedIn](https://linkedin.com/in/mohammed-el-makhloufi)

## License

Educational project, shared for learning purposes.
