import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import extract_features

CLASSES = ["obj_car", "obj_ship"]   

def load(data_dir):
    X, y = [], []
    for idx, name in enumerate(CLASSES):
        for f in sorted((Path(data_dir) / name).iterdir()):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)


def estimators(balanced):
    cw = "balanced" if balanced else None
    est = {
        "SVM (RBF)": SVC(kernel="rbf", class_weight=cw, random_state=42),
        "SVM (linéaire)": SVC(kernel="linear", class_weight=cw, random_state=42),
        "SVM (sigmoïde)": SVC(kernel="sigmoid", class_weight=cw, random_state=42),
        "Arbre de décision": DecisionTreeClassifier(class_weight=cw, random_state=42),
        "Forêt aléatoire": RandomForestClassifier(class_weight=cw, random_state=42),
    }
    if not balanced:  
        est["KNN (k=3)"] = KNeighborsClassifier(n_neighbors=3)
        est["KNN (k=5)"] = KNeighborsClassifier(n_neighbors=5)
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/DB2C")
    args = ap.parse_args()

    X, y = load(args.data_dir)
    print(f"Images : {len(y)} | obj_car = {np.sum(y == 0)} ({np.mean(y == 0) * 100:.1f} %) "
          f"| obj_ship = {np.sum(y == 1)} ({np.mean(y == 1) * 100:.1f} %)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    F_tr = np.array([extract_features(i) for i in X_tr])
    F_te = np.array([extract_features(i) for i in X_te])
    print(f"Test : {len(y_te)} images ({np.sum(y_te == 0)} voitures, {np.sum(y_te == 1)} bateaux)")
    baseline = max(np.bincount(y_te)) / len(y_te)
    print(f"Référence « toujours voiture » : accuracy = {baseline * 100:.2f} %\n")

    rows = []
    for balanced in (False, True):
        for name, clf in estimators(balanced).items():
            model = make_pipeline(StandardScaler(), clf).fit(F_tr, y_tr)
            y_pred = model.predict(F_te)
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
            p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, labels=[1], zero_division=0)
            rows.append({
                "Modèle": name,
                "Version": "class_weight=balanced" if balanced else "standard",
                "Accuracy (%)": accuracy_score(y_te, y_pred) * 100,
                "Balanced acc. (%)": balanced_accuracy_score(y_te, y_pred) * 100,
                "Precision ship (%)": p[0] * 100,
                "Recall ship (%)": r[0] * 100,
                "F1 ship (%)": f[0] * 100,
                "F1 macro (%)": f1_score(y_te, y_pred, average="macro", zero_division=0) * 100,
                "Bateaux ratés (FN)": int(fn),
                "Voitures prises pour bateau (FP)": int(fp),
            })

    df = pd.DataFrame(rows).sort_values(["Version", "Recall ship (%)"], ascending=[False, False])
    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "imbalance_analysis.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(df.round(2).to_string(index=False))
    print(f"\nFichier écrit : {out / 'imbalance_analysis.csv'}")


if __name__ == "__main__":
    main()