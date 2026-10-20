import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          
ROOT = HERE.parent                              
sys.path.insert(0, str(ROOT))             

from src.features import extract_features 

DEFAULT_DATA_DIR = ROOT / "data" / "DB2C"
DEFAULT_OUT_DIR = HERE / "features"
IMG_SIZE = 128
EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASSES = {"obj_car": ("car_features.csv", 0), "obj_ship": ("ship_features.csv", 1)}

def feature_names(n):
    if n != 59:
        return [f"f_{i:03d}" for i in range(n)]
    names = ["R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std"]
    names += [f"HSV_{i:02d}" for i in range(32)]
    names += ["GLCM_contrast", "GLCM_correlation", "GLCM_energy", "GLCM_homogeneity"]
    names += [f"LBP_{i:02d}" for i in range(10)]
    names += [f"Hu_{i}" for i in range(1, 8)]
    return names


def read_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))


def export(data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for folder, (csv_name, label) in CLASSES.items():
        rows, files = [], []
        for path in sorted((Path(data_dir) / folder).iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            img = read_image(path)
            if img is None:
                print(f"[ignorée] {path.name}")
                continue
            rows.append(extract_features(img))
            files.append(path.name)
        feats = np.array(rows)
        df = pd.DataFrame(feats, columns=feature_names(feats.shape[1]))
        df.insert(0, "label", label)
        df.insert(0, "image", files)
        df.to_csv(out_dir / csv_name, index=False)
        print(f"{folder:<9} -> {out_dir / csv_name}  ({feats.shape[0]} images x {feats.shape[1]} caractéristiques)")


def load_features(folder=DEFAULT_OUT_DIR):
    dfs = [pd.read_csv(Path(folder) / csv_name) for csv_name, _ in CLASSES.values()]
    df = pd.concat(dfs, ignore_index=True)
    return df.drop(columns=["image", "label"]).to_numpy(), df["label"].to_numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    export(args.data_dir, args.out_dir)