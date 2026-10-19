"""Application Streamlit : upload d'une image -> prédiction Voiture / Bateau."""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.data import CLASSES, IMG_SIZE, LABELS_FR
from src.features import FEATURE_GROUPS, extract_features
from src.models import MODEL_FILES

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

st.set_page_config(page_title="Classification d'images : Voiture vs Bateau", page_icon="🖼️", layout="wide")


@st.cache_resource
def load_model(slug):
    return joblib.load(MODELS / f"{slug}.joblib")


def preprocess(file):
    """Fichier uploadé -> image RGB 128x128 (même prétraitement qu'à l'entraînement)."""
    img = Image.open(file).convert("RGB")
    return cv2.resize(np.array(img), (IMG_SIZE, IMG_SIZE))


st.title("Classification supervisée d'images : Voiture vs Bateau")
st.caption("Atelier de classification supervisée — Pr. Abdelouahed Sabri | Dataset DB2C | "
           "caractéristiques : RGB, HSV, GLCM, LBP, moments de Hu")

if not (MODELS / f"{MODEL_FILES['SVM RBF']}.joblib").exists():
    st.error("Modèles introuvables : lance d'abord `python -m src.train --data-dir <DB2C>`.")
    st.stop()

tab_pred, tab_res = st.tabs(["🔎 Prédiction", "📊 Résultats du TP"])

# ---------------------------------------------------------------- Prédiction
with tab_pred:
    model_name = st.selectbox("Modèle", list(MODEL_FILES), index=0)
    file = st.file_uploader("Choisis une image (jpg, png)", type=["jpg", "jpeg", "png", "bmp"])

    if file is not None:
        img = preprocess(file)
        feats = extract_features(img)
        model = load_model(MODEL_FILES[model_name])
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        pred = int(np.argmax(proba))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption=f"Image redimensionnée en {IMG_SIZE}×{IMG_SIZE}", width=320)
        with c2:
            st.subheader(f"Prédiction : {LABELS_FR[CLASSES[pred]]} ({CLASSES[pred]})")
            st.metric("Confiance", f"{proba[pred] * 100:.1f} %")
            st.bar_chart(pd.DataFrame({"probabilité": proba}, index=[LABELS_FR[c] for c in CLASSES]))

        st.markdown("#### Les 6 modèles sur cette image")
        rows = []
        for name, slug in MODEL_FILES.items():
            p = load_model(slug).predict_proba(feats.reshape(1, -1))[0]
            k = int(np.argmax(p))
            rows.append({"Modèle": name, "Classe prédite": CLASSES[k], "Confiance": f"{p[k] * 100:.1f} %"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        with st.expander("Voir le vecteur de caractéristiques extrait"):
            start = 0
            for group, n in FEATURE_GROUPS:
                st.write(f"**{group}** ({n} valeurs)")
                st.code(np.array2string(feats[start:start + n], precision=4, max_line_width=100))
                start += n
    else:
        st.info("Upload une image pour lancer la prédiction.")

# ------------------------------------------------------------------ Résultats
with tab_res:
    metrics = RESULTS / "metrics.csv"
    if not metrics.exists():
        st.warning("Résultats non générés (lance `python -m src.train`).")
    else:
        st.markdown("#### Tableau comparatif (ensemble de test, classe positive = obj_ship)")
        df = pd.read_csv(metrics)
        st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "Modèle"}),
                     hide_index=True, width="stretch")
        if (RESULTS / "comparaison_accuracy.png").exists():
            st.image(str(RESULTS / "comparaison_accuracy.png"))
        sel = st.selectbox("Détails d'un modèle", list(MODEL_FILES), key="detail")
        slug = MODEL_FILES[sel]
        a, b = st.columns(2)
        if (RESULTS / f"confusion_{slug}.png").exists():
            a.image(str(RESULTS / f"confusion_{slug}.png"))
        if (RESULTS / f"rapport_{slug}.txt").exists():
            b.code((RESULTS / f"rapport_{slug}.txt").read_text(encoding="utf-8"))
