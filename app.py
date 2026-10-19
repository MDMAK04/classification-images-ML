import streamlit as st
import cv2
import numpy as np
import joblib
from src.features import extract_features

st.set_page_config(page_title="Car vs Ship Classifier", layout="wide")

st.title("Classification : Car vs Ship ")
st.markdown("Ce projet compare 7 algorithmes de Machine Learning pour classifier des images.")

# Sidebar
st.sidebar.header("Paramètres")
model_choice = st.sidebar.selectbox(
    "Choisissez le modèle :",
    ("SVM_RBF", "SVM_Linear", "SVM_Sigmoid", "KNN_k3", "KNN_k5", "Decision_Tree", "Random_Forest")
)

@st.cache_resource
def load_model(model_name):
    return joblib.load(f'models/{model_name}.joblib')

try:
    model = load_model(model_choice)
except Exception as e:
    st.error(f"Erreur de chargement. Avez-vous lancé `python src/train.py` ? Erreur : {e}")
    st.stop()

uploaded_file = st.file_uploader("Uploadez une image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Image uploadée", use_column_width=True)
    
    if st.button("Prédire"):
        features = extract_features(img_resized).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        classes = ["Car ", "Ship"]
        
        with col2:
            st.success(f"**Modèle utilisé :** {model_choice}")
            st.markdown(f"### Prédiction : **{classes[prediction]}**")
