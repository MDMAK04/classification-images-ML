from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import extract_features

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
IMG_SIZE = 128

MODELS = {
    "SVM (RBF)": "SVM_RBF",
    "SVM (linéaire)": "SVM_Linear",
    "SVM (sigmoïde)": "SVM_Sigmoid",
    "KNN (k=3)": "KNN_k3",
    "KNN (k=5)": "KNN_k5",
    "Arbre de décision": "Decision_Tree",
    "Random forest": "Random_Forest",
}
CLASSES = {0: ("Car", "🚗"), 1: ("Ship", "🚢")}


RESULTS = pd.DataFrame(
    [
        ("Random forest", 94.90, 95.20, 94.90, 94.56),
        ("SVM (linéaire)", 93.88, 93.88, 93.88, 93.88),
        ("SVM (sigmoïde)", 93.88, 93.83, 93.88, 93.57),
        ("Arbre de décision", 92.86, 92.63, 92.86, 92.60),
        ("KNN (k=5)", 91.84, 91.55, 91.84, 91.43),
        ("SVM (RBF)", 90.82, 91.75, 90.82, 89.53),
        ("KNN (k=3)", 89.80, 89.35, 89.80, 88.96),
    ],
    columns=["Modèle", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"],
)

st.set_page_config(page_title="Car vs Ship — Classification d'images", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model(slug):
    path = MODELS_DIR / f"{slug}.joblib"
    return joblib.load(path) if path.exists() else None


def decode_image(uploaded):
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))


def predict(model, feats):
    pred = int(model.predict(feats)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(feats)[0]
            proba = {int(c): float(v) for c, v in zip(model.classes_, p)}
        except Exception:
            proba = None
    return pred, proba


def confidence_message(conf):
    if conf is None:
        return "Ce modèle ne fournit pas de probabilités : seule la classe prédite est disponible."
    if conf >= 0.90:
        return "Le modèle est **très confiant** : l'image ressemble fortement aux exemples d'une seule classe."
    if conf >= 0.70:
        return "Le modèle est **plutôt confiant**, mais l'autre classe reste possible."
    return "Le modèle **hésite** : l'image est proche des deux classes. Prends ce résultat avec prudence."


st.title("Car vs Ship — Classification d'images")
st.caption("Classification supervisée avec caractéristiques extraites à la main (couleur, texture, forme) "
           "et comparaison de 7 algorithmes de Machine Learning.")

with st.sidebar:
    st.header("Paramètres")
    model_name = st.selectbox("Modèle utilisé", list(MODELS), index=list(MODELS).index("Random forest"))
    st.divider()
    st.markdown("**Projet**  \nAtelier de classification supervisée  \nPr. Abdelouahed Sabri")
    st.markdown("**Dataset** : DB2C  \n🚗 `obj_car` (classe 0)  \n🚢 `obj_ship` (classe 1)")
    st.markdown("**Entrée du modèle** : 59 caractéristiques par image (image redimensionnée en 128×128).")

tab_pred, tab_comp, tab_method = st.tabs(["🔎 Prédiction", "📊 Comparaison des modèles", "🧪 Méthodologie"])

with tab_pred:
    model = load_model(MODELS[model_name])
    if model is None:
        st.error(f"Modèle introuvable : `models/{MODELS[model_name]}.joblib`. "
                 "Lance d'abord l'entraînement (`train.py`) pour générer les modèles.")
        st.stop()

    uploaded = st.file_uploader("Choisis une image de Car ou de Ship (JPG ou PNG)",
                                type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("")
    else:
        rgb, small = decode_image(uploaded)
        if rgb is None:
            st.error("Impossible de lire cette image. Essaie un autre fichier JPG ou PNG.")
            st.stop()

        feats = extract_features(small).reshape(1, -1)
        pred, proba = predict(model, feats)
        label, icon = CLASSES[pred]
        conf = proba[pred] if proba else None

        left, right = st.columns([1, 1.2], gap="large")
        with left:
            with st.container(border=True):
                st.image(rgb, caption=f"Image uploadée ({rgb.shape[1]}×{rgb.shape[0]} px)", width=380)
        with right:
            with st.container(border=True):
                st.caption(f"Modèle : {model_name}")
                st.markdown(f"## {icon} {label}")
                if conf is not None:
                    st.metric("Confiance du modèle", f"{conf * 100:.1f} %")
                    for k, (name, ic) in CLASSES.items():
                        st.progress(proba.get(k, 0.0), text=f"{ic} {name} : {proba.get(k, 0.0) * 100:.1f} %")
                st.write(confidence_message(conf))

        st.subheader("Que disent les 7 modèles sur cette image ?")
        rows, votes = [], []
        for disp, slug in MODELS.items():
            m = load_model(slug)
            if m is None:
                continue
            p, pr = predict(m, feats)
            votes.append(p)
            rows.append({"Modèle": disp, "Prédiction": f"{CLASSES[p][1]} {CLASSES[p][0]}",
                         "Confiance": f"{pr[p] * 100:.1f} %" if pr else "—"})
        if rows:
            n, n_pred = len(votes), votes.count(pred)
            if n_pred == n:
                st.success(f"**Consensus total** : les {n} modèles prédisent {label}.")
            elif n_pred > n / 2:
                st.warning(f"**Majorité** : {n_pred} modèles sur {n} prédisent {label}. "
                           "Les autres ne sont pas d'accord : image possiblement ambiguë.")
            else:
                st.warning(f"Le modèle choisi ({model_name}) est **minoritaire** : "
                           f"seulement {n_pred} modèle(s) sur {n} prédisent {label}.")
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        if feats.shape[1] == 59:
            v = feats[0]
            with st.expander("🔬 Voir les caractéristiques extraites de cette image"):
                st.write("Le modèle ne regarde pas les pixels : il reçoit seulement ces **59 nombres**.")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Couleur RGB** (moyenne / écart-type)")
                    st.dataframe(pd.DataFrame({"Moyenne": v[0:6:2], "Écart-type": v[1:6:2]},
                                              index=["R", "G", "B"]).round(3))
                with c2:
                    st.markdown("**Histogramme HSV** (32 classes)")
                    st.bar_chart(pd.DataFrame({"valeur": v[6:38]}))
                with c3:
                    st.markdown("**Texture LBP** (10 classes)")
                    st.bar_chart(pd.DataFrame({"valeur": v[42:52]}))
                c4, c5 = st.columns(2)
                with c4:
                    st.markdown("**Texture GLCM**")
                    st.dataframe(pd.DataFrame({"valeur": v[38:42]},
                                              index=["contraste", "corrélation", "énergie", "homogénéité"]).round(4))
                with c5:
                    st.markdown("**Moments de Hu** (forme)")
                    st.dataframe(pd.DataFrame({"valeur": v[52:59]},
                                              index=[f"Hu{i}" for i in range(1, 8)]).round(4))

with tab_comp:
    st.subheader("Résultats sur l'ensemble de test")
    best = RESULTS.iloc[RESULTS["Accuracy (%)"].idxmax()]
    worst = RESULTS.iloc[RESULTS["Accuracy (%)"].idxmin()]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Meilleur modèle", best["Modèle"])
    m2.metric("Accuracy", f"{best['Accuracy (%)']:.2f} %")
    m3.metric("Precision", f"{best['Precision (%)']:.2f} %")
    m4.metric("F1-score", f"{best['F1-score (%)']:.2f} %")

    st.dataframe(RESULTS.style.format({c: "{:.2f}" for c in RESULTS.columns[1:]}), hide_index=True)

    st.markdown("#### Comment lire ces résultats")
    st.markdown(
        f"- **{best['Modèle']}** obtient la meilleure accuracy de cette exécution ({best['Accuracy (%)']:.2f} %), "
        f"et **{worst['Modèle']}** la plus faible ({worst['Accuracy (%)']:.2f} %).\n"
        f"- L'écart entre les deux est de **{best['Accuracy (%)'] - worst['Accuracy (%)']:.1f} points**. "
        "Avec un ensemble de test d'environ 100 images, une seule image mal classée pèse environ 1 point : "
        "les petits écarts entre modèles voisins sont donc à interpréter avec prudence.\n"
        "- Les caractéristiques sont les mêmes pour tous les modèles : les différences viennent de la façon dont "
        "chaque algorithme sépare les deux classes (frontière linéaire, noyau, voisins, arbres...).\n"
        "- Les résultats dépendent de la séparation apprentissage/test (graine fixée pour la reproductibilité)."
    )

    with st.expander("Rappel : accuracy, precision, recall, F1-score"):
        st.markdown(
            "- **Accuracy** : proportion d'images correctement classées.\n"
            "- **Precision** : parmi les images prédites dans une classe, proportion qui en font vraiment partie.\n"
            "- **Recall** : parmi les images d'une classe, proportion retrouvée par le modèle.\n"
            "- **F1-score** : moyenne harmonique de la precision et du recall.\n\n"
            "Les valeurs affichées sont des moyennes pondérées sur les deux classes. "
            "Une bonne accuracy ne garantit pas que *chaque* classe soit bien reconnue : "
            "il faut regarder la matrice de confusion."
        )

    st.markdown("#### Matrice de confusion")
    sel = st.selectbox("Modèle", list(MODELS), key="cm_model")
    slug = MODELS[sel].lower()
    found = [p for p in RESULTS_DIR.glob("*.png") if "confusion" in p.name.lower() and slug in p.name.lower()] \
        if RESULTS_DIR.exists() else []
    if found:
        st.image(str(found[0]), width=480)
    else:
        st.info("Aucune matrice de confusion trouvée dans `results/` pour ce modèle.")

with tab_method:
    st.subheader("Chaîne de traitement")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("**1. Image**  \nChargement, conversion RGB, redimensionnement 128×128")
    s2.markdown("**2. Caractéristiques**  \nCouleur, texture et forme : 59 nombres")
    s3.markdown("**3. Modèle**  \nStandardisation puis classifieur (SVM, KNN, arbre, forêt)")
    s4.markdown("**4. Décision**  \nClasse prédite (+ probabilités si disponibles)")

    st.subheader("Caractéristiques extraites")
    st.dataframe(pd.DataFrame(
        [("Statistiques de couleur RGB", "Moyenne et écart-type de chaque canal", 6),
         ("Histogramme HSV", "Teinte / saturation / valeur quantifiées (8×2×2)", 32),
         ("Texture GLCM", "Contraste, corrélation, énergie, homogénéité", 4),
         ("Texture LBP", "Histogramme des motifs binaires locaux (uniformes)", 10),
         ("Moments de Hu", "Descripteurs de forme invariants", 7),
         ("Total", "", 59)],
        columns=["Groupe", "Ce qu'il mesure", "Colonnes"]), hide_index=True)

    st.subheader("Pourquoi des caractéristiques plutôt que les pixels ?")
    st.write("Les vecteurs de caractéristiques résument l'essentiel (couleurs, textures, formes) et suppriment "
             "les détails inutiles. Les données deviennent beaucoup plus compactes, ce qui simplifie "
             "l'apprentissage des algorithmes classiques.")

    st.subheader("Limites")
    st.markdown(
        "- Les caractéristiques manuelles sont moins expressives que celles apprises par un réseau de neurones.\n"
        "- Le dataset est petit et ses images sont de très petite taille : le modèle généralise surtout aux "
        "images qui ressemblent à celles du dataset.\n"
        "- Le modèle ne gère que deux classes (Car, Ship)."
    )
