# Classification supervisée d'images : Voiture vs Bateau

Atelier de **classification supervisée d'images avec Machine Learning** (Pr. Abdelouahed Sabri).
Chaîne complète : préparation des données → extraction de caractéristiques → apprentissage → prédiction → évaluation → comparaison de 6 algorithmes.

🔗 **Application en ligne :** _(colle ici le lien Streamlit une fois déployé)_

## Données

Dataset **DB2C** : deux classes, `obj_car` (étiquette 0) et `obj_ship` (étiquette 1, classe positive).
Images redimensionnées en 128×128, séparation 80 % / 20 % stratifiée (`random_state=42`).
Voir [`data/README.md`](data/README.md).

## Caractéristiques extraites (59 colonnes)

| Groupe | Colonnes |
|---|---|
| Moyenne et écart-type des canaux R, G, B | 6 |
| Histogramme HSV 8×2×2 | 32 |
| Texture GLCM (contraste, corrélation, énergie, homogénéité) | 4 |
| Texture LBP uniforme | 10 |
| Moments de Hu (échelle log) | 7 |

## Modèles comparés

SVM RBF · SVM linéaire · SVM sigmoïde · KNN (k=3) · KNN (k=5) · Arbre de décision.
Chaque modèle est un pipeline `StandardScaler` + classifieur.

## Résultats

Générés par `src/train.py` dans `results/` (tableau `metrics.csv`, matrices de confusion, rapports de classification, graphique de comparaison).

_(Colle ici le tableau comparatif après ton exécution.)_

## Installation et utilisation

```bash
git clone https://github.com/MDMAK04/classification-images-ml.git
cd classification-images-ml
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 1) Entraîner et évaluer (écrit models/ et results/)
python -m src.train --data-dir /chemin/vers/DB2C

# 2) Lancer l'application
streamlit run app.py
```

Option : `--no-scaler` entraîne sans standardisation (résultats dans `results/sans_scaler/`) pour mesurer l'effet du `StandardScaler`.

## Déploiement (lien pour le prof)

1. Pousser le dépôt sur GitHub **avec** `models/` et `results/` (fichiers légers), sans le dataset.
2. Sur [share.streamlit.io](https://share.streamlit.io) : *New app* → choisir le dépôt → fichier principal `app.py` → *Deploy*.
3. Copier le lien public dans ce README et l'envoyer au prof.

## Structure

```
app.py          application Streamlit
src/            data.py, features.py, models.py, train.py
models/         modèles entraînés (.joblib)
results/        figures, métriques, rapports
notebooks/      notebook du TP
report/         rapport LaTeX
```
