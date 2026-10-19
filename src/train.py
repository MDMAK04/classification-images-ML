import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_data, split_data
from features import extract_features_batch
from models import get_models

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    X, y = load_data(data_dir='data/DB2C')
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Extraction des features (Train)...")
    X_train_feat = extract_features_batch(X_train)
    print("Extraction des features (Test)...")
    X_test_feat = extract_features_batch(X_test)

    models = get_models()
    results = []

    for name, pipeline in models.items():
        print(f"\nEntraînement de {name}...")
        pipeline.fit(X_train_feat, y_train)
        
        joblib.dump(pipeline, f'models/{name}.joblib')
        
        y_pred = pipeline.predict(X_test_feat)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Modèle": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4)
        })
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Car (0)', 'Ship (1)'], yticklabels=['Car (0)', 'Ship (1)'])
        plt.title(f'Matrice de Confusion - {name}')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        
        plt.savefig(f'results/confusion_matrix_{name}.png')
        plt.close() 

    df_results = pd.DataFrame(results)
    df_results.to_csv('results/metrics.csv', index=False)
    
    print("\n" + "="*50)
    print("--- RÉSULTATS FINAUX ---")
    print("="*50)
    print(df_results.to_string(index=False))
    print("\nToutes les métriques et matrices de confusion ont été sauvegardées dans 'results/'")

if __name__ == "__main__":
    main()