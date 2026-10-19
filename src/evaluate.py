import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comparison_plot():
    df = pd.read_csv('results/metrics.csv')
    
    df = df.sort_values(by='Accuracy', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Accuracy', y='Modèle', data=df, palette='viridis')
    plt.title("Comparaison des performances des algorithmes")
    plt.xlim(0, 1.0)
    plt.xlabel("Accuracy")
    plt.ylabel("Algorithme")
    
    for index, value in enumerate(df['Accuracy']):
        plt.text(value, index, f' {value:.2%}', va='center')
        
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png')
    plt.show()

if __name__ == "__main__":
    generate_comparison_plot()