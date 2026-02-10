"""
Script pour précalculer les sentiments des news
Génère un fichier CSV utilisable pour le backtest
"""
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Ajouter le dossier parent pour importer depuis src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ia.sentiment_analyzer import SentimentAnalyzer


def preprocess_news_sentiment(input_csv, output_csv):
    """
    Analyse les sentiments de toutes les news et sauvegarde les résultats
    
    Args:
        input_csv: Fichier d'entrée avec colonnes [timestamp, text, source]
        output_csv: Fichier de sortie avec colonnes [timestamp, text, sentiment, score, source]
    """
    print("=" * 70)
    print("PRÉTRAITEMENT DES SENTIMENTS POUR BACKTEST")
    print("=" * 70)
    
    # Charge les news
    print(f"\n1. Chargement des news depuis {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   ✓ {len(df)} news chargées")
    
    # Initialise le SentimentAnalyzer en mode LIVE (pas de CSV)
    print("\n2. Initialisation du modèle FinBERT...")
    analyzer = SentimentAnalyzer()
    
    # Analyse chaque news
    print("\n3. Analyse des sentiments...")
    sentiments = []
    scores = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyse en cours"):
        result = analyzer.analyze(row['text'])
        sentiments.append(result['label'])
        scores.append(result['score'])
    
    # Ajoute les résultats au DataFrame
    df['sentiment'] = sentiments
    df['score'] = scores
    
    # Sauvegarde
    print(f"\n4. Sauvegarde des résultats dans {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Statistiques
    print("\n" + "=" * 70)
    print("STATISTIQUES")
    print("=" * 70)
    print(f"Total news analysées : {len(df)}")
    print("\nDistribution des sentiments :")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment:10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nScore moyen par sentiment :")
    for sentiment in df['sentiment'].unique():
        avg_score = df[df['sentiment'] == sentiment]['score'].mean()
        print(f"  {sentiment:10s}: {avg_score:.4f}")
    
    print("\n" + "=" * 70)
    print(f"✅ Fichier généré : {output_csv}")
    print("   Ce fichier peut maintenant être utilisé pour le backtest")
    print("=" * 70)


def main():
    # Chemins des fichiers
    project_root = Path(__file__).parent.parent
    input_csv = project_root / "data" / "sentiment" / "news_example.csv"
    output_csv = project_root / "data" / "sentiment" / "news_sentiment.csv"
    
    # Vérification
    if not input_csv.exists():
        print(f"❌ Erreur : Fichier introuvable {input_csv}")
        print(f"   Créez d'abord un fichier avec les colonnes : timestamp, text, source")
        return
    
    # Traitement
    preprocess_news_sentiment(str(input_csv), str(output_csv))


if __name__ == "__main__":
    main()
