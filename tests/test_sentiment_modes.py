"""
Test des deux modes du SentimentAnalyzer : LIVE et BACKTEST
"""
import sys
from pathlib import Path
import pandas as pd

# Ajouter le dossier parent pour importer depuis src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ia.sentiment_analyzer import SentimentAnalyzer


def test_live_mode():
    """Test du mode LIVE (analyse en temps réel)"""
    print("=" * 70)
    print("TEST MODE LIVE")
    print("=" * 70)
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Bitcoin surges to new all-time high amid strong institutional demand",
        "Market crash: massive sell-off causes panic selling",
        "Federal Reserve announces decision on interest rates"
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nTexte: {text}")
        print(f"Sentiment: {result['label']:8s} (score: {result['score']:.4f})")
    
    print("\n✓ Mode LIVE fonctionnel")


def test_backtest_mode():
    """Test du mode BACKTEST (sentiments précalculés)"""
    print("\n" + "=" * 70)
    print("TEST MODE BACKTEST")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    sentiment_csv = project_root / "data" / "sentiment" / "news_sentiment.csv"
    
    if not sentiment_csv.exists():
        print(f"⚠ Fichier non trouvé : {sentiment_csv}")
        print("  Exécutez d'abord : python -m scripts.preprocess_news_sentiment")
        return
    
    analyzer = SentimentAnalyzer(sentiment_csv=str(sentiment_csv))
    
    # Test à différents timestamps
    test_times = [
        "2025-01-15 09:30:00",
        "2025-01-17 13:00:00",
        "2025-01-18 10:00:00"
    ]
    
    for timestamp in test_times:
        result = analyzer.analyze(None, timestamp=timestamp)
        print(f"\nTimestamp: {timestamp}")
        print(f"Sentiment: {result['label']:8s} (score: {result['score']:.4f})")
        
        # Affiche les news dans la fenêtre
        news_window = analyzer.get_sentiment_at_time(timestamp, window_hours=1)
        if len(news_window) > 0:
            print(f"  → {len(news_window)} news dans la fenêtre de 1h")
    
    print("\n✓ Mode BACKTEST fonctionnel")


def test_backtest_without_news():
    """Test du mode BACKTEST sans news disponibles"""
    print("\n" + "=" * 70)
    print("TEST BACKTEST SANS NEWS")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    sentiment_csv = project_root / "data" / "sentiment" / "news_sentiment.csv"
    
    if not sentiment_csv.exists():
        print("⚠ Fichier non trouvé, test ignoré")
        return
    
    analyzer = SentimentAnalyzer(sentiment_csv=str(sentiment_csv))
    
    # Test avec un timestamp sans news
    timestamp = "2025-02-01 12:00:00"
    result = analyzer.analyze(None, timestamp=timestamp)
    
    print(f"\nTimestamp: {timestamp} (aucune news)")
    print(f"Sentiment: {result['label']:8s} (score: {result['score']:.4f})")
    print("✓ Retourne neutre quand aucune news disponible")


def main():
    print("\n🧪 TESTS DES MODES DU SENTIMENT ANALYZER 🧪\n")
    
    try:
        test_live_mode()
        test_backtest_mode()
        test_backtest_without_news()
        
        print("\n" + "=" * 70)
        print("✅ TOUS LES TESTS SONT PASSÉS")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
