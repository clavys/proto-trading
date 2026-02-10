"""
Test unitaire pour SentimentAnalyzer
Vérifie le bon fonctionnement de l'analyse de sentiment avec FinBERT
"""
import sys
from pathlib import Path

# Ajouter le dossier parent pour importer depuis src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ia.sentiment_analyzer import SentimentAnalyzer


def test_initialization():
    """Test l'initialisation du SentimentAnalyzer"""
    print("=" * 50)
    print("TEST 1: Initialisation du SentimentAnalyzer")
    print("=" * 50)
    
    try:
        analyzer = SentimentAnalyzer()
        print("✓ Initialisation réussie")
        return analyzer
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return None


def test_analyze_positive():
    """Test l'analyse d'un sentiment positif"""
    print("\n" + "=" * 50)
    print("TEST 2: Analyse de sentiment positif")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    text = "Bitcoin price skyrockets as adoption increases. Strong bullish signals."
    
    try:
        result = analyzer.classifier(text)[0]
        print(f"Texte: {text}")
        print(f"Sentiment: {result['label']}")
        print(f"Score: {result['score']:.4f}")
        
        if result['label'] in ['positive', 'POSITIVE']:
            print("✓ Sentiment positif détecté correctement")
        else:
            print(f"⚠ Attendu: positif, Obtenu: {result['label']}")
    except Exception as e:
        print(f"✗ Erreur lors de l'analyse: {e}")


def test_analyze_negative():
    """Test l'analyse d'un sentiment négatif"""
    print("\n" + "=" * 50)
    print("TEST 3: Analyse de sentiment négatif")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    text = "Market crash: massive sell-off causes panic. Bearish trend continues."
    
    try:
        result = analyzer.classifier(text)[0]
        print(f"Texte: {text}")
        print(f"Sentiment: {result['label']}")
        print(f"Score: {result['score']:.4f}")
        
        if result['label'] in ['negative', 'NEGATIVE']:
            print("✓ Sentiment négatif détecté correctement")
        else:
            print(f"⚠ Attendu: négatif, Obtenu: {result['label']}")
    except Exception as e:
        print(f"✗ Erreur lors de l'analyse: {e}")


def test_analyze_neutral():
    """Test l'analyse d'un sentiment neutre"""
    print("\n" + "=" * 50)
    print("TEST 4: Analyse de sentiment neutre")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    text = "The FED will announce its decision tomorrow. Market awaits."
    
    try:
        result = analyzer.classifier(text)[0]
        print(f"Texte: {text}")
        print(f"Sentiment: {result['label']}")
        print(f"Score: {result['score']:.4f}")
        
        if result['label'] in ['neutral', 'NEUTRAL']:
            print("✓ Sentiment neutre détecté correctement")
        else:
            print(f"⚠ Attendu: neutre, Obtenu: {result['label']}")
    except Exception as e:
        print(f"✗ Erreur lors de l'analyse: {e}")


def main():
    """Lance tous les tests"""
    print("\n🤖 TESTS DU SENTIMENT ANALYZER 🤖\n")
    
    # Test d'initialisation
    analyzer = test_initialization()
    
    if analyzer is None:
        print("\n❌ Échec de l'initialisation, arrêt des tests")
        return
    
    # Tests d'analyse
    test_analyze_positive()
    test_analyze_negative()
    test_analyze_neutral()
    
    print("\n" + "=" * 50)
    print("✅ Tous les tests sont terminés")
    print("=" * 50)


if __name__ == "__main__":
    main()
