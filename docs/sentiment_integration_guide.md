# Guide Complet : Intégrer SentimentAnalyzer dans test_backtest.py

## 🎯 Objectif
Utiliser l'analyse de sentiment FinBERT pour filtrer les signaux de trading en backtest.

---

## 📋 Prérequis

1. **Environnement Python configuré** avec PyTorch 2.6+
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Données de prix** dans `data/raw/` (ex: BTCUSDT-1m-2025-01.csv)

3. **Données de news** pour créer le CSV de sentiments

---

## 🚀 Étape 1 : Créer les données de news

### Fichier : `data/sentiment/news_example.csv`

Format requis : `timestamp,text,source`

```csv
2025-01-15 09:00:00,"Bitcoin breaks $50000 resistance level",twitter
2025-01-15 10:30:00,"Institutional investors announce purchases",news
2025-01-15 14:00:00,"Market shows bullish momentum",twitter
2025-01-16 08:00:00,"Slight pullback in Bitcoin price",news
2025-01-17 11:00:00,"Market concerns over regulatory news",news
2025-01-17 13:30:00,"Bitcoin drops below $49000",twitter
2025-01-18 10:00:00,"Technical indicators suggest reversal",twitter
```

**Important** : Les timestamps doivent être dans la même plage que vos données de prix !

---

## 🔄 Étape 2 : Générer le CSV de sentiments

Exécutez le script de prétraitement :

```bash
python -m scripts.preprocess_news_sentiment
```

**Résultat** : Génère `data/sentiment/news_sentiment.csv` avec colonnes :
```
timestamp,text,sentiment,score,source
2025-01-15 09:00:00,"Bitcoin breaks $50000...",positive,0.8309,twitter
```

---

## 💡 Étape 3 : Utiliser dans test_backtest.py

### Code complet modifié

```python
# tests/test_backtest.py
import pandas as pd
from src.core.data.handler import DataHandler
from src.strategies.sma_sentiment import SMASentimentStrategy  # ✓ Import du sentiment
from src.utils.backtest import Backtester
from src.utils.visualizer import plot_backtest_results

def run_simulation():
    # 1. Charger le fichier CSV
    path = "data/raw/BTCUSDT-1m-2025-01.csv"
    raw_data = pd.read_csv(path, header=None)
    
    # 2. Transformer les données au format standard
    data = DataHandler.normalize_binance_klines(raw_data)
    
    # 3. ✨ STRATÉGIE AVEC SENTIMENT
    strategy = SMASentimentStrategy(
        fast_period=19,
        slow_period=72,
        stop_loss_pct=0.006,
        cooldown=6,
        sentiment_csv="data/sentiment/news_sentiment.csv",  # ✓ Active le sentiment
        sentiment_threshold=0.7,                             # ✓ Score minimum
        verbose=False
    )
    
    # 4. Lancer le backtester (AUCUNE MODIFICATION NÉCESSAIRE)
    backtester = Backtester(strategy=strategy, initial_balance=1000, fee=0.0001)
    results = backtester.run(data, metadata={"symbol": "BTCUSDT"})
    
    # 5. Afficher les résultats
    print(f"--- Rapport de Simulation ---")
    print(f"Période : {data['timestamp'].min()} à {data['timestamp'].max()}")
    print(f"Solde final : {results['final_balance']:.2f} USDT")
    print(f"Nombre de trades : {results['num_trades']}")
    print(f"Win Rate : {results['win_rate_pct']:.2f}%")
    print(f"ROI : {results['roi_pct']:.2f}%")
    print(f"Max Drawdown : {results['max_drawdown_pct']:.2f}%")
    
    # 6. Visualiser
    plot_backtest_results(
        data, 
        results['trades'],
        results['equity_curve']
    )

if __name__ == "__main__":
    run_simulation()
```

---

## ⚙️ Paramètres de SMASentimentStrategy

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `fast_period` | int | 29 | Période de la SMA rapide |
| `slow_period` | int | 132 | Période de la SMA lente |
| `cooldown` | int | 15 | Bougies entre deux signaux |
| `stop_loss_pct` | float | 0.0 | Stop loss en % (0.02 = 2%) |
| `sentiment_csv` | str | None | Chemin vers CSV ou None pour LIVE |
| `sentiment_threshold` | float | 0.7 | Score minimum (0.0-1.0) |
| `verbose` | bool | False | Active les logs de debug |

---

## 🔀 Comparaison : Avec/Sans Sentiment

### Code pour tester les deux

```python
from src.strategies.sma_crossover import SMACrossStrategyReverse

# SANS SENTIMENT (baseline)
strategy_without = SMACrossStrategyReverse(
    fast_period=19, 
    slow_period=72, 
    stop_loss_pct=0.006, 
    cooldown=6, 
    verbose=False
)

# AVEC SENTIMENT
strategy_with = SMASentimentStrategy(
    fast_period=19,
    slow_period=72,
    stop_loss_pct=0.006,
    cooldown=6,
    sentiment_csv="data/sentiment/news_sentiment.csv",
    sentiment_threshold=0.7,
    verbose=False
)

# Tester les deux
for name, strategy in [("SANS Sentiment", strategy_without), 
                       ("AVEC Sentiment", strategy_with)]:
    backtester = Backtester(strategy=strategy, initial_balance=1000, fee=0.0001)
    results = backtester.run(data, metadata={"symbol": "BTCUSDT"})
    
    print(f"\n{name}:")
    print(f"  Trades: {results['num_trades']}")
    print(f"  ROI: {results['roi_pct']:.2f}%")
    print(f"  Win Rate: {results['win_rate_pct']:.1f}%")
```

---

## 🎓 Fonctionnement du SentimentAnalyzer

### Mode BACKTEST
```
1. Récupère le timestamp de la bougie actuelle
2. Cherche les news dans une fenêtre de ±1h
3. Calcule le sentiment dominant pondéré par les scores
4. Retourne {'label': 'positive/negative/neutral', 'score': float}
```

### Mode LIVE
```
1. Reçoit un texte de news (via API ou événement)
2. Analyse en temps réel avec le modèle FinBERT
3. Retourne le sentiment du texte
```

### Si pas de news disponible
```
Retourne {'label': 'neutral', 'score': 0.5}
Signal technique accepté par défaut
```

---

## 📊 Exemple de résultats

```
SANS Sentiment:
  Trades: 378
  ROI: +26.64%
  Win Rate: 63.5%
  Max Drawdown: 8.55%

AVEC Sentiment:
  Trades: 5
  ROI: -4.39%
  Win Rate: 20.0%
  Max Drawdown: 5.07%
```

⚠️ **Note** : Les résultats dépendent de la couverture des données de news. 
Avec peu de news, le sentiment réduira les opportunités de trading.

---

## ✅ Checklist

- [ ] Créer `data/sentiment/news_example.csv` avec vos données
- [ ] Exécuter `python -m scripts.preprocess_news_sentiment`
- [ ] Vérifier que `data/sentiment/news_sentiment.csv` a été généré
- [ ] Modifier `test_backtest.py` selon l'exemple ci-dessus
- [ ] Lancer : `python -m tests.test_backtest`

---

## 🔗 Fichiers concernés

- **Stratégie** : [src/strategies/sma_sentiment.py](../../src/strategies/sma_sentiment.py)
- **Analyzer** : [src/ia/sentiment_analyzer.py](../../src/ia/sentiment_analyzer.py)
- **Script de génération** : [scripts/preprocess_news_sentiment.py](../../scripts/preprocess_news_sentiment.py)
- **Tests** : [tests/test_sentiment_modes.py](../../tests/test_sentiment_modes.py)
- **Données** : [data/sentiment/](../../data/sentiment/)
