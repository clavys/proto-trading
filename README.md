# 🚀 Road2Million - Trading Bot

Bot de trading algorithmique pour Hyperliquid avec backtesting, optimisation et visualisation.

---

## 📦 Installation

### 1. Créer l'environnement virtuel

**Windows (PowerShell) :**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/MacOS :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu de requirements.txt :**
- pandas, numpy (manipulation de données)
- matplotlib (visualisation)
- hyperliquid-python-sdk (API trading)
- torch, transformers (IA/NLP)

### 3. Installer PyTorch (GPU ou CPU)

**Pour GPU (RTX 4080 SUPER, etc.) :**
```bash
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

**Pour CPU seulement :**
```bash
pip install torch
```

> **Note :** Par défaut, `pip install torch` installe la version CPU. Pour utiliser votre GPU NVIDIA, utilisez la commande GPU ci-dessus.

---

## 🎯 Commandes disponibles

### 📊 Scripts utilitaires

#### Visualiser les données (chandeliers style TradingView)
```bash
python scripts/view_raw_data.py
```
Affiche un graphique en chandeliers avec fond noir et volume pour analyser les données historiques.

#### Télécharger des données
```bash
python scripts/download_data.py
```
Télécharge les données de marché depuis Binance/Hyperliquid (à configurer).

---

### 🧪 Backtesting

#### Tester une stratégie
```bash
python tests/test_backtest.py
```
Lance un backtest sur les données historiques avec la stratégie SMA Crossover.

**Résultats affichés :**
- PnL (profit/perte)
- ROI %
- Nombre de trades
- Win rate
- Max drawdown

---

### ⚙️ Optimisation

#### 1. Lancer une Grid Search
```bash
python tests/run_optimization.py
```
Teste automatiquement des centaines de combinaisons de paramètres et sauvegarde les résultats dans `optimization_results.csv`.

**Paramètres testés :**
- `fast_period` : Période SMA rapide
- `slow_period` : Période SMA lente
- `stop_loss_pct` : Pourcentage de stop loss
- `cooldown` : Période de cooldown entre trades

**Exemple de sortie :**
```
✓ Meilleure config trouvée: ROI = 45.2%
  fast_period: 18
  slow_period: 72
  stop_loss_pct: 0.005
  cooldown: 8
```

#### 2. Analyser les résultats d'optimisation
```python
# Dans un script Python ou notebook
from src.optimization.analyzer import GridSearchAnalyzer

analyzer = GridSearchAnalyzer('optimization_results.csv', target_metric='roi_pct')
analyzer.run_all_analysis()
```

**Analyses générées :**
- **Importance des paramètres** : Quels paramètres ont le plus d'impact
- **Distribution** : Performance moyenne par valeur de paramètre
- **Heatmaps** : Corrélations entre 2 paramètres
- **Insights** : Meilleures valeurs trouvées
- **Recommandations** : Grille affinée pour la prochaine optimisation

**Méthodes disponibles :**
```python
analyzer.plot_parameter_importance()        # Graphique d'importance
analyzer.plot_parameter_distribution()      # Distribution par paramètre
analyzer.plot_heatmap('fast_period', 'slow_period')  # Heatmap 2D
analyzer.get_parameter_insights()           # Insights textuels
analyzer.get_optimization_recommendations() # Recommandations de grille
analyzer.export_recommendations_to_config() # Exporte vers recommended_param_grid.py
```

---

## 🔧 Workflow typique

**Note:** Assurez-vous que l'environnement virtuel est activé (`.\venv\Scripts\Activate.ps1` sur Windows)

### 1️⃣ Visualiser les données
```bash
python scripts/view_raw_data.py
```

### 2️⃣ Tester la stratégie (backtest)
```bash
python tests/test_backtest.py
```

### 3️⃣ Optimiser les paramètres (Grid Search)
```bash
python tests/run_optimization.py
```

### 4️⃣ Analyser les résultats
```python
# Dans un script Python ou en console interactive
from src.optimization.analyzer import GridSearchAnalyzer
analyzer = GridSearchAnalyzer()
analyzer.run_all_analysis()
```

### 5️⃣ Utiliser les paramètres recommandés
Copier le contenu de `recommended_param_grid.py` dans `run_optimization.py` et relancer.

---

## ✨ Fonctionnalités

- ✅ **Visualisation** : Chandeliers style TradingView 
- ✅ **Backtesting** : Test de stratégies sur données historiques
- ✅ **Grid Search** : Optimisation automatique (multiprocessing)
- ✅ **Analyzer** : Analyse approfondie des résultats
- ✅ **Stratégies** : SMA Crossover (+ variations)
- 🚧 **Trading Live** : Hyperliquid testnet/mainnet (en développement)

---

## 🛠️ Problèmes courants

**Module introuvable ?**
```bash
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**utiliser GPU pour ia**
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
