# Guide d'Entraînement du Modèle LSTM

## 📋 Table des matières
1. [Démarrage rapide](#démarrage-rapide)
2. [Installation et prérequis](#installation-et-prérequis)
3. [Utilisation du script](#utilisation-du-script)
4. [Paramètres et configurations](#paramètres-et-configurations)
5. [Optimisations de performance](#optimisations-de-performance)
6. [Résolution des problèmes](#résolution-des-problèmes)
7. [Pipeline complet](#pipeline-complet)

---

## 🚀 Démarrage rapide

### Entraînement simple
```bash
cd c:\Users\reppe\vscode_projet\Road2Million
python scripts/train_lstm.py --data data/raw/BTCUSDT-1m-2025-01.csv
```

### Entraînement optimisé (CUDA + Mixed Precision)
```bash
python scripts/train_lstm.py \
  --data data/raw/BTCUSDT-1m-2025-01.csv \
  --device cuda \
  --amp \
  --epochs 100 \
  --batch-size 128
```

---

## 🔧 Installation et prérequis

### 1. Vérifier PyTorch
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Si GPU CUDA (recommandé)
```bash
# Pour CUDA 12.4 (recommandé)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Vérifier l'installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3. Dépendances
```bash
pip install pandas numpy scikit-learn joblib
```

---

## 📝 Utilisation du script

### Syntax de base
```bash
python scripts/train_lstm.py [OPTIONS]
```

### Exemple complet avec tous les paramètres
```bash
python scripts/train_lstm.py \
  --data data/raw/BTCUSDT-1m-2025-03.csv \
  --output-dir models/lstm \
  --epochs 150 \
  --batch-size 64 \
  --seq-length 60 \
  --hidden-size 128 \
  --num-layers 3 \
  --learning-rate 0.0005 \
  --device cuda \
  --amp \
  --warmup-epochs 5 \
  --dropout 0.3
```

---

## ⚙️ Paramètres et configurations

### Paramètres obligatoires
| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `--data` | Chemin vers le CSV | `data/raw/BTCUSDT-1m-2025-01.csv` |

### Paramètres d'entraînement
| Paramètre | Description | Défaut | Recommandé |
|-----------|-------------|--------|-----------|
| `--epochs` | Nombre d'épocas | 50 | 100-150 |
| `--batch-size` | Taille du batch | 32 | 64-256 (GPU) |
| `--learning-rate` | Taux d'apprentissage | 0.001 | 0.0005-0.001 |
| `--seq-length` | Longueur de séquence | 60 | 30-120 |
| `--dropout` | Taux dropout | 0.2 | 0.2-0.5 |

### Paramètres du modèle
| Paramètre | Description | Défaut | Pour gros dataset |
|-----------|-------------|--------|------------------|
| `--hidden-size` | Couche cachée LSTM | 64 | 128-256 |
| `--num-layers` | Nombre couches LSTM | 2 | 2-4 |

### Paramètres système
| Paramètre | Description | Options |
|-----------|-------------|---------|
| `--device` | Device d'entraînement | `cuda` (GPU), `cpu`, `auto` |
| `--amp` | Mixed Precision (⚡ rapide) | Flag (ajouter pour activer) |
| `--output-dir` | Dossier de sortie | Défaut: `models` |
| `--warmup-epochs` | Warmup learning rate | Défaut: 5 |

---

## ⚡ Optimisations de performance

### 1. **Mixed Precision (AMP)** - 🏆 Recommandé
**Accélération: ~2x plus rapide | Mémoire: -50%**

Activer automatiquement:
```bash
python scripts/train_lstm.py --data ... --amp
```

- Compatible CUDA et CPU
- Perte de précision négligeable
- Réduit la mémoire GPU de moitié

### 2. **Batch Size**
**Impact: Plus élevé = plus rapide**

```bash
# CPU: 32-64
# GPU (6GB VRAM): 128-256
# GPU (12GB+ VRAM): 512-1024

python scripts/train_lstm.py --batch-size 256 --data ...
```

### 3. **Learning Rate Scheduling**
Le script utilise un **warmup linéaire** (5 épocas par défaut):
- Épocas 0-4: LR augmente graduellement
- Après: LR décroît avec cosine annealing

```bash
python scripts/train_lstm.py --warmup-epochs 10 --data ...
```

### 4. **Device**
```bash
# Meilleur: GPU (CUDA)
python scripts/train_lstm.py --device cuda --data ...

# Fallback: CPU
python scripts/train_lstm.py --device cpu --data ...

# Auto-detect
python scripts/train_lstm.py --device auto --data ...
```

### 5. **Taille de données**
**Plus petit dataset = plus rapide**

```bash
# 🟢 Petit (entraînement rapide)
--seq-length 30 --batch-size 256

# 🟡 Moyen
--seq-length 60 --batch-size 128

# 🔴 Gros (meilleure qualité, plus lent)
--seq-length 120 --batch-size 32
```

---

## 📊 Temps d'entraînement estimé

### Configuration par machine

#### CPU Moderne (Intel i7/AMD Ryzen 7)
```
Dataset: 100K samples | Epochs: 50
Batch size: 32
⏱️ Temps: 30-60 minutes
```

#### GPU (RTX 3060 / 6GB)
```
Dataset: 100K samples | Epochs: 50
Batch size: 128 | AMP: Activé
⏱️ Temps: 3-5 minutes
```

#### GPU (RTX 4090 / 24GB)
```
Dataset: 100K samples | Epochs: 50
Batch size: 512 | AMP: Activé
⏱️ Temps: <1 minute
```

---

## 📁 Sorties de l'entraînement

Après entraînement, dans le dossier `models/`:

```
models/
├── lstm_btc_20260215_143022.pth           # Poids du modèle
├── lstm_btc_20260215_143022_scaler.pkl    # Normalisation (obligatoire)
└── lstm_btc_20260215_143022_config.json   # Configuration
```

### Fichier config.json
```json
{
  "input_size": 1,
  "hidden_size": 64,
  "num_layers": 2,
  "seq_length": 60,
  "features": ["Close"],
  "test_loss": 0.00234,
  "best_val_loss": 0.00198,
  "trained_on": "data/raw/BTCUSDT-1m-2025-01.csv",
  "timestamp": "20260215_143022"
}
```

---

## 🔄 Pipeline complet

### Étape 1: Entraîner le modèle
```bash
python scripts/train_lstm.py \
  --data data/raw/BTCUSDT-1m-2025-03.csv \
  --epochs 100 \
  --batch-size 128 \
  --device cuda \
  --amp
```

**Output**: 3 fichiers dans `models/`

### Étape 2: Tester le modèle
```python
from src.ia.price_predictor import PricePredictor

# Charger le modèle entraîné
predictor = PricePredictor(
    model_path="models/lstm_btc_20260215_143022.pth",
    scaler_path="models/lstm_btc_20260215_143022_scaler.pkl",
    config_path="models/lstm_btc_20260215_143022_config.json"
)

# Faire des prédictions
prices = predictor.predict(sequence)
```

### Étape 3: Intégrer au bot
```python
from src.ia.price_predictor import PricePredictor
from src.strategies.sma_sentiment import SMASentimentStrategy

# Dans AlgBot.py
self.price_predictor = PricePredictor(
    model_path="models/lstm_btc_latest.pth",
    scaler_path="models/lstm_btc_latest_scaler.pkl",
    config_path="models/lstm_btc_latest_config.json"
)
```

---

## ❓ Résolution des problèmes

### Erreur: "CUDA out of memory"
```bash
# Réduire batch size
python scripts/train_lstm.py --batch-size 32 --device cuda --data ...

# Ou utiliser CPU
python scripts/train_lstm.py --device cpu --data ...

# Ou réduire seq_length
python scripts/train_lstm.py --seq-length 30 --data ...
```

### Erreur: "No module named 'lstm_models'"
```bash
# Vérifier que lstm_models.py existe
ls src/ia/lstm_models.py

# Ajouter au path si nécessaire
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Entraînement très lent (CPU)
```bash
# Installer GPU PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Ou réduire la complexité du modèle
python scripts/train_lstm.py --hidden-size 32 --num-layers 1 --data ...
```

### Loss ne baisse pas
```bash
# Réduire learning rate
python scripts/train_lstm.py --learning-rate 0.0001 --data ...

# Augmenter warmup
python scripts/train_lstm.py --warmup-epochs 10 --data ...

# Vérifier les données
python scripts/view_raw_data.py data/raw/BTCUSDT-1m-2025-01.csv
```

---

## 🎯 Recommandations

### Pour débuter (Testing)
```bash
python scripts/train_lstm.py \
  --data data/raw/BTCUSDT-1m-2025-01.csv \
  --epochs 20 \
  --batch-size 64 \
  --device auto
```

### Production (Haute qualité)
```bash
python scripts/train_lstm.py \
  --data data/raw/BTCUSDT-1m-2025-03.csv \
  --epochs 150 \
  --batch-size 256 \
  --hidden-size 256 \
  --num-layers 3 \
  --device cuda \
  --amp \
  --learning-rate 0.0003 \
  --warmup-epochs 10
```

### Production pour RTX4080s (Haute qualité)
python scripts/train_lstm.py \
  --data data/raw/BTCUSDT-1a-2025.csv \
  --epochs 150 \
  --batch-size 512 \
  --device cuda \
  --amp \
  --hidden-size 256 \
  --num-layers 3 \
  --seq-length 60 \
  --learning-rate 0.0003 \
  --warmup-epochs 10

### Ensemble Models (Meilleure précision)
```bash
# Entraîner 3-5 modèles avec différents seeds
for seed in {1..5}; do
  python scripts/train_lstm.py --data data/raw/BTCUSDT-1m-2025-03.csv --seed $seed
done
```

---

## 📚 Ressources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [LSTM Best Practices](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
