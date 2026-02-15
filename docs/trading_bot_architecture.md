#  Structure du projet

---

##  Arborescence du projet

```text
hyperliquid-trading-bot/
│
├── src/
│   ├── core/                            # Logique principale du bot
|   |   ├── signal.py                    # Contrat de communication entre les stratégies et le bot
│   │   ├── bot.py                       # Exécute le loop principal
│   │   ├── exchange.py                  # Interface avec l’API Hyperliquid (Info / Exchange)
│   │   ├── strategy_manager.py          # Gère plusieurs stratégies et les signaux combinés
│   │   ├── risk_manager.py              # Valide chaque signal pour vérifier s'il respecte les limites de levier, la taille maximale de position et l'exposition globale du capital.
│   │   ├── OrderManager                 # Gère le cycle de vie des ordres (suivi des exécutions partielles, annulation des ordres expirés, gestion des Stop-Loss et Take-Profit)
│   │   ├── position_tracker.py          # Suit l'état des positions ouvertes en direct (prix d'entrée, taille actuelle, PnL non réalisé)
│   │   └── data/
│   │        ├── handler.py              # Transforme les données brutes (venant de fichiers CSV, de l'API Hyperliquid ou d'autres échanges) en un format standardisé
│   │        ├── providers.py            # Regroupe les différentes sources d'accès aux données historiques (API REST, Fichiers locaux .csv, DB lecture/écriture).
│   │        ├── recorder.py             # Archiviste (Écriture CSV/Fichiers).
│   │        └── live_data_provider.py   # Maintient un canal de communication permanent avec les exchanges (WebSocket - Réception uniquement)
│   │ 
│   ├── strategies/                      # Répertoire des stratégies de trading
│   │   ├── sma_crossover.py             # Stratégie moyenne mobile simple
│   │   ├── rsi_reversal.py              # 
│   │   ├── macd_trend.py                # 
│   │   └── trend_matrix.py              # 
│   │
│   ├── optimization/                    #
│   │   ├── grid_search.py               #
│   │   ├── analyzer.py                  #
│   │   └── walk_forward.py              # Potentiel ajout
│   │
│   ├──utils/
│   │   ├── config.py                    # Chargeur de config
│   │   ├── logger.py                    # Système multi-logs
│   │   ├── backtest.py                  #
│   │   ├── visualizer.py                #
│   │   ├── math_utils.py                # Pour les fonctions de précision
│   │   └── time_utils.py                # Pour convertir les formats de temps (ex: millisecondes vers objets datetime lisibles ou calcul de timestamps pour les requêtes API)
│   │
│   └── ia/                              # section Intelligence Artificielle / Machine Learning
│       ├── sentiment_analyzer.py        # 
│       ├── models.py                    # Définition mathématique du réseau de neurones
│       └── price_predictor.py           # Le "cerveau" en action. Elle charge les poids entraînés, transforme les données live (via un scaler) et fournit la prédiction brute.
│ 
├─  scripts/
│   ├── train_lstm.py                    # transforme fichiers CSV bruts en fenêtres de données utilisables pour l'entraînement IA.
│   └── download_data.py                 # --------->  FOR YOU  <---------
│ 
├── config/                              # Configurations du projet
│   ├── config_testnet.json              # Paramètres pour le testnet
│   ├── config_mainnet.json              # Paramètres pour le mainnet
│   ├── pairs.json                       # Liste des paires / actifs à trader
│   └── .env                             # (non versionné)
│
├── data/                                # Données locales (non versionnées)
│   ├── raw/                             # Données brutes (candles téléchargées)
│   ├── processed/                       # Données prêtes à l’analyse
│   └── results/                         # Résultats de backtests ou logs PnL
│
├── notebooks/                           # Analyses et tests exploratoires
│   ├── strategie_SMA.ipynb
│   ├── indicateurs_comparaison.ipynb
│   └── matrice_tendance.ipynb
│
├── tests/                              # Tests unitaires
│   ├── test_exchange.py
│   ├── test_sma.py
│   └── test_bot.py
│
├── .env.example                         # Exemple d’environnement (à copier en .env)
├── .gitignore                           # Fichiers à ignorer par Git
├── requirements.txt                     # Librairies Python nécessaires
├── README.md                            # Documentation de base
└── main.py                              # Point d’entrée du bot

Example data/
data/
├── raw/                                 # Données brutes reçues depuis l’API
│   ├── ETH_USD_1m.csv
│   └── BTC_USD_1m.csv
│
├── processed/                           # Données nettoyées + indicateurs calculés
│   ├── ETH_USD_1m_processed.parquet
│   └── BTC_USD_1m_processed.parquet
│
└── results/                             # Résultats de backtests, logs de performance
    ├── test_sma.json
    └── pnl_log.csv
