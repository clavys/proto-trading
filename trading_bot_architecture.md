hyperliquid-trading-bot/
│
├── src/
│ ├── core/ # Logique principale du bot
│ │ ├── bot.py # Gère les stratégies et exécute le loop principal
│ │ ├── exchange.py # Interface avec l’API Hyperliquid (Info / Exchange)
│ │ ├── data.py # Récupération et prétraitement des données (candles, OHLCV)
│ │ └── strategy_manager.py # Gère plusieurs stratégies et les signaux combinés
│ │
│ ├── strategies/ # Répertoire des stratégies de trading
│ │ ├── sma_crossover.py # Stratégie moyenne mobile simple
│ │ ├── rsi_reversal.py # Stratégie RSI
│ │ ├── macd_trend.py # Stratégie MACD
│ │ └── trend_matrix.py # Matrice de tendance multi-crypto
│ │
│ ├── utils/ # Fonctions utilitaires
│ │ ├── config.py # Chargement du fichier de config + .env
│ │ ├── logger.py # Gestion centralisée des logs
│ │ └── backtest.py # Module de backtest et d’analyse de performance
│ │
│ └── ai/ # (future) section Intelligence Artificielle / Machine Learning
│ ├── ml_models.py # Modèles de machine learning (Random Forest, etc.)
│ ├── deep_learning.py # Réseaux neuronaux (LSTM / CNN / etc.)
│ └── reinforcement.py # Agent de trading par renforcement
│
├── config/ # Configurations du projet
│ ├── config_testnet.json # Paramètres pour le testnet
│ ├── config_mainnet.json # Paramètres pour le mainnet
│ └── pairs.json # Liste des paires / actifs à trader
│
├── data/ # Données locales (non versionnées)
│ ├── raw/ # Données brutes (candles téléchargées)
│ ├── processed/ # Données prêtes à l’analyse
│ └── results/ # Résultats de backtests ou logs PnL
│
├── notebooks/ # Analyses et tests exploratoires
│ ├── strategie_SMA.ipynb
│ ├── indicateurs_comparaison.ipynb
│ └── matrice_tendance.ipynb
│
├── tests/ # Tests unitaires
│ ├── test_exchange.py
│ ├── test_sma.py
│ └── test_bot.py
│
├── .env.example # Exemple d’environnement (à copier en .env)
├── .gitignore # Fichiers à ignorer par Git
├── requirements.txt # Librairies Python nécessaires
├── README.md # Documentation de base
└── main.py # Point d’entrée du bot



