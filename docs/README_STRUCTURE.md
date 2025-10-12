# `main.py` — Point d’entrée du bot Hyperliquid

**Rôle :**  
`main.py` est le point d’entrée principal du projet. C’est le fichier que l’on exécute pour lancer le bot, que ce soit en mode testnet, mainnet ou backtest.

---

## Fonctions principales

- **Initialisation de la configuration :**  
  Charge les paramètres depuis les fichiers JSON (`config_testnet.json` / `config_mainnet.json`) et les variables d’environnement (`.env`).

- **Initialisation du logging :**  
  Configure les logs pour suivre les événements, les erreurs et les décisions du bot.

- **Création de l’instance du bot :**  
  Importe et initialise la classe `TradingBot` depuis `bot.py`.

- **Lancement du loop principal :**  
  Exécute le bot en continu ou déclenche un backtest selon le mode choisi.

---

## Exemple d’utilisation

```bash
# Lancer le bot sur le testnet
python main.py --mode testnet

# Lancer un backtest
python main.py --mode backtest
```

<br><br><br><br><br>

# `src/bot.py` — Logique principale du bot 

**Rôle :**  
`bot.py` contient la logique métier complète du bot de trading. C’est ici que les décisions d’achat/vente sont générées et exécutées.

---

## Fonctions principales

- **Récupération des données de marché :**  
  Utilise `exchange.py` ou `data.py` pour obtenir les candles OHLCV et autres informations nécessaires aux stratégies.

- **Gestion des stratégies :**  
  Passe les données au `strategy_manager.py` qui combine les signaux de différentes stratégies (SMA, RSI, MACD, trend matrix).

- **Exécution des ordres :**  
  Envoie les ordres d’achat/vente via l’API Hyperliquid (`exchange.order()`), en respectant les positions actuelles et la gestion du risque.

- **Loop de trading continu :**  
  Le bot tourne en boucle, récupérant les nouvelles données, calculant les signaux et exécutant les ordres à intervalles réguliers.

- **Suivi des positions et PnL :**  
  Gère les positions ouvertes, calcule le PnL et peut enregistrer l’historique pour analyse ou backtest.





