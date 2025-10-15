import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.dummy_strategy import DummyStrategy
from src.utils.backtest import Backtester

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Nombre de bougies fictives
num_candles = 100 

# Date de départ
start_time = datetime(2025, 10, 15, 0, 0)

# Générer les timestamps
timestamps = [start_time + timedelta(minutes=i) for i in range(num_candles)]

# Générer des prix fictifs
np.random.seed(42)  # pour reproductibilité
prices = np.cumsum(np.random.randn(num_candles)) + 100  # fluctuation autour de 100

# Créer OHLCV fictif
data = pd.DataFrame({
    "timestamp": timestamps,
    "open": prices + np.random.rand(num_candles),
    "high": prices + np.random.rand(num_candles),
    "low": prices - np.random.rand(num_candles),
    "close": prices,
    "volume": np.random.randint(1, 100, size=num_candles)
})




strategy = DummyStrategy(switch_every=5)
backtester = Backtester(strategy=strategy, initial_balance=1000, fee=0.001, window=5)
    
results = backtester.run(data)
    
print("Final Balance:", results["final_balance"])
print("Total PnL:", results["total_pnl"])
print("Number of Trades:", results["num_trades"])
print("Average PnL per Trade:", results["avg_pnl"])
    
assert results["final_balance"] != 1000  # Le solde final devrait changer
assert results["num_trades"] > 0  # Il devrait y avoir au moins un trade
print(results["final_balance"], results["total_pnl"], results["num_trades"], results["avg_pnl"])

