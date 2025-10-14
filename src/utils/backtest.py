
import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    def __init__(self, strategy, initial_balance=1000, fee=0.001):
        self.strategy = strategy
        self.balance = initial_balance
        self.fee = fee
        self.trades = []

    def run(self, data: pd.DataFrame):
        """
        data: DataFrame avec colonnes ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        signals = self.strategy.generate_signals(data)
        position = 0  # 0 = neutre, 1 = long, -1 = short
        entry_price = 0

        for i in range(len(data)):
            price = data['close'].iloc[i]
            signal = signals.iloc[i]

            if signal == 1 and position == 0:  # Buy
                position = 1
                entry_price = price

            elif signal == -1 and position == 1:  # Sell
                pnl = (price - entry_price) / entry_price * self.balance
                self.balance += pnl - (self.balance * self.fee)
                self.trades.append(pnl)
                position = 0

        return self._summary()

    def _summary(self):
        total_pnl = sum(self.trades)
        return {
            "final_balance": self.balance,
            "total_pnl": total_pnl,
            "num_trades": len(self.trades),
            "avg_pnl": np.mean(self.trades) if self.trades else 0,
        }
