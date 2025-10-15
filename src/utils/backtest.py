import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, strategy, initial_balance=1000, fee=0.001, window=20):
        """
        strategy : objet stratégie avec méthode generate_signal(current_candle, history_slice)
        initial_balance : capital initial
        fee : frais par trade (ex: 0.001 = 0.1%)
        window : nombre de bougies nécessaires pour les indicateurs (SMA, RSI, etc.)
        """
        self.strategy = strategy
        self.balance = initial_balance
        self.fee = fee
        self.trades = []
        self.window = window  # pour history_slice

    def run(self, data: pd.DataFrame):
        """
        data: DataFrame avec colonnes ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        position = 0  # 0 = neutre, 1 = long
        entry_price = 0

        for i in range(len(data)):
            # history_slice = les dernières 'window' bougies avant la bougie courante
            start_idx = max(0, i - self.window)
            history_slice = data.iloc[start_idx:i]  # DataFrame vide si i < window
            current_candle = data.iloc[i]

            # générer le signal pour cette bougie
            signal = self.strategy.generate_signal(current_candle)

            price = current_candle['close']

            # logique d'achat / vente simple
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
