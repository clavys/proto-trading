import pandas as pd
import numpy as np
from src.core.signal import SignalAction, TradeSignal

class Backtester:
    def __init__(self, strategy, initial_balance=1000, fee=0.001):
        """
        :param strategy: Instance d'une classe héritant de BaseStrategy
        :param initial_balance: Capital de départ
        :param fee: Frais par transaction (ex: 0.001 pour 0.1%)
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee = fee
        
        self.equity_curve = []
        self.trades = []
        self.position = None  # Stocke le TradeSignal actuel ou None

    def run(self, data: pd.DataFrame, metadata: dict = None):
        """
        Exécute la simulation sur un DataFrame standardisé.
        """
        if metadata is None:
            metadata = {'symbol': 'UNKNOWN'}

        # On commence après que suffisamment de données soient disponibles pour les indicateurs
        # Si la stratégie a un attribut min_data_required, on l'utilise
        start_idx = getattr(self.strategy, 'min_data_required', 1)

        for i in range(start_idx, len(data)):
            # On passe une vue des données jusqu'à l'instant T (fenêtre glissante)
            history = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            timestamp = data.iloc[i].get('timestamp', i)

            # 1. Générer le signal
            signal = self.strategy.generate_signal(history, metadata)

            # 2. Logique d'exécution
            self._handle_signal(signal, current_price, timestamp)

            # 3. Sauvegarde de la valeur du portefeuille
            self.equity_curve.append(self.balance)

        return self._summary()

    def _handle_signal(self, signal: TradeSignal, price: float, timestamp):
        # Fermeture de position si signal opposé ou CLOSE
        if self.position:
            should_close = (
                (self.position.action == SignalAction.LONG and signal.action in [SignalAction.SHORT, SignalAction.CLOSE]) or
                (self.position.action == SignalAction.SHORT and signal.action in [SignalAction.LONG, SignalAction.CLOSE])
            )
            
            if should_close:
                self._close_position(price, timestamp)

        # Ouverture de position
        if self.position is None:
            if signal.action in [SignalAction.LONG, SignalAction.SHORT]:
                self._open_position(signal, price, timestamp)

    def _open_position(self, signal: TradeSignal, price: float, timestamp):
        self.position = signal
        self.entry_price = price
        self.entry_time = timestamp
        # Appliquer les frais à l'entrée
        self.balance -= self.balance * self.fee

    def _close_position(self, price: float, timestamp):
        # Calcul du PnL en fonction du sens (Long ou Short)
        if self.position.action == SignalAction.LONG:
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - price) / self.entry_price
        
        # Levier (par défaut 1 si non spécifié)
        leverage = getattr(self.position, 'leverage', 1)
        pnl_cash = (self.balance * pnl_pct) * leverage
        
        self.balance += pnl_cash
        self.balance -= self.balance * self.fee  # Frais de sortie
        
        self.trades.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl_pct': pnl_pct,
            'pnl_cash': pnl_cash,
            'type': self.position.action
        })
        self.position = None

    def _summary(self):
        df_trades = pd.DataFrame(self.trades)
        total_pnl = self.balance - self.initial_balance
        roi = (total_pnl / self.initial_balance) * 100
        
        win_rate = 0
        if not df_trades.empty:
            win_rate = (df_trades['pnl_cash'] > 0).sum() / len(df_trades) * 100

        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_pnl_cash": total_pnl,
            "roi_pct": roi,
            "win_rate_pct": win_rate,
            "num_trades": len(self.trades),
            "trades": self.trades
        }