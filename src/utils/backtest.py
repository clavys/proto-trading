import pandas as pd
import numpy as np
from src.core.signal import SignalAction, TradeSignal

class Backtester:
    def __init__(self, strategy, initial_balance=1000, fee=0.00035):
        """
        :param strategy: Instance d'une classe héritant de BaseStrategy
        :param initial_balance: Capital de départ
        :param fee: Frais par transaction (ex: 0.001 pour 0.1%)
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.available_balance = initial_balance  # Cash disponible
        self.invested_amount = 0  # Montant investi dans la position actuelle
        self.fee = fee
        
        self.equity_curve = []
        self.drawdown_curve = []
        self.trades = []
        self.position = None  # Stocke le TradeSignal actuel ou None
        self.peak_balance = initial_balance  # Peak pour calcul du drawdown
        self.last_price = initial_balance  # Dernier prix pour position ouverte

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
            self.last_price = current_price  # Track le dernier prix

            # 1. Générer le signal
            signal = self.strategy.generate_signal(history, metadata)

            # 2. Logique d'exécution
            self._handle_signal(signal, current_price, timestamp)

            # 3. Calcul du portefeuille total et sauvegarde
            total_balance = self.available_balance + self._get_position_value(current_price)
            self.equity_curve.append(total_balance)
            
            # Tracking du drawdown
            if total_balance > self.peak_balance:
                self.peak_balance = total_balance
            drawdown = ((self.peak_balance - total_balance) / self.peak_balance) * 100
            self.drawdown_curve.append(drawdown)

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
        
        # Montant investi avec frais
        # On utilise tout le cash disponible
        self.invested_amount = self.available_balance * (1 - self.fee)
        self.available_balance -= self.invested_amount

    def _close_position(self, price: float, timestamp):
        # Calcul du PnL en fonction du sens (Long ou Short)
        if self.position.action == SignalAction.LONG:
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - price) / self.entry_price
        
        # Levier (par défaut 1 si non spécifié)
        leverage = getattr(self.position, 'leverage', 1)
        pnl_cash = (self.invested_amount * pnl_pct) * leverage
        
        # Récupérer le montant investi + PnL, en déduisant les frais de sortie
        exit_amount = self.invested_amount + pnl_cash
        exit_amount -= exit_amount * self.fee  # Frais de sortie
        
        self.available_balance += exit_amount
        
        self.trades.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl_pct': pnl_pct,
            'pnl_cash': pnl_cash,
            'type': self.position.action,
            'leverage': leverage
        })
        self.position = None
        self.invested_amount = 0

    def _summary(self):
        df_trades = pd.DataFrame(self.trades)
        
        # Solde final = cash disponible + position ouverte (si existe)
        final_balance = self.available_balance + self._get_position_value(self.last_price)
        
        total_pnl = final_balance - self.initial_balance
        roi = (total_pnl / self.initial_balance) * 100
        
        # Métriques de trades
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        
        if not df_trades.empty:
            winning_trades = df_trades[df_trades['pnl_cash'] > 0]
            losing_trades = df_trades[df_trades['pnl_cash'] <= 0]
            
            win_rate = (len(winning_trades) / len(df_trades)) * 100
            
            if len(winning_trades) > 0:
                avg_win = winning_trades['pnl_cash'].mean()
            if len(losing_trades) > 0:
                avg_loss = losing_trades['pnl_cash'].mean()
            
            total_wins = winning_trades['pnl_cash'].sum()
            total_losses = abs(losing_trades['pnl_cash'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Drawdown
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Sharpe Ratio (annualisé, assumant 252 jours de trading)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return {
            "initial_balance": self.initial_balance,
            "final_balance": final_balance,
            "total_pnl_cash": total_pnl,
            "roi_pct": roi,
            "num_trades": len(self.trades),
            "win_rate_pct": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "drawdown_curve": self.drawdown_curve
        }
    
    def _get_position_value(self, current_price: float) -> float:
        """Calcule la valeur actuelle de la position ouverte"""
        if not self.position or self.invested_amount == 0:
            return 0
        
        if self.position.action == SignalAction.LONG:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        leverage = getattr(self.position, 'leverage', 1)
        unrealized_pnl = (self.invested_amount * pnl_pct) * leverage
        
        return self.invested_amount + unrealized_pnl
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02) -> float:
        """Calcule le Sharpe Ratio annualisé"""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        # Annualisé (252 jours de trading)
        excess_returns = np.mean(returns) - (risk_free_rate / 252)
        sharpe = (excess_returns / np.std(returns)) * np.sqrt(252)
        
        return round(sharpe, 2)