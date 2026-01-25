import pandas as pd
import numpy as np
import os
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
        self.position_min_price = None  # Minimum de prix atteint dans la position (pour LONG)
        self.position_max_price = None  # Maximum de prix atteint dans la position (pour SHORT)

    @staticmethod
    def extract_symbol_from_path(filepath: str) -> str:
        """
        Extrait le symbole du chemin du fichier (ex: 'data/raw/BTCUSDT-1m-2026-01-14.csv' -> 'BTCUSDT')
        """
        try:
            filename = os.path.basename(filepath)
            symbol = filename.split('-')[0]  # Prend le premier segment avant le '-'
            return symbol
        except:
            return 'UNKNOWN'

    def run(self, data: pd.DataFrame, metadata: dict = None, detailed: bool = True, filepath: str = None):
        """
        Exécute la simulation sur un DataFrame standardisé.
        :param data: DataFrame avec les données de marché
        :param metadata: Metadata optionnelles (symbole, etc.)
        :param detailed: Si False, mode léger pour l'optimisation (pas de stockage des courbes)
        :param filepath: Chemin du fichier pour extraire le symbol automatiquement
        """
        if metadata is None:
            metadata = {}
        
        # Auto-extraction du symbol si filepath fourni et symbol pas dans metadata
        if filepath and 'symbol' not in metadata:
            metadata['symbol'] = self.extract_symbol_from_path(filepath)
        
        # Fallback si toujours pas de symbol
        if 'symbol' not in metadata:
            metadata['symbol'] = 'UNKNOWN'

        # ✨ PRÉ-CALCUL des indicateurs (une seule fois, vectorisé)
        data = self.strategy.prepare_indicators(data)

        # On commence après que suffisamment de données soient disponibles pour les indicateurs
        # Si la stratégie a un attribut min_data_required, on l'utilise
        start_idx = getattr(self.strategy, 'min_data_required', 1)

        for i in range(start_idx, len(data)):
            current_price = data.iloc[i]['close']
            timestamp = data.iloc[i].get('timestamp', i)
            self.last_price = current_price  # Track le dernier prix

            # 1. Générer le signal (passer le DataFrame ENTIER + index courant)
            signal = self.strategy.generate_signal(data, i, metadata)

            # 2. Logique d'exécution
            self._handle_signal(signal, current_price, timestamp)

            # 3. Mode détaillé : stockage des courbes
            if detailed:
                total_balance = self.available_balance + self._get_position_value(current_price)
                self.equity_curve.append(total_balance)
                
                # Tracking du drawdown
                if total_balance > self.peak_balance:
                    self.peak_balance = total_balance
                drawdown = ((self.peak_balance - total_balance) / self.peak_balance) * 100
                self.drawdown_curve.append(drawdown)
            else:
                # Mode léger : juste update du peak pour max_drawdown final
                total_balance = self.available_balance + self._get_position_value(current_price)
                if total_balance > self.peak_balance:
                    self.peak_balance = total_balance

        return self._summary(detailed=detailed)

    def _handle_signal(self, signal: TradeSignal, price: float, timestamp):
        # 1. VÉRIFICATION DU STOP-LOSS (PRIORITAIRE)
        # Le SL se déclenche INDÉPENDAMMENT du signal et immédiatement
        if self.position and self._check_stop_loss(price):
            self._close_position(price, timestamp, reason='stop_loss')
            self.position_min_price = None
            self.position_max_price = None
            # La position est fermée, on ne traite pas d'autres signaux ce tick
            return
        
        # 2. Tracker le min/max du prix pendant la position (pour validation du SL)
        if self.position:
            if self.position.action == SignalAction.LONG:
                if self.position_min_price is None:
                    self.position_min_price = price
                else:
                    self.position_min_price = min(self.position_min_price, price)
            else:  # SHORT
                if self.position_max_price is None:
                    self.position_max_price = price
                else:
                    self.position_max_price = max(self.position_max_price, price)
        
        # 3. Fermeture de position si signal opposé ou CLOSE
        if self.position:
            should_close = (
                (self.position.action == SignalAction.LONG and signal.action in [SignalAction.SHORT, SignalAction.CLOSE]) or
                (self.position.action == SignalAction.SHORT and signal.action in [SignalAction.LONG, SignalAction.CLOSE])
            )
            
            if should_close:
                self._close_position(price, timestamp, reason='signal')
                self.position_min_price = None
                self.position_max_price = None

        # 4. Ouverture de position
        if self.position is None:
            if signal.action in [SignalAction.LONG, SignalAction.SHORT]:
                self._open_position(signal, price, timestamp)
                self.position_min_price = None
                self.position_max_price = None

    def _open_position(self, signal: TradeSignal, price: float, timestamp):
        self.position = signal
        self.entry_price = float(price)  # Force float type
        self.entry_time = timestamp
        
        # Stocker le stop-loss avec typage float strict
        self.position.stop_loss = float(signal.stop_loss) if signal.stop_loss is not None else None
        
        # Montant investi avec frais
        # On utilise tout le cash disponible
        self.entry_fee = self.available_balance * self.fee
        self.invested_amount = self.available_balance - self.entry_fee
        self.available_balance = 0

    def _check_stop_loss(self, current_price: float) -> bool:
        """
        Vérifie si le stop-loss est atteint pour la position actuelle.
        Retourne True si le SL est hit, False sinon.
        Cette vérification est INDÉPENDANTE du signal stratégique et se déclenche automatiquement.
        """
        if not self.position or self.position.stop_loss is None:
            return False
        
        current_price = float(current_price)  # Force float type
        stop_loss = float(self.position.stop_loss)  # Force float type
        
        if self.position.action == SignalAction.LONG:
            # Pour un LONG, le SL est atteint si le prix baisse AU OU SOUS le stop_loss
            return current_price <= stop_loss
        else:  # SHORT
            # Pour un SHORT, le SL est atteint si le prix monte AU OU AU-DESSUS du stop_loss
            return current_price >= stop_loss
    
    def _close_position(self, price: float, timestamp, reason: str = 'signal'):
        """
        Ferme la position actuelle.
        :param price: Prix de fermeture (float)
        :param timestamp: Timestamp de la fermeture
        :param reason: Raison de la fermeture ('signal', 'stop_loss', etc.)
        """
        price = float(price)  # Force float type
        
        # Calcul du PnL en fonction du sens (Long ou Short)
        if self.position.action == SignalAction.LONG:
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - price) / self.entry_price
        
        # Levier (par défaut 1 si non spécifié)
        leverage = getattr(self.position, 'leverage', 1)
        pnl_cash = (self.invested_amount * pnl_pct) * leverage
        
        # Frais de SORTIE calculés sur la valeur FINALE de la position
        # (capital investi + PnL brut), pas sur l'entrée
        final_position_value = self.invested_amount + pnl_cash
        exit_fee = abs(final_position_value) * self.fee  # Basé sur la valeur finale
        
        # Le montant final après frais de sortie
        exit_amount = final_position_value - exit_fee
        
        # Le vrai PnL cash après TOUS les frais (entry_fee + exit_fee)
        pnl_cash_after_fees = pnl_cash - self.entry_fee - exit_fee
        
        self.available_balance += exit_amount
        
        self.trades.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl_pct': pnl_pct,
            'pnl_cash': pnl_cash_after_fees,
            'type': self.position.action,
            'leverage': leverage,
            'exit_reason': reason,
            'stop_loss': self.position.stop_loss
        })
        self.position = None
        self.invested_amount = 0

    def _summary(self, detailed: bool = True):
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
        max_drawdown = 0
        if detailed and self.drawdown_curve:
            max_drawdown = max(self.drawdown_curve)
        elif not detailed:
            # En mode léger, calculer le max drawdown final
            total_balance_final = self.available_balance + self._get_position_value(self.last_price)
            max_drawdown = ((self.peak_balance - total_balance_final) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
        
        # Sharpe Ratio (annualisé, assumant 252 jours de trading) - seulement en mode détaillé
        sharpe_ratio = self._calculate_sharpe_ratio() if detailed else 0
        
        summary = {
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
        }
        
        # Ajouter les courbes seulement en mode détaillé
        if detailed:
            summary["equity_curve"] = self.equity_curve
            summary["drawdown_curve"] = self.drawdown_curve
        
        return summary
    
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