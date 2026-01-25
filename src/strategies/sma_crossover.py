from src.strategies.base_strategy import BaseStrategy
from src.core.signal import TradeSignal, SignalAction, OrderType
import pandas as pd

class SMACrossStrategyReverse(BaseStrategy):
    def __init__(self, fast_period: int = 29, slow_period: int = 132, cooldown: int = 15, stop_loss_pct: float = 0.0, verbose: bool = False):
        super().__init__(name=f"SMA_Cross_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.cooldown = cooldown
        self.stop_loss_pct = stop_loss_pct  # En % (ex: 0.02 = 2%)
        self.verbose = verbose  # Flag pour les prints de debug
        
        self.min_data_required = slow_period + 5
        self.last_signal_index = -cooldown

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pré-calcul des SMA sur le DataFrame entier (vectorisé, très rapide).
        Appelé UNE SEULE FOIS avant le backtest.
        """
        df = df.copy()
        df['fast_sma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_sma'] = df['close'].rolling(window=self.slow_period).mean()
        return df

    @classmethod
    def validate_params(cls, params: dict) -> bool:
        """
        Valide que la SMA rapide est plus petite que la SMA lente.
        """
        if "fast_period" in params and "slow_period" in params:
            if params["fast_period"] >= params["slow_period"]:
                return False
        return True

    def generate_signal(self, df: pd.DataFrame, current_index: int, metadata: dict) -> TradeSignal:
        symbol = metadata.get('symbol', 'UNKNOWN')
        
        if current_index < self.min_data_required:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)

        # Les SMA sont pré-calculées dans df['fast_sma'] et df['slow_sma']
        # On lit directement les valeurs à l'index courant (très rapide)
        f_curr = df['fast_sma'].iloc[current_index]
        f_prev = df['fast_sma'].iloc[current_index - 1]
        s_curr = df['slow_sma'].iloc[current_index]
        s_prev = df['slow_sma'].iloc[current_index - 1]
        current_price = df['close'].iloc[current_index]

        action = SignalAction.HOLD

        # Détection du croisement
        #is_bullish_cross = (f_prev <= s_prev and f_curr > s_curr)
        #is_bearish_cross = (f_prev >= s_prev and f_curr < s_curr)

        is_bearish_cross = (f_prev <= s_prev and f_curr > s_curr)
        is_bullish_cross = (f_prev >= s_prev and f_curr < s_curr)

        if is_bullish_cross:
            if (current_index - self.last_signal_index) >= self.cooldown:
                action = SignalAction.LONG
                if self.verbose:
                    print(f"DEBUG: Signal LONG détecté à {df['timestamp'].iloc[current_index]} | Fast: {f_curr:.2f} < Slow: {s_curr:.2f}")

        elif is_bearish_cross:
            if (current_index - self.last_signal_index) >= self.cooldown:
                action = SignalAction.SHORT
                if self.verbose:
                    print(f"DEBUG: Signal SHORT détecté à {df['timestamp'].iloc[current_index]} | Fast: {f_curr:.2f} > Slow: {s_curr:.2f}")

        if action != SignalAction.HOLD:
            self.last_signal_index = current_index

        # Calculer le stop_loss si défini
        stop_loss_price = None
        if action != SignalAction.HOLD and self.stop_loss_pct > 0:
            if action == SignalAction.LONG:
                stop_loss_price = float(current_price * (1 - float(self.stop_loss_pct)))
            else:  # SHORT
                stop_loss_price = float(current_price * (1 + float(self.stop_loss_pct)))

        return TradeSignal(
            action=action,
            symbol=symbol,
            order_type=OrderType.MARKET,
            price=current_price,
            stop_loss=stop_loss_price
        )