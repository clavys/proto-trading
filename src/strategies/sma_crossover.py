from src.strategies.base_strategy import BaseStrategy
from src.core.signal import TradeSignal, SignalAction, OrderType
import pandas as pd

class SMACrossStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 9, slow_period: int = 21, min_delta_pct: float = 0.0, cooldown: int = 5, verbose: bool = False):
        super().__init__(name=f"SMA_Cross_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_delta_pct = min_delta_pct
        self.cooldown = cooldown
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
        is_bullish_cross = (f_prev <= s_prev and f_curr > s_curr)
        is_bearish_cross = (f_prev >= s_prev and f_curr < s_curr)

        # Filtre Delta
        delta = abs(f_curr - s_curr)
        required_delta = current_price * self.min_delta_pct

        if is_bullish_cross and delta >= required_delta:
            if (current_index - self.last_signal_index) >= self.cooldown:
                action = SignalAction.LONG
                if self.verbose:
                    print(f"DEBUG: Signal LONG détecté à {df['timestamp'].iloc[current_index]} | Fast: {f_curr:.2f} > Slow: {s_curr:.2f}")

        elif is_bearish_cross and delta >= required_delta:
            if (current_index - self.last_signal_index) >= self.cooldown:
                action = SignalAction.SHORT
                if self.verbose:
                    print(f"DEBUG: Signal SHORT détecté à {df['timestamp'].iloc[current_index]} | Fast: {f_curr:.2f} < Slow: {s_curr:.2f}")

        if action != SignalAction.HOLD:
            self.last_signal_index = current_index

        return TradeSignal(
            action=action,
            symbol=symbol,
            order_type=OrderType.MARKET,
            price=current_price
        )