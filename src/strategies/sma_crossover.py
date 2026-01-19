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

    def generate_signal(self, df: pd.DataFrame, metadata: dict) -> TradeSignal:
        symbol = metadata.get('symbol', 'UNKNOWN')
        current_idx = len(df)
        
        if current_idx < self.min_data_required:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)

        # Calcul des SMA
        fast_sma = df['close'].rolling(window=self.fast_period).mean()
        slow_sma = df['close'].rolling(window=self.slow_period).mean()

        # On prend les 2 dernières valeurs CLOSES (-1 et -2)
        f_curr, f_prev = fast_sma.iloc[-1], fast_sma.iloc[-2]
        s_curr, s_prev = slow_sma.iloc[-1], slow_sma.iloc[-2]
        current_price = df['close'].iloc[-1]

        action = SignalAction.HOLD

        # Détection du croisement
        #is_bullish_cross = (f_prev <= s_prev and f_curr > s_curr)
        #is_bearish_cross = (f_prev >= s_prev and f_curr < s_curr)

        is_bullish_cross = (f_prev >= s_prev and f_curr < s_curr)
        is_bearish_cross = (f_prev <= s_prev and f_curr > s_curr)

        # Filtre Delta
        delta = abs(f_curr - s_curr)
        required_delta = current_price * self.min_delta_pct

        if is_bullish_cross and delta >= required_delta:
            if (current_idx - self.last_signal_index) >= self.cooldown:
                action = SignalAction.LONG
                if self.verbose:
                    print(f"DEBUG: Signal LONG détecté à {df['timestamp'].iloc[-1]} | Fast: {f_curr:.2f} > Slow: {s_curr:.2f}")

        elif is_bearish_cross and delta >= required_delta:
            if (current_idx - self.last_signal_index) >= self.cooldown:
                action = SignalAction.SHORT
                if self.verbose:
                    print(f"DEBUG: Signal SHORT détecté à {df['timestamp'].iloc[-1]} | Fast: {f_curr:.2f} < Slow: {s_curr:.2f}")

        if action != SignalAction.HOLD:
            self.last_signal_index = current_idx

        return TradeSignal(
            action=action,
            symbol=symbol,
            order_type=OrderType.MARKET,
            price=current_price
        )