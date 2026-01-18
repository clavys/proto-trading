from src.strategies.base_strategy import BaseStrategy
from src.core.signal import TradeSignal, SignalAction, OrderType
import pandas as pd

class SMACrossStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        super().__init__(name=f"SMA_Cross_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_data_required = slow_period + 2

    def generate_signal(self, df: pd.DataFrame, metadata: dict) -> TradeSignal:
        symbol = metadata.get('symbol', 'UNKNOWN')
        
        if len(df) < self.min_data_required:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)

        # UTILISATION DE PANDAS STANDARD (Plus de bug d'installation !)
        fast_sma = df['close'].rolling(window=self.fast_period).mean()
        slow_sma = df['close'].rolling(window=self.slow_period).mean()

        f_curr, f_prev = fast_sma.iloc[-2], fast_sma.iloc[-3]
        s_curr, s_prev = slow_sma.iloc[-2], slow_sma.iloc[-3]

        action = SignalAction.HOLD
        if f_prev <= s_prev and f_curr > s_curr:
            action = SignalAction.LONG
        elif f_prev >= s_prev and f_curr < s_curr:
            action = SignalAction.SHORT

        return TradeSignal(
            action=action,
            symbol=symbol,
            order_type=OrderType.MARKET,
            price=df['close'].iloc[-1]
        )