from src.strategies.base_strategy import BaseStrategy
from src.core.signal import TradeSignal, SignalAction, OrderType
import pandas as pd
import numpy as np

class SMACrossEnhanced(BaseStrategy):
    def __init__(self, 
                 fast_period: int = 9, 
                 slow_period: int = 21, 
                 trend_period: int = 200,
                 atr_period: int = 14,
                 tp_mult: float = 2.0,  # Take Profit: 2x ATR
                 sl_mult: float = 1.5,  # Stop Loss: 1.5x ATR
                 use_trend_filter: bool = True,
                 cooldown: int = 5):
        
        super().__init__(name=f"SMA_Enhanced_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.use_trend_filter = use_trend_filter
        self.cooldown = cooldown
        
        self.min_data_required = max(slow_period, trend_period, atr_period) + 5
        self.last_signal_index = -cooldown

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Moyennes mobiles
        df['fast_sma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_sma'] = df['close'].rolling(window=self.slow_period).mean()
        df['trend_sma'] = df['close'].rolling(window=self.trend_period).mean()
        
        # Volatilité (ATR simplifié)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        return df

    def generate_signal(self, df: pd.DataFrame, current_index: int, metadata: dict) -> TradeSignal:
        symbol = metadata.get('symbol', 'UNKNOWN')
        
        if current_index < self.min_data_required:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)

        # Valeurs actuelles
        f_curr, f_prev = df['fast_sma'].iloc[current_index], df['fast_sma'].iloc[current_index - 1]
        s_curr, s_prev = df['slow_sma'].iloc[current_index], df['slow_sma'].iloc[current_index - 1]
        price = df['close'].iloc[current_index]
        trend = df['trend_sma'].iloc[current_index]
        atr = df['atr'].iloc[current_index]

        action = SignalAction.HOLD
        tp, sl = None, None

        # Détection du croisement
        is_bullish_cross = (f_prev <= s_prev and f_curr > s_curr)
        is_bearish_cross = (f_prev >= s_prev and f_curr < s_curr)

        # Application du cooldown
        can_trade = (current_index - self.last_signal_index) >= self.cooldown

        if can_trade:
            # Signal LONG : Croisement Haussier + Au-dessus de la SMA 200
            if is_bullish_cross:
                if not self.use_trend_filter or price > trend:
                    action = SignalAction.LONG
                    sl = price - (atr * self.sl_mult)
                    tp = price + (atr * self.tp_mult)

            # Signal SHORT : Croisement Baissier + En-dessous de la SMA 200
            elif is_bearish_cross:
                if not self.use_trend_filter or price < trend:
                    action = SignalAction.SHORT
                    sl = price + (atr * self.sl_mult)
                    tp = price - (atr * self.tp_mult)

        if action != SignalAction.HOLD:
            self.last_signal_index = current_index

        return TradeSignal(
            action=action,
            symbol=symbol,
            price=price,
            stop_loss=sl,
            take_profit=tp
        )