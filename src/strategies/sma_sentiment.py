"""
Exemple d'intégration du SentimentAnalyzer dans une stratégie
Ce pattern peut être réutilisé dans n'importe quelle stratégie
"""
from src.strategies.base_strategy import BaseStrategy
from src.core.signal import TradeSignal, SignalAction, OrderType
from src.ia.sentiment_analyzer import SentimentAnalyzer
import pandas as pd


class SMASentimentStrategy(BaseStrategy):
    """
    Stratégie SMA Crossover avec filtre de sentiment
    
    Pattern réutilisable :
    1. Initialiser SentimentAnalyzer dans __init__
    2. Utiliser analyzer.analyze() dans generate_signal()
    3. Le mode (backtest/live) est automatiquement géré
    """
    
    def __init__(
        self, 
        fast_period: int = 29, 
        slow_period: int = 132,
        cooldown: int = 15,
        stop_loss_pct: float = 0.0,
        sentiment_csv: str = None,  # Chemin vers CSV ou None pour mode live
        sentiment_threshold: float = 0.7,
        verbose: bool = False
    ):
        super().__init__(name=f"SMA_Sentiment_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.cooldown = cooldown
        self.stop_loss_pct = stop_loss_pct
        self.sentiment_threshold = sentiment_threshold
        self.verbose = verbose
        
        self.min_data_required = slow_period + 5
        self.last_signal_index = -cooldown
        
        # 🎯 INTÉGRATION DU SENTIMENT ANALYZER
        # Automatiquement en mode BACKTEST si sentiment_csv fourni, sinon LIVE
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_csv=sentiment_csv)
        
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pré-calcul des SMA"""
        df = df.copy()
        df['fast_sma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_sma'] = df['close'].rolling(window=self.slow_period).mean()
        return df
    
    @classmethod
    def validate_params(cls, params: dict) -> bool:
        """Validation des paramètres"""
        if "fast_period" in params and "slow_period" in params:
            if params["fast_period"] >= params["slow_period"]:
                return False
        return True
    
    def generate_signal(self, df: pd.DataFrame, current_index: int, metadata: dict) -> TradeSignal:
        """
        Génère un signal en combinant analyse technique et sentiment
        """
        symbol = metadata.get('symbol', 'UNKNOWN')
        
        if current_index < self.min_data_required:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)
        
        # 1. ANALYSE TECHNIQUE (SMA Crossover)
        f_curr = df['fast_sma'].iloc[current_index]
        f_prev = df['fast_sma'].iloc[current_index - 1]
        s_curr = df['slow_sma'].iloc[current_index]
        s_prev = df['slow_sma'].iloc[current_index - 1]
        current_price = df['close'].iloc[current_index]
        
        is_bullish_cross = (f_prev <= s_prev and f_curr > s_curr)
        is_bearish_cross = (f_prev >= s_prev and f_curr < s_curr)
        
        # Pas de croisement ou cooldown
        if not (is_bullish_cross or is_bearish_cross):
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)
        
        if (current_index - self.last_signal_index) < self.cooldown:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)
        
        # 2. ANALYSE DE SENTIMENT
        # 🎯 Le SentimentAnalyzer gère automatiquement backtest vs live
        timestamp = df['timestamp'].iloc[current_index]
        
        if self.sentiment_analyzer.mode == "backtest":
            # Mode backtest : récupère depuis CSV
            sentiment_result = self.sentiment_analyzer.analyze(None, timestamp=timestamp)
        else:
            # Mode live : sans news, on accepte le signal (ou implémenter API de news)
            sentiment_result = {'label': 'neutral', 'score': 1.0}
        
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']
        
        # 3. VALIDATION DU SIGNAL
        action = SignalAction.HOLD
        
        if is_bullish_cross:
            # LONG validé seulement si sentiment positif
            if sentiment_label == 'positive' and sentiment_score >= self.sentiment_threshold:
                action = SignalAction.LONG
                if self.verbose:
                    print(f"✓ LONG @ {timestamp} | SMA: {f_curr:.2f}>{s_curr:.2f} | "
                          f"Sentiment: {sentiment_label} ({sentiment_score:.2f})")
            elif self.verbose:
                print(f"✗ LONG rejeté @ {timestamp} | Sentiment: {sentiment_label} ({sentiment_score:.2f})")
        
        elif is_bearish_cross:
            # SHORT validé seulement si sentiment négatif
            if sentiment_label == 'negative' and sentiment_score >= self.sentiment_threshold:
                action = SignalAction.SHORT
                if self.verbose:
                    print(f"✓ SHORT @ {timestamp} | SMA: {f_curr:.2f}<{s_curr:.2f} | "
                          f"Sentiment: {sentiment_label} ({sentiment_score:.2f})")
            elif self.verbose:
                print(f"✗ SHORT rejeté @ {timestamp} | Sentiment: {sentiment_label} ({sentiment_score:.2f})")
        
        if action != SignalAction.HOLD:
            self.last_signal_index = current_index
        
        # 4. STOP LOSS
        stop_loss_price = None
        if action != SignalAction.HOLD and self.stop_loss_pct > 0:
            if action == SignalAction.LONG:
                stop_loss_price = float(current_price * (1 - self.stop_loss_pct))
            else:
                stop_loss_price = float(current_price * (1 + self.stop_loss_pct))
        
        return TradeSignal(
            action=action,
            symbol=symbol,
            price=float(current_price),
            stop_loss=stop_loss_price,
            order_type=OrderType.MARKET
        )
