from abc import ABC, abstractmethod
import pandas as pd
from src.core.signal import TradeSignal, SignalAction

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, metadata: dict) -> TradeSignal:
        """
        Méthode principale que chaque stratégie doit implémenter.
        :param df: DataFrame contenant les colonnes OHLCV standard.
        :param metadata: Dictionnaire (ex: {'symbol': 'BTC', 'interval': '1m'}).
        :return: Une instance de TradeSignal.
        """
        pass