# Exemple pour src/strategies
from src.core.signal import TradeSignal

class BaseStrategy:
    def __init__(self, name: str):
        self.name = name

    def generate_signal(self, dataframe) -> TradeSignal:
        raise NotImplementedError("Chaque stratégie doit implémenter cette méthode")