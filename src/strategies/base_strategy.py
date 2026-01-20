from abc import ABC, abstractmethod
import pandas as pd
from src.core.signal import TradeSignal, SignalAction

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, current_index: int, metadata: dict) -> TradeSignal:
        """
        Méthode principale que chaque stratégie doit implémenter.
        :param df: DataFrame complet avec les colonnes OHLCV standard et indicateurs pré-calculés.
        :param current_index: Index de la bougie actuelle (pour la fenêtre glissante).
        :param metadata: Dictionnaire (ex: {'symbol': 'BTC', 'interval': '1m'}).
        :return: Une instance de TradeSignal.
        """
        pass

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pré-calcul des indicateurs sur le DataFrame ENTIER avant le backtest.
        À surcharger dans les stratégies qui ont des indicateurs custom.
        :param df: DataFrame contenant les colonnes OHLCV standard.
        :return: DataFrame modifié avec les colonnes d'indicateurs ajoutées.
        """
        # Par défaut, ne rien ajouter. Les stratégies surchargeront.
        return df

    @classmethod
    def validate_params(cls, params: dict) -> bool:
        """
        Valide les paramètres avant le backtest.
        À surcharger dans les stratégies qui ont des contraintes spécifiques.
        :param params: Dictionnaire des paramètres à valider
        :return: True si valide, False sinon
        """
        return True