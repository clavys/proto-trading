# src/core/strategy_manager.py
from typing import List
from src.core.signal import TradeSignal, SignalAction

class StrategyManager:
    def __init__(self, strategies: List):
        """
        :param strategies: Liste d'instances de stratégies (ex: [SMAStrategy(), RSIStrategy()])
        """
        self.strategies = strategies
        self.logger = None # Tu pourras lier ton logger ici plus tard

    def get_combined_signal(self, dataframe, symbol: str) -> TradeSignal:
        """
        Récupère les signaux de toutes les stratégies et décide de l'action finale.
        """
        signals = []
        
        # 1. Collecter les signaux de chaque stratégie
        for strategy in self.strategies:
            sig = strategy.generate_signal(dataframe)
            signals.append(sig)
            # Optionnel : loguer ce que chaque stratégie dit
            # print(f"Stratégie {strategy.name} propose : {sig.action}")

        # 2. Logique de décision (Consensus)
        # On vérifie si toutes les stratégies (qui ne disent pas HOLD) sont d'accord
        actions = [s.action for s in signals if s.action != SignalAction.HOLD]

        if not actions:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)

        # Si toutes les stratégies actives disent LONG
        if all(a == SignalAction.LONG for a in actions) and len(actions) == len(self.strategies):
            return signals[0] # On retourne le premier signal LONG (avec ses TP/SL)

        # Si toutes les stratégies actives disent SHORT
        if all(a == SignalAction.SHORT for a in actions) and len(actions) == len(self.strategies):
            return signals[0]

        # Par défaut, si elles ne sont pas d'accord, on ne fait rien
        return TradeSignal(action=SignalAction.HOLD, symbol=symbol)
