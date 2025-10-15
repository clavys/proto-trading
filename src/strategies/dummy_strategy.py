import pandas as pd

class DummyStrategy:
    """
    Stratégie de test (bouchon) pour vérifier le bon fonctionnement du backtest.
    Elle ne repose sur aucun indicateur technique.
    Retourne maintenant **un seul signal à la fois** pour le backtest.
    """

    def __init__(self, switch_every=5):
        self.switch_every = switch_every
        self.counter = 0  # compteur pour suivre la bougie courante

    def generate_signal(self, candle) -> int:
        """
        Retourne un signal pour **une seule bougie** :
        1 = buy, -1 = sell, 0 = neutre
        """
        self.counter += 1
        if self.counter <= self.switch_every:
            return 0
        elif (self.counter // self.switch_every) % 2 == 0:
            return 1
        else:
            return -1