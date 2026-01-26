"""
Data Providers Module
Regroupe les différentes sources d'accès aux données historiques
(API REST, Fichiers locaux .csv, DB lecture/écriture).
"""

import pandas as pd
import os
from typing import Optional
from .handler import DataHandler


class BaseDataProvider:
    """
    Classe de base pour tous les fournisseurs de données.
    """
    
    def __init__(self):
        pass
    
    def fetch_data(self, params):
        """
        Récupère les données selon les paramètres fournis.
        """
        pass


class LocalDataProvider(BaseDataProvider):
    """
    Utile pour le BACKTEST. Charge les fichiers stockés sur ton PC.
    """
    def load_from_file(self, file_path: str, is_binance: bool = False, col_map: dict = None) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

        extension = file_path.split('.')[-1].lower()
        
        # Lecture du fichier
        if extension == 'csv':
            # Binance n'a pas de headers, les autres CSV en ont généralement
            header = None if is_binance else 0
            df = pd.read_csv(file_path, header=header)
        elif extension == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Format non supporté : CSV ou Parquet uniquement.")

        # Normalisation
        if is_binance:
            return DataHandler.normalize_binance_klines(df)
        return DataHandler.normalize_ohlcv(df, col_map)


class HyperliquidDataProvider(BaseDataProvider):
    """
    Utile pour le LIVE. Récupère les données en temps réel via l'API.
    """
    def __init__(self, info_client):
        super().__init__()
        self.info = info_client

    def fetch_candles(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        # Ici tu placeras ta logique de AlgBotGpt.py pour appeler l'API
        # hl_map = {'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        # candles = self.info.candle_snapshot(...)
        # return DataHandler.normalize_ohlcv(pd.DataFrame(candles), hl_map)
        pass


class APIDataProvider(BaseDataProvider):
    """
    Fournisseur de données via API REST générique.
    """
    
    def __init__(self, api_url, api_key=None):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
    
    def fetch_data(self, params):
        """
        Récupère les données depuis l'API.
        """
        pass


class CSVDataProvider(BaseDataProvider):
    """
    Fournisseur de données depuis des fichiers CSV locaux.
    Alias de LocalDataProvider pour compatibilité.
    """
    
    def __init__(self, data_directory):
        super().__init__()
        self.data_directory = data_directory
    
    def fetch_data(self, params):
        """
        Récupère les données depuis des fichiers CSV.
        """
        pass


class DatabaseDataProvider(BaseDataProvider):
    """
    Fournisseur de données depuis une base de données.
    """
    
    def __init__(self, db_connection):
        super().__init__()
        self.db_connection = db_connection
    
    def fetch_data(self, params):
        """
        Récupère les données depuis la base de données.
        """
        pass
    
    def write_data(self, data):
        """
        Écrit des données dans la base de données.
        """
        pass
