import pandas as pd
import numpy as np
import os
from typing import Optional

class DataHandler:
    """
    Cette classe est le 'cerveau' de la standardisation. 
    Elle transforme les données brutes (Binance, Hyperliquid, CSV) en format utilisable par tes stratégies.
    """
    
    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame, col_map: dict = None) -> pd.DataFrame:
        """Standardise les colonnes pour les stratégies (OHLCV classique)."""
        if col_map:
            df = df.rename(columns=col_map)
        
        # Colonnes minimales requises pour que pandas_ta et tes stratégies fonctionnent
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Nettoyage et tri
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Conversion forcée en numérique pour éviter les erreurs de calcul (SMA, etc.)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[required].reset_index(drop=True)

    @staticmethod
    def normalize_binance_klines(df: pd.DataFrame) -> pd.DataFrame:
        """
        Spécifique au format brut des fichiers CSV de Binance (sans headers).
        """
        # On ne garde que les 6 premières colonnes (OHLCV + Timestamp)
        df = df.iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Détection automatique de l'unité du timestamp (ms ou us)
        ts_sample = str(df['timestamp'].iloc[0])
        unit = 'us' if len(ts_sample) > 13 else 'ms'
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit)
        
        # Conversion numérique
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

class LocalDataProvider:
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

class HyperliquidDataProvider:
    """
    Utile pour le LIVE. Récupère les données en temps réel via l'API.
    """
    def __init__(self, info_client):
        self.info = info_client

    def fetch_candles(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        # Ici tu placeras ta logique de AlgBotGpt.py pour appeler l'API
        # hl_map = {'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        # candles = self.info.candle_snapshot(...)
        # return DataHandler.normalize_ohlcv(pd.DataFrame(candles), hl_map)
        pass