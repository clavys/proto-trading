import pandas as pd
import numpy as np
from typing import Optional

class DataHandler:
    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame, col_map: dict = None) -> pd.DataFrame:
        """Standardise les colonnes pour les stratégies."""
        if col_map:
            df = df.rename(columns=col_map)
        
        # Colonnes minimales requises
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Conversion du timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Conversion numérique
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[required].reset_index(drop=True)

class HyperliquidDataProvider:
    """
    Spécifique à Hyperliquid, utilise la logique que tu as dans AlgBotGpt.py
    """
    def __init__(self, info_client):
        self.info = info_client

    def fetch_candles(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        # On peut réutiliser ici ta logique de AlgBotGpt.py
        # Mais on retourne un DataFrame déjà nettoyé par DataHandler
        
        # Simulation de la récupération (logique simplifiée de ton AlgBotGpt.py)
        # candles = self.info.candle_snapshot(...)
        
        # Mapping spécifique à Hyperliquid
        # t: timestamp (ms), o: open, h: high, l: low, c: close, v: volume
        hl_map = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }
        
        # Exemple de transformation
        # df = DataHandler.normalize_ohlcv(pd.DataFrame(candles), hl_map)
        # return df
        pass

class LocalDataProvider:
    """Charge des données depuis le disque (CSV ou Parquet)."""
    
    def load_from_file(self, file_path: str, col_map: dict = None) -> pd.DataFrame:
        """
        Charge un fichier et le normalise.
        :param file_path: Chemin vers le fichier .csv ou .parquet
        :param col_map: Mapping si les noms de colonnes diffèrent (ex: {'Date': 'timestamp'})
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

        extension = file_path.split('.')[-1].lower()
        
        if extension == 'csv':
            df = pd.read_csv(file_path)
        elif extension == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Format de fichier non supporté (CSV ou Parquet uniquement).")

        return DataHandler.normalize_ohlcv(df, col_map)
    
class DataHandler:
    @staticmethod
    def normalize_binance_klines(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit les klines brutes de Binance vers le format standard.
        """
        # 1. On ne garde que les 6 premières colonnes
        df = df.iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # 2. Conversion du timestamp 
        # Si ton timestamp a 16 chiffres (microsecondes), on utilise unit='us'
        # S'il en a 13 (millisecondes), on utilise unit='ms'
        ts_length = len(str(df['timestamp'].iloc[0]))
        unit = 'us' if ts_length > 13 else 'ms'
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit)
        
        # 3. Conversion des prix en nombres (float)
        cols_to_fix = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_fix:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]