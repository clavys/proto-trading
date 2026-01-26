"""
Data Handler Module
Transforme les données brutes (venant de fichiers CSV, de l'API Hyperliquid
ou d'autres échanges) en un format standardisé.
"""

import pandas as pd
import numpy as np
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
