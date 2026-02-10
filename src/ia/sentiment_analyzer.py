import torch
from transformers import pipeline
import pandas as pd
from datetime import datetime
from pathlib import Path

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert", sentiment_csv=None):
        """
        Initialise le SentimentAnalyzer
        
        Args:
            model_name: Modèle HuggingFace à utiliser (défaut: FinBERT)
            sentiment_csv: Chemin vers CSV précalculé pour backtest (None = mode live)
        """
        self.mode = "backtest" if sentiment_csv else "live"
        self.sentiment_data = None
        
        if self.mode == "backtest":
            # Mode backtest : charge les sentiments précalculés
            print(f"Mode BACKTEST : Chargement des sentiments depuis {sentiment_csv}")
            self.sentiment_data = pd.read_csv(sentiment_csv)
            self.sentiment_data['timestamp'] = pd.to_datetime(self.sentiment_data['timestamp'])
            self.sentiment_data.set_index('timestamp', inplace=True)
            print(f"✓ {len(self.sentiment_data)} sentiments chargés")
            self.classifier = None
        else:
            # Mode live : initialise le modèle IA
            print("Mode LIVE : Initialisation du modèle IA...")
            self.device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if self.device == 0 else "CPU"
            print(f"Initialisation de FinBERT sur {device_name}...")
            
            self.classifier = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                device=self.device
            )
    
    def analyze(self, text, timestamp=None):
        """
        Analyse le sentiment d'un texte
        
        Args:
            text: Texte à analyser
            timestamp: Pour mode backtest, récupère le sentiment précalculé
            
        Returns:
            dict: {'label': 'positive/negative/neutral', 'score': float}
        """
        if self.mode == "backtest":
            if timestamp is None:
                raise ValueError("timestamp requis en mode backtest")
            
            # Recherche du sentiment le plus proche dans le temps
            timestamp = pd.to_datetime(timestamp)
            
            # Trouve les news dans une fenêtre de ±1h
            window_start = timestamp - pd.Timedelta(hours=1)
            window_end = timestamp + pd.Timedelta(hours=1)
            
            window_data = self.sentiment_data[
                (self.sentiment_data.index >= window_start) & 
                (self.sentiment_data.index <= window_end)
            ]
            
            if len(window_data) == 0:
                # Pas de news dans la fenêtre, retourne neutre
                return {'label': 'neutral', 'score': 0.5}
            
            # Calcule le sentiment moyen pondéré par les scores
            sentiment_counts = window_data['sentiment'].value_counts()
            dominant_sentiment = sentiment_counts.idxmax()
            avg_score = window_data[window_data['sentiment'] == dominant_sentiment]['score'].mean()
            
            return {'label': dominant_sentiment, 'score': avg_score}
        
        else:
            # Mode live : analyse en temps réel
            result = self.classifier(text)[0]
            return result
    
    def get_sentiment_at_time(self, timestamp, window_hours=1):
        """
        Récupère toutes les news dans une fenêtre temporelle (mode backtest)
        
        Args:
            timestamp: Timestamp de référence
            window_hours: Fenêtre en heures (défaut: 1h)
            
        Returns:
            DataFrame avec les news et sentiments dans la fenêtre
        """
        if self.mode != "backtest":
            raise ValueError("Cette méthode n'est disponible qu'en mode backtest")
        
        timestamp = pd.to_datetime(timestamp)
        window_start = timestamp - pd.Timedelta(hours=window_hours)
        window_end = timestamp
        
        return self.sentiment_data[
            (self.sentiment_data.index >= window_start) & 
            (self.sentiment_data.index <= window_end)
        ]