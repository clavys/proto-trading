import torch
import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path
from .lstm_models import PriceLSTM

class PricePredictor:
    def __init__(self, model_path, scaler_path, config_path=None, features=['close']):
        """
        Prédicteur de prix basé sur LSTM.
        
        Args:
            model_path: Chemin vers le fichier .pth contenant les poids du modèle
            scaler_path: Chemin vers le fichier .pkl contenant le scaler
            config_path: Chemin vers le fichier JSON de configuration (optionnel)
            features: Liste des colonnes à utiliser pour la prédiction
        """
        self.features = features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vérifier l'existence des fichiers
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler introuvable: {scaler_path}")
        
        # Charger la configuration si disponible
        self.config = self._load_config(config_path, model_path)
        
        # Charger l'architecture avec la bonne configuration
        self.model = PriceLSTM(
            input_size=self.config.get('input_size', len(features)),
            hidden_size=self.config.get('hidden_size', 64),
            num_layers=self.config.get('num_layers', 2),
            seq_length=self.config.get('seq_length', 60)
        ).to(self.device)
        
        # Charger les poids entraînés
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Charger le scaler (MinMaxScaler) utilisé pendant l'entraînement
        self.scaler = joblib.load(scaler_path)
        
        print(f"✓ Modèle chargé sur {self.device}")
        print(f"✓ Configuration: {self.config}")
    
    def _load_config(self, config_path, model_path):
        """Charge la configuration du modèle depuis un fichier JSON."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Essayer de charger un fichier config à côté du modèle
        default_config_path = Path(model_path).with_suffix('.json')
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                return json.load(f)
        
        # Configuration par défaut
        return {
            'input_size': len(self.features),
            'hidden_size': 64,
            'num_layers': 2,
            'seq_length': 60
        }

    def predict(self, df_slice):
        """
        Prend les n dernières bougies et prédit le prix suivant.
        
        Args:
            df_slice: DataFrame avec au moins les colonnes spécifiées dans self.features
            
        Returns:
            float: Prix prédit
            
        Raises:
            ValueError: Si df_slice n'a pas la bonne forme ou les bonnes colonnes
        """
        # Validation des inputs
        if not isinstance(df_slice, pd.DataFrame):
            raise TypeError("df_slice doit être un pandas DataFrame")
        
        missing_features = [f for f in self.features if f not in df_slice.columns]
        if missing_features:
            raise ValueError(f"Colonnes manquantes dans df_slice: {missing_features}")
        
        expected_seq_len = self.config.get('seq_length', 60)
        if len(df_slice) < expected_seq_len:
            raise ValueError(
                f"df_slice doit contenir au moins {expected_seq_len} lignes, "
                f"mais n'en contient que {len(df_slice)}"
            )
        
        try:
            # 1. Sélectionner les features et normaliser
            data = df_slice[self.features].values[-expected_seq_len:]
            data_scaled = self.scaler.transform(data)
            
            # 2. Créer le tensor (batch_size=1, seq_len, features)
            input_tensor = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
            
            # 3. Inférence
            with torch.no_grad():
                prediction_scaled = self.model(input_tensor)
            
            # 4. Denormalisation pour avoir un prix réel
            # Note: on suppose que la première feature est 'close'
            prediction_array = prediction_scaled.cpu().numpy()
            
            # Si multi-features, créer un array avec la bonne forme pour inverse_transform
            if len(self.features) > 1:
                dummy_array = np.zeros((1, len(self.features)))
                dummy_array[0, 0] = prediction_array[0, 0]
                prediction_price = self.scaler.inverse_transform(dummy_array)[0, 0]
            else:
                prediction_price = self.scaler.inverse_transform(prediction_array)[0, 0]
            
            return float(prediction_price)
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la prédiction: {str(e)}")
    
    def predict_next_n(self, df_slice, n_steps=5):
        """
        Prédiction itérative des n prochains prix (approche autorégressive).
        
        Args:
            df_slice: DataFrame avec les données historiques
            n_steps: Nombre de pas futurs à prédire
            
        Returns:
            list: Liste des n prochains prix prédits
        """
        predictions = []
        current_data = df_slice.copy()
        
        for _ in range(n_steps):
            next_price = self.predict(current_data)
            predictions.append(next_price)
            
            # Ajouter la prédiction aux données pour la prochaine itération
            new_row = current_data.iloc[-1].copy()
            new_row['close'] = next_price
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        return predictions
    
    @staticmethod
    def save_config(model, save_path):
        """
        Sauvegarde la configuration du modèle dans un fichier JSON.
        
        Args:
            model: Instance de PriceLSTM
            save_path: Chemin où sauvegarder (ex: 'model_config.json')
        """
        config = model.get_config()
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Configuration sauvegardée: {save_path}")