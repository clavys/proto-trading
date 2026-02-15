import torch
import torch.nn as nn

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, seq_length=60):
        """
        Architecture LSTM pour prédiction de prix.
        
        Args:
            input_size: Nombre de features en entrée (1 pour close seul, 5+ pour OHLCV)
            hidden_size: Taille de la couche cachée LSTM
            num_layers: Nombre de couches LSTM empilées
            dropout: Taux de dropout entre les couches LSTM (0 à 1)
            seq_length: Longueur de séquence attendue (pour référence)
        """
        super(PriceLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # LSTM avec dropout entre les couches
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization sur la sortie LSTM
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout avant la couche finale
        self.dropout = nn.Dropout(dropout)
        
        # Couches fully connected avec activation
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor de shape (batch, seq_len, input_size)
            
        Returns:
            Tensor de shape (batch, 1) avec la prédiction de prix
        """
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Prendre la dernière sortie de la séquence
        last_output = lstm_out[:, -1, :]
        
        # Batch normalization
        normalized = self.batch_norm(last_output)
        
        # Couches fully connected avec dropout
        out = self.dropout(normalized)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_config(self):
        """Retourne la configuration du modèle pour la sauvegarde."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'seq_length': self.seq_length
        }