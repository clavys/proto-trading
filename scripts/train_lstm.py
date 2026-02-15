#!/usr/bin/env python3
"""
Script d'entraînement pour le modèle PriceLSTM.
Entraîne un LSTM pour la prédiction de prix BTC/USDT.

OPTIMISATIONS:
- Mixed Precision (AMP) pour 2x speedup
- Learning Rate Warmup + Cosine Annealing
- Gradient Clipping pour stabilité
- Early Stopping pour converge rapide
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import joblib
import argparse
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time

# Importer depuis le projet
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ia.lstm_models import PriceLSTM


class PriceDataset(Dataset):
    """Dataset custom pour les séquences de prix."""
    
    def __init__(self, data, seq_length=60, features=['close']):
        """
        Args:
            data: DataFrame avec colonnes OHLCV
            seq_length: Longueur des séquences
            features: Colonnes à utiliser
        """
        self.seq_length = seq_length
        self.features = features
        
        # Normaliser les données avec MinMaxScaler
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(data[features])
        
        # Pré-allocate arrays pour performance
        self.sequences = torch.FloatTensor(
            len(self.scaled_data) - seq_length, seq_length, len(features)
        )
        self.targets = torch.FloatTensor(len(self.scaled_data) - seq_length, 1)
        
        # Remplir les sequences en une seule opération (plus rapide)
        for i in range(len(self.scaled_data) - seq_length):
            self.sequences[i] = torch.from_numpy(
                self.scaled_data[i:i + seq_length].astype(np.float32)
            )
            self.targets[i, 0] = self.scaled_data[i + seq_length, 0]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMTrainer:
    """Classe pour entraîner le modèle LSTM avec optimisations."""
    
    def __init__(self, model, device, learning_rate=0.001, use_amp=False, warmup_epochs=5):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'  # AMP seulement sur GPU
        self.scaler = GradScaler() if self.use_amp else None
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        
        # Learning Rate Scheduler avec Warmup
        self.warmup_epochs = warmup_epochs
        self.base_lr = learning_rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=learning_rate * 0.01
        )
    
    def _get_warmup_lr(self, epoch, total_warmup_epochs, base_lr):
        """Calcul du learning rate avec warmup linéaire."""
        if epoch < total_warmup_epochs:
            return base_lr * (epoch + 1) / total_warmup_epochs
        else:
            return base_lr
    
    def train_epoch(self, train_loader, epoch, total_warmup_epochs):
        """Entraîne une époque avec AMP optionnel."""
        self.model.train()
        total_loss = 0
        
        # Update LR for warmup
        if epoch < total_warmup_epochs:
            new_lr = self._get_warmup_lr(epoch, total_warmup_epochs, self.base_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(dtype=torch.float16):
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Évalue le modèle sur le validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_amp:
                    with autocast(dtype=torch.float16):
                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """Entraîne le modèle avec early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'Epoch':<8}{'LR':<12}{'Train Loss':<15}{'Val Loss':<15}{'Status':<20}")
        print("-" * 70)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch, self.warmup_epochs)
            val_loss = self.validate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler après warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                status = "✓ Best"
            else:
                patience_counter += 1
                status = f"No improve ({patience_counter}/{patience})"
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"{epoch+1:<8}{current_lr:<12.6f}{train_loss:<15.6f}{val_loss:<15.6f}{status:<20}")
            
            if patience_counter >= patience:
                print(f"\n→ Early stopping à l'époque {epoch+1}")
                break
        
        elapsed = time.time() - start_time
        return best_val_loss, elapsed


def load_and_prepare_data(data_path, test_split=0.2, val_split=0.1):
    """Charge et prépare les données pour l'entraînement."""
    print(f"Chargement des données depuis {data_path}...")
    
    df = pd.read_csv(data_path)
    
    # Convertir les colonnes de temps si nécessaire
    if 'Open time' in df.columns:
        df['Open time'] = pd.to_datetime(df['Open time'])
        df = df.sort_values('Open time')
    
    # Garder seulement les colonnes OHLCV
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [c for c in ohlcv_cols if c in df.columns]
    
    if 'Close' not in available_cols:
        raise ValueError(f"Colonnes disponibles: {df.columns.tolist()}")
    
    df = df[available_cols].copy()
    df = df.dropna()
    
    print(f"✓ {len(df)} candlesticks chargées")
    print(f"Colonnes utilisées: {available_cols}")
    
    # Split train/val/test
    total_len = len(df)
    test_size = int(total_len * test_split)
    val_size = int((total_len - test_size) * val_split)
    train_size = total_len - test_size - val_size
    
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size + val_size]
    df_test = df.iloc[train_size + val_size:]
    
    print(f"Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    return df_train, df_val, df_test, available_cols


def main():
    parser = argparse.ArgumentParser(description="Entraîner le modèle PriceLSTM")
    parser.add_argument("--data", type=str, default="data/raw/BTCUSDT-1m-2025-01.csv",
                        help="Chemin vers le fichier CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    parser.add_argument("--seq-length", type=int, default=60, help="Longueur de séquence")
    parser.add_argument("--hidden-size", type=int, default=64, help="Taille couche cachée")
    parser.add_argument("--num-layers", type=int, default=2, help="Nombre de couches LSTM")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--dropout", type=float, default=0.2, help="Taux de dropout")
    parser.add_argument("--output-dir", type=str, default="models", help="Répertoire de sauvegarde")
    parser.add_argument("--device", type=str, default="auto", 
                        help="'cuda', 'cpu', ou 'auto'")
    parser.add_argument("--amp", action="store_true", 
                        help="Activer Mixed Precision (AMP) pour speedup")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Nombre d'épocas de warmup")
    
    args = parser.parse_args()
    
    # Configuration du device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Entraînement LSTM sur {device}")
    if args.amp and device.type == 'cuda':
        print("⚡ Mixed Precision (AMP) activé")
    
    # Créer le répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Charger les données
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Fichier introuvable: {data_path}")
        return
    
    df_train, df_val, df_test, features = load_and_prepare_data(str(data_path))
    
    # Créer les datasets
    print("\nPréparation des datasets...")
    train_dataset = PriceDataset(df_train, seq_length=args.seq_length, features=features)
    val_dataset = PriceDataset(df_val, seq_length=args.seq_length, features=features)
    test_dataset = PriceDataset(df_test, seq_length=args.seq_length, features=features)
    
    print(f"Samples: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Créer les DataLoaders avec pin_memory pour GPU
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        num_workers=0
    )
    
    # Créer le modèle
    model = PriceLSTM(
        input_size=len(features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seq_length=args.seq_length
    )
    
    print(f"\nArchitecture:")
    print(f"  Input size: {len(features)} (features: {features})")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Seq length: {args.seq_length}")
    
    # Entraîner
    trainer = LSTMTrainer(
        model, 
        device, 
        learning_rate=args.learning_rate,
        use_amp=args.amp,
        warmup_epochs=args.warmup_epochs
    )
    best_val_loss, elapsed = trainer.train(
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        patience=15
    )
    
    # Évaluer sur le test set
    print("\nÉvaluation sur test set...")
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Temps écoulé
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"⏱️ Temps total: {hours}h {minutes}m {seconds}s")
    
    # Sauvegarder le modèle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lstm_btc_{timestamp}"
    
    model_path = output_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Modèle sauvegardé: {model_path}")
    
    # Sauvegarder le scaler
    scaler_path = output_dir / f"{model_name}_scaler.pkl"
    joblib.dump(train_dataset.scaler, scaler_path)
    print(f"✓ Scaler sauvegardé: {scaler_path}")
    
    # Sauvegarder la configuration
    config = {
        'input_size': len(features),
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'seq_length': args.seq_length,
        'dropout': args.dropout,
        'features': features,
        'test_loss': float(test_loss),
        'best_val_loss': float(best_val_loss),
        'trained_on': str(data_path),
        'timestamp': timestamp,
        'device': str(device),
        'amp_enabled': args.amp and device.type == 'cuda',
        'training_time_seconds': elapsed
    }
    
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config sauvegardée: {config_path}")
    
    print(f"\n✅ Entraînement terminé!")
    print(f"Modèle: {model_name}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"À utiliser avec PricePredictor:")
    print(f"  model_path='{model_path}'")
    print(f"  scaler_path='{scaler_path}'")
    print(f"  config_path='{config_path}'")


if __name__ == "__main__":
    main()
