import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple
import logging
import os

logger = logging.getLogger(__name__)

class TradingLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(TradingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_dim, 1)  # Simple attention
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Predict next return (regression)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)
    
    def predict(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            pred = self.forward(x.unsqueeze(0)).item()
        return pred


class ModelTrainer:
    def __init__(self, model: TradingLSTM, lr: float = 0.001, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def prepare_data(self, df: pd.DataFrame, seq_len: int, feature_cols: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for LSTM"""
        data = df[feature_cols].values.astype(np.float32)
        targets = df['returns'].shift(-1).values.astype(np.float32)  # next period return
        
        X, y = [], []
        for i in range(seq_len, len(data) - 1):
            X.append(data[i-seq_len:i])
            y.append(targets[i])
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        
        # Normalize features (per sequence or global - here simple z-score)
        mean = X.mean(dim=(0,1), keepdim=True)
        std = X.std(dim=(0,1), keepdim=True) + 1e-8
        X = (X - mean) / std
        
        return X.to(self.device), y.to(self.device)
    
    def train(self, df: pd.DataFrame, seq_len: int, feature_cols: list, 
              epochs: int = 20, batch_size: int = 64, val_split: float = 0.2) -> dict:
        """Train with early stopping and validation"""
        X, y = self.prepare_data(df, seq_len, feature_cols)
        
        n_val = int(len(X) * val_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val).item()
            
            self.model.train()
            
            avg_train_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        return {'final_val_loss': best_val_loss, 'epochs_trained': epoch + 1}
    
    def fine_tune(self, new_df: pd.DataFrame, seq_len: int, feature_cols: list, epochs: int = 5):
        """Incremental learning on new data"""
        logger.info("Fine-tuning model on new data...")
        X, y = self.prepare_data(new_df, seq_len, feature_cols)
        
        if len(X) < 10:
            logger.warning("Not enough new data for fine-tuning")
            return
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        
        logger.info(f"Fine-tuning completed for {epochs} epochs")
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"No model found at {path}, starting fresh")
