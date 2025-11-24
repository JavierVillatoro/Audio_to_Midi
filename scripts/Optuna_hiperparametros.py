import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import optuna # <-- Importar Optuna

# Asegúrate de que las utilidades estén disponibles
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels

# -------------------------
# Cargar y Preparar Dataset (Se mantiene igual)
# -------------------------
X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)

X_val_padded = pad_sequences(X_val, max_len=X_train_padded.shape[1])
Y_val_padded = pad_labels(Y_val, max_len=Y_train_padded.shape[1])

X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_padded, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_padded, dtype=torch.long)

class AudioMIDIDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = AudioMIDIDataset(X_train_tensor, Y_train_tensor)
val_dataset   = AudioMIDIDataset(X_val_tensor, Y_val_tensor)

# El DataLoader se define DENTRO de la función objective para usar el batch_size sugerido
# y el device se mantiene global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Modelo CRNN (Se mantiene igual)
# -------------------------
class CRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# ----------------------------------------------------
# 1. FUNCIÓN OBJETIVO DE OPTUNA
# ----------------------------------------------------
def objective(trial):
    # Definición de Hiperparámetros a Optimizar
    
    # 1. Hiperparámetros del Modelo (Arquitectura)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
    
    # 2. Hiperparámetros de Entrenamiento
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32]) # Sugiere potencias de 2
    
    # Inicializar modelo y mover al dispositivo
    input_dim = X_train_tensor.shape[2]
    model = CRNN(input_dim, hidden_dim=hidden_dim)
    model.to(device)

    # Dataloaders con el batch_size sugerido
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size)

    # Definir Criterio y Optimizador con el lr sugerido
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Parámetros de Entrenamiento Fijo para el Estudio
    # n_epochs debe ser bajo (ej. 5) o usar Poda/Early Stopping para acelerar el proceso
    n_epochs = 5 

    # Loop de Entrenamiento Simplificado
    for epoch in range(n_epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, 4), Y_batch.view(-1))
            loss.backward()
            optimizer.step()
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.view(-1, 4), Y_batch.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # **Reporte Intermedio y Poda de Optuna**
        # Reporta la métrica de validación a Optuna
        trial.report(avg_val_loss, epoch)

        # Revisa si Optuna quiere detener este trial
        if trial.should_prune():
            # Levanta la excepción para descartar este trial
            raise optuna.exceptions.TrialPruned()

    # Optuna SIEMPRE debe devolver la métrica final que quieres optimizar
    return avg_val_loss # Queremos MINIMIZAR la pérdida de validación

# ----------------------------------------------------
# 2. EJECUTAR EL ESTUDIO DE OPTUNA
# ----------------------------------------------------
if __name__ == '__main__':
    # Usaremos un pruner para detener ensayos poco prometedores temprano
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3) 

    # Crear el estudio: queremos MINIMIZAR la pérdida de validación
    study = optuna.create_study(direction='minimize', pruner=pruner)

    # Ejecutar 50 ensayos (trials)
    print("Iniciando la optimización de hiperparámetros con Optuna...")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Imprimir los resultados
    print("\n---------------------------------------------------")
    print(" Optimización Finalizada ")
    print("---------------------------------------------------")
    print(f"Mejor Pérdida de Validación: {study.best_value:.4f}")
    print("Mejores Hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Puedes acceder al mejor modelo con study.best_trial
    # print(study.best_trial)