import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import datetime
import json
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                             classification_report, balanced_accuracy_score)
import matplotlib.pyplot as plt 
from tqdm import tqdm
import optuna

# -------------------------------------------------------------------------
# 0. CONFIGURACIÓN E IMPORTACIONES
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock functions por si no tienes los archivos locales a mano
try:
    from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels
except ImportError:
    print("ADVERTENCIA: Usando funciones Mock (asegúrate de tener tus datos reales).")
    def load_full_dataset(x, y): return np.random.rand(100, 300, 252), np.random.randint(0, 4, (100, 300))
    def split_dataset(x, y): return x, y, x, y, x, y
    def pad_sequences(x, max_len=None): return x
    def pad_labels(y, max_len=None): return y

# Configuración global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 10         # Número de intentos de Optuna
N_EPOCHS_OPTUNA = 10  # Épocas rápidas por intento
FINAL_EPOCHS = 50     # Épocas del entrenamiento final
OUTPUT_DIM = 4 

print(f"--- Usando dispositivo: {DEVICE} ---")

# -------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO
# -------------------------------------------------------------------------
if os.path.exists("dataset_X.npy"):
    print("Cargando datasets .npy...")
    X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
else:
    print("Generando datos dummy para prueba...")
    X, Y = np.random.rand(100, 300, 252), np.random.randint(0, 4, (100, 300))

X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

# Padding y Tensores
X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)
max_len = X_train_padded.shape[1]
X_val_padded = pad_sequences(X_val, max_len=max_len)
Y_val_padded = pad_labels(Y_val, max_len=max_len)

X_train_tensor = torch.tensor(X_train_padded.astype(np.float32))
Y_train_tensor = torch.tensor(Y_train_padded.astype(np.int64))
X_val_tensor = torch.tensor(X_val_padded.astype(np.float32))
Y_val_tensor = torch.tensor(Y_val_padded.astype(np.int64))
    
INPUT_DIM = X_train_tensor.shape[2] 

# Pesos de clase
def calculate_class_weights_soft(y_tensor, power=0.75):
    y_flat = y_tensor.flatten().numpy()
    classes, counts = np.unique(y_flat, return_counts=True)
    weights = 1.0 / (counts ** power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

CLASS_WEIGHTS = calculate_class_weights_soft(Y_train_tensor)

class AudioMIDIDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

train_dataset = AudioMIDIDataset(X_train_tensor, Y_train_tensor)
val_dataset   = AudioMIDIDataset(X_val_tensor, Y_val_tensor)

# -------------------------------------------------------------------------
# 2. ARQUITECTURA (SE-BLOCK + UNet + BiLSTM)
# -------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DoubleConvSE(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x

class AudioUNetWithBiLSTMEnd(nn.Module):
    def __init__(self, n_channels, n_classes, lstm_hidden_size=64, dropout_rate=0.1):
        super(AudioUNetWithBiLSTMEnd, self).__init__()
        self.inc = DoubleConvSE(n_channels, 64, dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConvSE(64, 128, dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConvSE(128, 256, dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConvSE(256, 512, dropout_rate))
        
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConvSE(512, 256, dropout_rate)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConvSE(256, 128, dropout_rate)
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConvSE(128, 64, dropout_rate)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, 
                            num_layers=2, batch_first=True, bidirectional=True,
                            dropout=dropout_rate if dropout_rate < 1 else 0)
        self.outc = nn.Conv1d(lstm_hidden_size * 2, n_classes, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = self._pad_and_cat(x, x3)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = self._pad_and_cat(x, x2)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = self._pad_and_cat(x, x1)
        x_cnn_out = self.conv_up3(x) 
        
        x_lstm_in = x_cnn_out.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        x_lstm_out, _ = self.lstm(x_lstm_in)
        
        logits = self.outc(x_lstm_out.permute(0, 2, 1))
        return logits.permute(0, 2, 1)

    def _pad_and_cat(self, x_up, x_skip):
        diff = x_skip.size(2) - x_up.size(2)
        if diff != 0: x_up = F.pad(x_up, [diff // 2, diff - diff // 2])
        return torch.cat([x_skip, x_up], dim=1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

# -------------------------------------------------------------------------
# 3. OPTUNA Y UTILS
# -------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    acc = accuracy_score(y_true_flat, y_pred_flat)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    return acc, f1_macro

def objective(trial):
    # Espacio de búsqueda
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128])
    batch_size = trial.suggest_categorical("batch_size", [4, 8]) 
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
    gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)

    t_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    v_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = AudioUNetWithBiLSTMEnd(INPUT_DIM, OUTPUT_DIM, lstm_hidden, dropout_rate).to(DEVICE)
    criterion = FocalLoss(alpha=CLASS_WEIGHTS, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Barra de progreso para las épocas dentro de un Trial (leave=False para que desaparezca al terminar)
    epoch_pbar = tqdm(range(N_EPOCHS_OPTUNA), desc=f"Trial {trial.number}", leave=False, position=1)
    
    for epoch in epoch_pbar:
        model.train()
        for X_b, Y_b in t_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs.reshape(-1, OUTPUT_DIM), Y_b.view(-1))
            loss.backward()
            optimizer.step()
        
        # Validación
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_b, Y_b in v_loader:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                out = model(X_b)
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(Y_b.cpu().numpy())
        
        y_p = np.concatenate(all_preds).flatten()
        y_t = np.concatenate(all_targets).flatten()
        _, f1_m = compute_metrics(y_t, y_p)
        
        # Actualizar info en la barra del trial
        epoch_pbar.set_postfix(f1_val=f"{f1_m:.4f}")
        
        trial.report(f1_m, epoch)
        if trial.should_prune(): 
            raise optuna.TrialPruned()

    return f1_m

# -------------------------------------------------------------------------
# 4. EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"\n--- 1. INICIANDO BÚSQUEDA OPTUNA ({N_TRIALS} Trials) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    # Barra de progreso principal para los Trials de Optuna
    with tqdm(total=N_TRIALS, desc="Progreso Global Optuna", position=0) as pbar:
        def callback(study, trial):
            pbar.update(1)
            try:
                pbar.set_postfix(best_f1=f"{study.best_value:.4f}")
            except: pass # Si es el primer trial a veces falla

        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    best_params = study.best_params
    print(f"\n✅ Optuna finalizado. Mejores Parámetros: {best_params}")

    # -------------------------------------------------------------------------
    # 5. RE-ENTRENAMIENTO PROFESIONAL
    # -------------------------------------------------------------------------
    print("\n--- 2. ENTRENANDO MODELO DEFINITIVO (Best Params) ---")
    final_bs = best_params["batch_size"]
    loader_train = DataLoader(train_dataset, batch_size=final_bs, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=final_bs)
    
    final_model = AudioUNetWithBiLSTMEnd(
        INPUT_DIM, OUTPUT_DIM, 
        lstm_hidden_size=best_params["lstm_hidden"], 
        dropout_rate=best_params["dropout"]
    ).to(DEVICE)
    
    final_criterion = FocalLoss(alpha=CLASS_WEIGHTS, gamma=best_params["focal_gamma"])
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='max', factor=0.5, patience=4)

    best_val_f1 = 0
    best_val_loss = float('inf')
    
    train_losses, val_losses = [], []

    # BARRA PRINCIPAL DEL ENTRENAMIENTO FINAL
    outer_pbar = tqdm(range(FINAL_EPOCHS), desc="Entrenamiento Final", position=0)

    for epoch in outer_pbar:
        final_model.train()
        t_loss = 0
        
        # Barra interna para los batches
        inner_pbar = tqdm(loader_train, desc=f"Ep {epoch+1} Batches", leave=False, position=1)
        
        for X_b, Y_b in inner_pbar:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            final_optimizer.zero_grad()
            out = final_model(X_b)
            loss = final_criterion(out.reshape(-1, OUTPUT_DIM), Y_b.view(-1))
            loss.backward()
            final_optimizer.step()
            t_loss += loss.item()
            
            # Mostrar loss instantáneo en la barra de batches
            inner_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_losses.append(t_loss / len(loader_train))
        
        # Validación
        final_model.eval()
        v_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_b, Y_b in loader_val:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                out = final_model(X_b)
                loss = final_criterion(out.reshape(-1, OUTPUT_DIM), Y_b.view(-1))
                v_loss += loss.item()
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(Y_b.cpu().numpy())
        
        avg_val_loss = v_loss / len(loader_val)
        val_losses.append(avg_val_loss)
        
        y_p = np.concatenate(all_preds).flatten()
        y_t = np.concatenate(all_targets).flatten()
        acc, f1_m = compute_metrics(y_t, y_p)
        
        scheduler.step(f1_m)
        
        # Actualizar la barra principal con las métricas de validación
        outer_pbar.set_postfix(val_loss=f"{avg_val_loss:.4f}", val_f1=f"{f1_m:.4f}")
        
        # Guardar mejor modelo basado en F1 Macro
        if f1_m > best_val_f1:
            best_val_f1 = f1_m
            best_val_loss = avg_val_loss 
            torch.save(final_model.state_dict(), f"best_model_{timestamp}.pth")

    # -------------------------------------------------------------------------
    # 6. GENERACIÓN DE REPORTE Y GUARDADO DE DATOS
    # -------------------------------------------------------------------------
    print("\n--- 3. GENERANDO REPORTE FINAL ---")
    
    # Cargar el mejor modelo (no necesariamente el de la última época)
    final_model.load_state_dict(torch.load(f"best_model_{timestamp}.pth"))
    final_model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, Y_b in loader_val:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            out = final_model(X_b)
            preds = torch.argmax(out, dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(Y_b.cpu().numpy())

    y_p = np.concatenate(all_preds).flatten()
    y_t = np.concatenate(all_targets).flatten()
    
    # Métricas "PRO"
    final_acc = accuracy_score(y_t, y_p)
    final_f1_macro = f1_score(y_t, y_p, average='macro', zero_division=0)
    final_f1_weighted = f1_score(y_t, y_p, average='weighted', zero_division=0)
    final_bal_acc = balanced_accuracy_score(y_t, y_p)
    
    # Nombres de clases (AJUSTAR SEGÚN TUS NOTAS)
    class_names = ['Silencio (0)', 'Clase 1', 'Clase 2', 'Clase 3']
    if len(np.unique(y_t)) > len(class_names): # Si hay más clases de las esperadas
         class_names = [f"Class {i}" for i in range(OUTPUT_DIM)]

    report_text = classification_report(y_t, y_p, target_names=class_names, digits=4)
    
    # --- ESCRITURA DEL ARCHIVO DE RESULTADOS ---
    filename = f"resultados_final_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("======================================================\n")
        f.write(f"REPORTE FINAL DE ENTRENAMIENTO - {timestamp}\n")
        f.write("======================================================\n\n")
        
        f.write("1. HIPERPARÁMETROS GANADORES (OPTUNA):\n")
        f.write(json.dumps(best_params, indent=4))
        f.write("\n\n")
        
        f.write("2. CONFIGURACIÓN DEL ENTRENAMIENTO FINAL:\n")
        f.write(f"   - Epochs Totales: {FINAL_EPOCHS}\n")
        f.write(f"   - Class Weights (Power 0.75): {CLASS_WEIGHTS.cpu().numpy()}\n")
        f.write(f"   - Device: {DEVICE}\n\n")
        
        f.write("3. MÉTRICAS GLOBALES (MEJOR MODELO EN VALIDACIÓN):\n")
        f.write(f"   - Best Val Loss (Focal):  {best_val_loss:.6f}\n")
        f.write(f"   - Accuracy Global:        {final_acc:.6f}\n")
        f.write(f"   - Balanced Accuracy:      {final_bal_acc:.6f}\n")
        f.write(f"   - F1-Score Macro:         {final_f1_macro:.6f}\n")
        f.write(f"   - F1-Score Weighted:      {final_f1_weighted:.6f}\n\n")
        
        f.write("4. REPORTE DE CLASIFICACIÓN DETALLADO:\n")
        f.write(report_text)
        f.write("\n")
        
        f.write("5. MATRIZ DE CONFUSIÓN (TEXTO):\n")
        cm = confusion_matrix(y_t, y_p)
        f.write(str(cm))
        f.write("\n\n======================================================\n")

    print(f"✅ Resultados guardados en: {filename}")
    
    # Gráficas
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Learning Curve (Best F1: {best_val_f1:.4f})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_curve_{timestamp}.png")
    print(f"✅ Gráfica guardada en: loss_curve_{timestamp}.png")