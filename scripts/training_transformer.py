import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 

# Ajusta el path según tu estructura de carpetas original
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels

# -------------------------------------------------------------------------
# 1. Carga y Preprocesamiento de Datos
# -------------------------------------------------------------------------

print("--- Cargando Dataset ---")
X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

# Padding
X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)

X_val_padded = pad_sequences(X_val, max_len=X_train_padded.shape[1])
Y_val_padded = pad_labels(Y_val, max_len=Y_train_padded.shape[1])

# Conversión de tipos explícita
X_train_padded = X_train_padded.astype(np.float32)
X_val_padded = X_val_padded.astype(np.float32)
Y_train_padded = Y_train_padded.astype(np.int64)
Y_val_padded = Y_val_padded.astype(np.int64)

# Creación de Tensores
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_padded, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_padded, dtype=torch.long)

# Dataset y DataLoader
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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8)

# -------------------------------------------------------------------------
# 2. Arquitectura del Modelo: AudioTransNet (CNN + Transformer)
# -------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crear matriz de codificación posicional constante
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AudioTransNet(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, output_dim=4, dropout=0.1):
        super(AudioTransNet, self).__init__()
        
        # --- A. Feature Extractor (CNN) ---
        # Reduce ruido y proyecta features antes del Transformer
        self.conv_embedding = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # --- B. Positional Encoding ---
        # Da noción de "tiempo" y "orden" a la secuencia
        self.pos_encoder = PositionalEncoding(d_model, max_len=2000, dropout=dropout)
        
        # --- C. Transformer Encoder ---
        # Captura dependencias globales (Self-Attention)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # --- D. Clasificador ---
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x input: (Batch, Time, Freq) -> (Batch, 666, 252)
        
        # 1. Adaptar para CNN (Batch, Freq, Time)
        x = x.permute(0, 2, 1) 
        
        # 2. CNN
        x = self.conv_embedding(x) # -> (Batch, d_model, Time)
        
        # 3. Adaptar para Transformer (Batch, Time, d_model)
        x = x.permute(0, 2, 1)
        
        # 4. Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # 5. Salida
        out = self.fc(x) # -> (Batch, Time, Output_dim)
        return out

# -------------------------------------------------------------------------
# 3. Configuración del Entrenamiento
# -------------------------------------------------------------------------

input_dim = X_train_tensor.shape[2] # 252
BEST_HIDDEN_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciar Modelo
model = AudioTransNet(input_dim, d_model=BEST_HIDDEN_DIM, nhead=4, num_layers=3)
model.to(device)

criterion = nn.CrossEntropyLoss()
BEST_LR = 0.0009341062209856291 
optimizer = optim.Adam(model.parameters(), lr=BEST_LR)

# Configuración Dinámica de Guardado
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
MODEL_PATH = f'best_model_transformer_{timestamp}.pth'
PLOT_LOSS_NAME = f'loss_plot_{timestamp}.png'
PLOT_CM_NAME = f'confusion_matrix_{timestamp}.png'

# Parámetros Avanzados
n_epochs = 100               # Aumentado para dar tiempo al Transformer
patience_limit = 15          # Early Stopping
patience_counter = 0
BEST_VAL_LOSS = float('inf')

# Scheduler: Reduce el LR si la pérdida se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_losses = []
val_losses = []

# -------------------------------------------------------------------------
# 4. Loop de Entrenamiento
# -------------------------------------------------------------------------
print(f"--- Iniciando entrenamiento en {device} ---")
print(f"Modelo: AudioTransNet | Epochs: {n_epochs} | Output: {MODEL_PATH}")

for epoch in range(n_epochs):
    # --- Training ---
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Flatten para CrossEntropy: (Batch*Time, Classes) vs (Batch*Time)
        loss = criterion(outputs.reshape(-1, 4), Y_batch.view(-1))
        loss.backward()
        
        # Clip Gradients: Crucial para estabilidad en Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # --- Validation ---
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.reshape(-1, 4), Y_batch.view(-1))
            val_loss += loss.item()
            
            # Métricas
            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            targets = Y_batch.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Info y Scheduler
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{n_epochs} | LR: {current_lr:.6f} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    scheduler.step(avg_val_loss)

    # --- Guardado y Early Stopping ---
    if avg_val_loss < BEST_VAL_LOSS:
        print(f"  ✅ Mejora detectada ({BEST_VAL_LOSS:.4f} -> {avg_val_loss:.4f}). Guardando en {MODEL_PATH}...")
        BEST_VAL_LOSS = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ⚠️ Sin mejora ({patience_counter}/{patience_limit})")
        
    if patience_counter >= patience_limit:
        print("⛔ Early Stopping activado. Deteniendo entrenamiento.")
        break

print("\n--- Entrenamiento Finalizado ---")

# -------------------------------------------------------------------------
# 5. Análisis de Resultados
# -------------------------------------------------------------------------

# Gráfica de Pérdida
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'Training vs Validation Loss\nBest Val: {BEST_VAL_LOSS:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(PLOT_LOSS_NAME)
print(f"Gráfica guardada: {PLOT_LOSS_NAME}")

# Matriz de Confusión
cm = confusion_matrix(all_targets, all_preds, labels=np.arange(4))
print("\nMatriz de Confusión (Última Época):")
print(cm)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
plt.title('Matriz de Confusión - AudioTransNet')
plt.colorbar()
tick_marks = np.arange(4)
plt.xticks(tick_marks, ['0 (Sil)', '1 (Do)', '2 (Mi)', '3 (Sol)'])
plt.yticks(tick_marks, ['0', '1', '2', '3'])

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig(PLOT_CM_NAME)
print(f"Matriz guardada: {PLOT_CM_NAME}")
plt.show()