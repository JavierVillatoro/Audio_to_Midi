import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from tqdm import tqdm 

# -------------------------------------------------------------------------
# 0. CONFIGURACIÓN DE RUTAS
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels

# -------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# -------------------------------------------------------------------------
print("--- 1. Cargando Dataset (Modo U-Net + BiLSTM Final) ---")

# Cargar archivos .npy
X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

# Padding
print("Aplicando padding...")
X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)

max_len = X_train_padded.shape[1]
X_val_padded = pad_sequences(X_val, max_len=max_len)
Y_val_padded = pad_labels(Y_val, max_len=max_len)

# Conversión a tipos
X_train_padded = X_train_padded.astype(np.float32)
X_val_padded = X_val_padded.astype(np.float32)
Y_train_padded = Y_train_padded.astype(np.int64)
Y_val_padded = Y_val_padded.astype(np.int64)

# Tensores
X_train_tensor = torch.tensor(X_train_padded)
Y_train_tensor = torch.tensor(Y_train_padded)
X_val_tensor = torch.tensor(X_val_padded)
Y_val_tensor = torch.tensor(Y_val_padded)

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
# 2. ARQUITECTURA U-NET + BiLSTM (AL FINAL)
# -------------------------------------------------------------------------
print("--- 2. Inicializando Arquitectura U-Net con BiLSTM Final ---")

class DoubleConv(nn.Module):
    """(Convolución => BatchNorm => ReLU) dos veces"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), 
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AudioUNetWithBiLSTMEnd(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AudioUNetWithBiLSTMEnd, self).__init__()
        
        # --- ENCODER (Igual que antes) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(256, 512))
        
        # --- DECODER (Igual que antes) ---
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        # --- ### NUEVO: CAPA RECURRENTE (Bi-LSTM) ### ---
        # Recibe 64 canales (salida de conv_up3)
        # Hidden Size = 64 (puedes variarlo), Bidireccional = True
        self.lstm_hidden_size = 64
        self.lstm = nn.LSTM(input_size=64, 
                            hidden_size=self.lstm_hidden_size, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True)
        
        # --- SALIDA MODIFICADA ---
        # Como es bidireccional, la salida es hidden_size * 2
        self.outc = nn.Conv1d(self.lstm_hidden_size * 2, n_classes, kernel_size=1)

    def forward(self, x):
        # x original: (Batch, Time, Freq) -> Permutar a (Batch, Freq, Time)
        x = x.permute(0, 2, 1)
        
        # Encoder
        x1 = self.inc(x)       
        x2 = self.down1(x1)    
        x3 = self.down2(x2)    
        x4 = self.down3(x3)    
        
        # Decoder
        x = self.up1(x4)
        x = self._pad_and_cat(x, x3)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = self._pad_and_cat(x, x2)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = self._pad_and_cat(x, x1)
        x_cnn_out = self.conv_up3(x) 
        # Hasta aquí es la U-Net normal. x_cnn_out shape: (Batch, 64, Time)
        
        # --- ### NUEVO: PROCESAMIENTO LSTM ### ---
        
        # 1. Preparar dimensiones para LSTM: (Batch, Time, Features)
        x_lstm_in = x_cnn_out.permute(0, 2, 1)
        
        # 2. Optimización de memoria para GPU (opcional pero recomendado)
        self.lstm.flatten_parameters()
        
        # 3. Paso Forward LSTM
        # Salida: (Batch, Time, hidden_size*2)
        x_lstm_out, _ = self.lstm(x_lstm_in)
        
        # 4. Volver a dimensiones de Conv1d: (Batch, hidden_size*2, Time)
        x_final = x_lstm_out.permute(0, 2, 1)
        
        # --- CAPA FINAL ---
        logits = self.outc(x_final)
        
        # Devolver (Batch, Time, Clases)
        return logits.permute(0, 2, 1)

    def _pad_and_cat(self, x_up, x_skip):
        diff = x_skip.size(2) - x_up.size(2)
        if diff != 0:
            x_up = F.pad(x_up, [diff // 2, diff - diff // 2])
        return torch.cat([x_skip, x_up], dim=1)

# -------------------------------------------------------------------------
# 3. CONFIGURACIÓN DEL ENTRENAMIENTO
# -------------------------------------------------------------------------

# Parámetros
INPUT_DIM = X_train_tensor.shape[2] 
OUTPUT_DIM = 4 
N_EPOCHS = 100 

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Instanciar Modelo (AHORA USAMOS LA CLASE NUEVA)
model = AudioUNetWithBiLSTMEnd(n_channels=INPUT_DIM, n_classes=OUTPUT_DIM)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

# Archivos de salida
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
MODEL_PATH = f'best_unet_lstm_{timestamp}.pth' # Nombre actualizado
PLOT_NAME = f'unet_lstm_loss_{timestamp}.png' # Nombre actualizado

# Variables de control
best_val_loss = float('inf')
train_losses = []
val_losses = []

# -------------------------------------------------------------------------
# 4. BUCLE DE ENTRENAMIENTO
# -------------------------------------------------------------------------
print(f"--- 3. Iniciando Entrenamiento (100 Épocas) ---")

for epoch in range(N_EPOCHS):
    # --- FASE DE ENTRENAMIENTO ---
    model.train()
    total_train_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False)
    
    for X_batch, Y_batch in progress_bar:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        
        loss = criterion(outputs.reshape(-1, OUTPUT_DIM), Y_batch.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- FASE DE VALIDACIÓN ---
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs.reshape(-1, OUTPUT_DIM), Y_batch.view(-1))
            total_val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            targets = Y_batch.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{N_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        print(f"  ✅ ¡Nuevo Record! ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Guardando modelo...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)

print("\n--- Entrenamiento Finalizado ---")
print(f"Mejor modelo guardado en: {MODEL_PATH}")

# -------------------------------------------------------------------------
# 5. ANÁLISIS Y GRÁFICAS
# -------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'U-Net + BiLSTM Training Results (Best Val: {best_val_loss:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(PLOT_NAME)
print(f"Gráfica guardada: {PLOT_NAME}")

cm = confusion_matrix(all_targets, all_preds, labels=np.arange(OUTPUT_DIM))
print("\nMatriz de Confusión (Última época):")
print(cm)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
plt.title('Confusion Matrix - U-Net + BiLSTM')
plt.colorbar()
tick_marks = np.arange(OUTPUT_DIM)
plt.xticks(tick_marks, ['0', '1', '2', '3'])
plt.yticks(tick_marks, ['0', '1', '2', '3'])

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig(f"confusion_matrix_unet_lstm_{timestamp}.png")
plt.show()