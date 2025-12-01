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
from tqdm import tqdm # Barra de carga

# -------------------------------------------------------------------------
# 0. CONFIGURACIÓN DE RUTAS
# -------------------------------------------------------------------------
# Asegúrate de que esto apunta a donde tienes 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels

# -------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# -------------------------------------------------------------------------
print("--- 1. Cargando Dataset (Modo U-Net) ---")

# Cargar archivos .npy
X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

# Padding (Rellenar secuencias para que tengan la misma longitud)
print("Aplicando padding...")
X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)

# Usamos la longitud máxima de entrenamiento para validar
max_len = X_train_padded.shape[1]
X_val_padded = pad_sequences(X_val, max_len=max_len)
Y_val_padded = pad_labels(Y_val, max_len=max_len)

# Conversión a tipos de dato correctos para PyTorch
# X -> float32 (Features)
# Y -> int64 (Clases/Etiquetas)
X_train_padded = X_train_padded.astype(np.float32)
X_val_padded = X_val_padded.astype(np.float32)
Y_train_padded = Y_train_padded.astype(np.int64)
Y_val_padded = Y_val_padded.astype(np.int64)

# Crear Tensores
X_train_tensor = torch.tensor(X_train_padded)
Y_train_tensor = torch.tensor(Y_train_padded)
X_val_tensor = torch.tensor(X_val_padded)
Y_val_tensor = torch.tensor(Y_val_padded)

# Crear Dataset y DataLoader
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

# Batch size 8 suele ser bueno para U-Net en GPUs pequeñas/medias
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8)

# -------------------------------------------------------------------------
# 2. ARQUITECTURA U-NET 1D (La Joya de la Corona)
# -------------------------------------------------------------------------
print("--- 2. Inicializando Arquitectura U-Net ---")

class DoubleConv(nn.Module):
    """(Convolución => BatchNorm => ReLU) dos veces"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1), #De 3x3 por eso kernel_size 3 , mirar unet paper
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # Un poco de Dropout ayuda a que no memorice el dataset pequeño
            nn.Dropout(0.1), 
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AudioUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AudioUNet, self).__init__()
        
        # --- ENCODER (Bajada / Compresión) ---
        # Entra: (Batch, Frecuencias, Tiempo)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(256, 512))
        
        # --- DECODER (Subida / Reconstrucción) ---
        # Usamos ConvTranspose para aumentar la resolución temporal
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256) # 512 in porque concatena (256+256)
        
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        # --- SALIDA ---
        self.outc = nn.Conv1d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # x original: (Batch, Time, Freq)
        # Permutamos a: (Batch, Freq, Time) porque Conv1d quiere canales primero
        x = x.permute(0, 2, 1)
        
        # Encoder
        x1 = self.inc(x)       # Guardamos para Skip Connection 1
        x2 = self.down1(x1)    # Guardamos para Skip Connection 2
        x3 = self.down2(x2)    # Guardamos para Skip Connection 3
        x4 = self.down3(x3)    # Cuello de botella (Bottleneck)
        
        # Decoder con Skip Connections
        x = self.up1(x4)
        x = self._pad_and_cat(x, x3) # Concatenar con x3
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = self._pad_and_cat(x, x2) # Concatenar con x2
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = self._pad_and_cat(x, x1) # Concatenar con x1
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        
        # Volvemos a permutar para devolver (Batch, Time, Clases)
        return logits.permute(0, 2, 1)

    def _pad_and_cat(self, x_up, x_skip):
        # A veces, al hacer pooling, se pierde 1 pixel si es impar. 
        # Esta función ajusta el tamaño para poder concatenar sin errores.
        diff = x_skip.size(2) - x_up.size(2)
        if diff != 0:
            # Padding: (Izquierda, Derecha)
            x_up = F.pad(x_up, [diff // 2, diff - diff // 2])
        return torch.cat([x_skip, x_up], dim=1)

# -------------------------------------------------------------------------
# 3. CONFIGURACIÓN DEL ENTRENAMIENTO
# -------------------------------------------------------------------------

# Parámetros
INPUT_DIM = X_train_tensor.shape[2] # 252 bins
OUTPUT_DIM = 4 # Silencio, Do, Mi, Sol
N_EPOCHS = 100 # U-Net aprende rápido, 100 suele ser suficiente

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Instanciar Modelo
model = AudioUNet(n_channels=INPUT_DIM, n_classes=OUTPUT_DIM)
model.to(device)

# Optimizador y Loss
# CrossEntropyLoss funciona bien. Si tienes mucho silencio, podrías añadir pesos (class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler: Reduce el Learning Rate si la validación se estanca
# --- CORREGIDO: Eliminado argumento 'verbose' ---
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

# Archivos de salida
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
MODEL_PATH = f'best_unet_{timestamp}.pth'
PLOT_NAME = f'unet_loss_{timestamp}.png'

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
    
    # Barra de progreso
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False)
    
    for X_batch, Y_batch in progress_bar:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X_batch)
        
        # Calcular Loss (Flatten de dimensiones Batch y Time)
        loss = criterion(outputs.reshape(-1, OUTPUT_DIM), Y_batch.view(-1))
        
        # Backward
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
            
            # Recoger predicciones para matriz de confusión
            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            targets = Y_batch.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Actualizar Scheduler
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Imprimir info
    print(f"Epoch {epoch+1}/{N_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Guardar si mejora
    if avg_val_loss < best_val_loss:
        print(f"  ✅ ¡Nuevo Record! ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Guardando modelo...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)

print("\n--- Entrenamiento Finalizado ---")
print(f"Mejor modelo guardado en: {MODEL_PATH}")

# -------------------------------------------------------------------------
# 5. ANÁLISIS Y GRÁFICAS
# -------------------------------------------------------------------------

# Gráfica de Pérdida
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'U-Net Training Results (Best Val: {best_val_loss:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(PLOT_NAME)
print(f"Gráfica guardada: {PLOT_NAME}")

# Matriz de Confusión
cm = confusion_matrix(all_targets, all_preds, labels=np.arange(OUTPUT_DIM))
print("\nMatriz de Confusión (Última época):")
print(cm)

# Visualizar Matriz
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
plt.title('Confusion Matrix - U-Net')
plt.colorbar()
tick_marks = np.arange(OUTPUT_DIM)
plt.xticks(tick_marks, ['0', '1', '2', '3'])
plt.yticks(tick_marks, ['0', '1', '2', '3'])

# Poner números en los cuadros
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig(f"confusion_matrix_unet_{timestamp}.png")
plt.show()