import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt 
from tqdm import tqdm 

# -------------------------------------------------------------------------
# 0. CONFIGURACIÓN DE RUTAS
# -------------------------------------------------------------------------
# Apuntar a la carpeta donde está 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels

# Configuración de Matplotlib
plt.style.use('ggplot')

# -------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# -------------------------------------------------------------------------
print("--- 1. Cargando Dataset (Modo U-Net) ---")

# Cargar archivos .npy
X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")

# División: 70% Train, 15% Val, 15% Test (según tu utils.py)
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

print(f"Dimensiones -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Padding Train
print("Aplicando padding...")
X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)

# Obtenemos la longitud máxima definida por el Train para aplicarla a todos
max_len = X_train_padded.shape[1]
print(f"Longitud de secuencia fijada en: {max_len}")

# Padding Val (usando max_len de train)
X_val_padded = pad_sequences(X_val, max_len=max_len)
Y_val_padded = pad_labels(Y_val, max_len=max_len)

# Convertir tipos
X_train_padded = X_train_padded.astype(np.float32)
Y_train_padded = Y_train_padded.astype(np.int64)
X_val_padded   = X_val_padded.astype(np.float32)
Y_val_padded   = Y_val_padded.astype(np.int64)

# Tensores
X_train_tensor = torch.tensor(X_train_padded)
Y_train_tensor = torch.tensor(Y_train_padded)
X_val_tensor   = torch.tensor(X_val_padded)
Y_val_tensor   = torch.tensor(Y_val_padded)

# Dataset Class
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
# 2. ARQUITECTURA U-NET 1D
# -------------------------------------------------------------------------
print("--- 2. Inicializando Arquitectura U-Net ---")

class DoubleConv(nn.Module):
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

class AudioUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AudioUNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(256, 512))
        
        # Decoder
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        # Salida
        self.outc = nn.Conv1d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, Freq, Time)
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
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits.permute(0, 2, 1) # (Batch, Time, Classes)

    def _pad_and_cat(self, x_up, x_skip):
        diff = x_skip.size(2) - x_up.size(2)
        if diff != 0:
            x_up = F.pad(x_up, [diff // 2, diff - diff // 2])
        return torch.cat([x_skip, x_up], dim=1)

# -------------------------------------------------------------------------
# 3. CONFIGURACIÓN DEL ENTRENAMIENTO
# -------------------------------------------------------------------------
INPUT_DIM = X_train_tensor.shape[2] 
OUTPUT_DIM = 4 
N_EPOCHS = 100 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = AudioUNet(n_channels=INPUT_DIM, n_classes=OUTPUT_DIM)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
MODEL_PATH = f'best_unet_{timestamp}.pth'
PLOT_NAME = f'unet_loss_{timestamp}.png'

best_val_loss = float('inf')
train_losses = []
val_losses = []

# -------------------------------------------------------------------------
# 4. BUCLE DE ENTRENAMIENTO
# -------------------------------------------------------------------------
print(f"--- 3. Iniciando Entrenamiento ({N_EPOCHS} Épocas) ---")

for epoch in range(N_EPOCHS):
    # --- TRAIN ---
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
    
    # --- VAL ---
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.reshape(-1, OUTPUT_DIM), Y_batch.view(-1))
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{N_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        print(f"  ✅ Record! ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Guardando...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)

print("\n--- Entrenamiento Finalizado ---")

# Gráfica de Entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'U-Net Training Results (Best Val: {best_val_loss:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(PLOT_NAME)
plt.close() # Cerrar para liberar memoria

# -------------------------------------------------------------------------
# 6. EVALUACIÓN FINAL CON EL TEST SET (EXAMEN FINAL)
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("--- 6. INICIANDO EVALUACIÓN EN TEST SET ---")
print("="*50)

# 1. Preprocesar Test (usando max_len de Train)
print("Preparando datos de Test...")
X_test_padded = pad_sequences(X_test, max_len=max_len)
Y_test_padded = pad_labels(Y_test, max_len=max_len)

X_test_padded = X_test_padded.astype(np.float32)
Y_test_padded = Y_test_padded.astype(np.int64)

X_test_tensor = torch.tensor(X_test_padded)
Y_test_tensor = torch.tensor(Y_test_padded)

test_dataset = AudioMIDIDataset(X_test_tensor, Y_test_tensor)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 2. Cargar MEJOR modelo
print(f"Cargando mejor modelo: {MODEL_PATH}")
best_model = AudioUNet(n_channels=INPUT_DIM, n_classes=OUTPUT_DIM)
best_model.load_state_dict(torch.load(MODEL_PATH))
best_model.to(device)
best_model.eval()

# 3. Inferencia
total_test_loss = 0
all_test_preds = []
all_test_targets = []

with torch.no_grad():
    for X_batch, Y_batch in tqdm(test_loader, desc="Evaluando Test"):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        outputs = best_model(X_batch)
        loss = criterion(outputs.reshape(-1, OUTPUT_DIM), Y_batch.view(-1))
        total_test_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=2)
        all_test_preds.extend(preds.view(-1).cpu().numpy())
        all_test_targets.extend(Y_batch.view(-1).cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
test_acc = accuracy_score(all_test_targets, all_test_preds)

print(f"\nRESULTADOS FINALES:")
print(f" -> Accuracy Global en Test: {test_acc*100:.2f}%")
print(f" -> Final Test Loss: {avg_test_loss:.4f}")
print("\nReporte Detallado:")
print(classification_report(all_test_targets, all_test_preds, digits=4))

# -------------------------------------------------------------------------
# 7. GRÁFICAS DEL TEST
# -------------------------------------------------------------------------

# A) Matriz de Confusión Test
cm_test = confusion_matrix(all_test_targets, all_test_preds, labels=np.arange(OUTPUT_DIM))

plt.figure(figsize=(8, 6))
plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'CONFUSION MATRIX - TEST SET\nAcc: {test_acc*100:.2f}%')
plt.colorbar()
tick_marks = np.arange(OUTPUT_DIM)
class_names = ['0', '1', '2', '3'] 
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

thresh = cm_test.max() / 2.
for i, j in np.ndindex(cm_test.shape):
    plt.text(j, i, format(cm_test[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm_test[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción Modelo')
plt.tight_layout()
plt.savefig(f"confusion_matrix_TEST_{timestamp}.png")
plt.close()

# B) Comparativa de Loss
losses = [train_losses[-1], best_val_loss, avg_test_loss]
labels = ['Final Train', 'Best Val', 'Test Set']
colors = ['#ff9999', '#66b3ff', '#99ff99']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, losses, color=colors, edgecolor='black')
plt.title('Comparativa Final de Pérdida (Loss)')
plt.ylabel('Cross Entropy Loss')
plt.ylim(0, max(losses) * 1.25)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"loss_comparison_{timestamp}.png")
plt.close()

print(f"\n¡Proceso Completo! Gráficas generadas con timestamp: {timestamp}")