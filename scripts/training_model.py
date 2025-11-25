import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 


# Cargar dataset

X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)

X_val_padded = pad_sequences(X_val, max_len=X_train_padded.shape[1])
Y_val_padded = pad_labels(Y_val, max_len=Y_train_padded.shape[1])

# Para X (las características): deben ser float (como lo espera el modelo CRNN)
X_train_padded = X_train_padded.astype(np.float32)
X_val_padded = X_val_padded.astype(np.float32)

# Para Y (las etiquetas): deben ser int (para CrossEntropyLoss)
Y_train_padded = Y_train_padded.astype(np.int64) # int64 es el valor por defecto que espera torch.long
Y_val_padded = Y_val_padded.astype(np.int64)

# --- Paso 4: Crear Tensores de PyTorch (Ahora esto funcionará) ---
# Ahora la conversión a tensor con el tipo PyTorch deseado es exitosa
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_padded, dtype=torch.long) # torch.long es el alias para torch.int64

X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_padded, dtype=torch.long)

# -------------------------
# Dataset y DataLoader
# -------------------------
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
# -------------------------
# Modelo CRNN simple (Sin cambios)
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

# -------------------------
# Inicializar modelo y parámetros (con mejores hiperparámetros)
# -------------------------
input_dim = X_train_tensor.shape[2]
# Aplicar el mejor hidden_dim (aunque es el default)
BEST_HIDDEN_DIM = 128
model = CRNN(input_dim, hidden_dim=BEST_HIDDEN_DIM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
# Aplicar el mejor learning rate de Optuna
BEST_LR = 0.0009341062209856291
optimizer = optim.Adam(model.parameters(), lr=BEST_LR)

# Parámetros de guardado y métricas
n_epochs = 50 
BEST_VAL_LOSS = float('inf')
MODEL_PATH = 'best_crnn_model.pth' # Archivo donde se guardarán los pesos
train_losses = []
val_losses = []

# -------------------------
# Loop de entrenamiento y guardado
# -------------------------
print(f"Iniciando entrenamiento en {device} por {n_epochs} épocas...")

for epoch in range(n_epochs):
    # --- Fase de Entrenamiento ---
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.view(-1, 4), Y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # --- Fase de Validación ---
    model.eval()
    val_loss = 0
    # Listas para Matriz de Confusión
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, 4), Y_batch.view(-1))
            val_loss += loss.item()
            
            # Recolección de predicciones y etiquetas
            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            targets = Y_batch.view(-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ** Lógica de Guardado del Mejor Modelo **
    if avg_val_loss < BEST_VAL_LOSS:
        print(f"  ✅ Pérdida de Validación mejorada ({BEST_VAL_LOSS:.4f} -> {avg_val_loss:.4f}). Guardando modelo en {MODEL_PATH}...")
        BEST_VAL_LOSS = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)

print("Entrenamiento finalizado. El mejor modelo ha sido guardado.")

# ----------------------------------------------------
# 📊 Análisis de Resultados
# ----------------------------------------------------

## 1. Gráfica Error/Época

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Pérdida de Entrenamiento')
plt.plot(val_losses, label='Pérdida de Validación')
plt.title('Pérdida (Loss) vs. Época')
plt.xlabel('Época')
plt.ylabel('Pérdida (Cross Entropy)')
plt.legend()
plt.grid(True)
plt.savefig('grafica_perdida_crnn.png')
plt.show()

## 2. Matriz de Confusión

# Calcular la matriz de confusión con los datos de la última época de validación
cm = confusion_matrix(all_targets, all_preds, labels=np.arange(4))

print("\n--- Matriz de Confusión (Última Época, Validación) ---")
print("Clases (0=background, 1, 2, 3)")
print(cm)

# Opcional: Visualización de la Matriz de Confusión (Requiere seaborn, pero lo mostramos con plt)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(4)
plt.xticks(tick_marks, ['0', '1', '2', '3'])
plt.yticks(tick_marks, ['0', '1', '2', '3'])

# Para poner los números en los cuadrados (si lo deseas)
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.savefig('matriz_confusion_crnn.png')
plt.show()