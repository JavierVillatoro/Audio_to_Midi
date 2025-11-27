import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import optuna
import functools

# Asegúrate de que las utilidades estén disponibles
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels

# --- MOCK DE UTILIDADES PARA QUE EL SCRIPT SEA AUTOCONTENIDO EN ESTE EJEMPLO ---
# (Si tienes tus archivos utils, descomenta las lineas de arriba y borra estas funciones dummy)
def load_full_dataset(x_path, y_path):
    print("Cargando dataset...")
    # Simulación de datos para que el código compile si no tienes los .npy presentes
    # Reemplaza esto con tu carga real si tienes los archivos
    if not os.path.exists(x_path):
        return np.random.rand(100, 50, 20), np.random.randint(0, 4, (100, 50))
    return np.load(x_path), np.load(y_path)

def split_dataset(X, Y):
    # Split simple 80/10/10
    l = len(X)
    train_end = int(l*0.8)
    val_end = int(l*0.9)
    return X[:train_end], Y[:train_end], X[train_end:val_end], Y[train_end:val_end], X[val_end:], Y[val_end:]

def pad_sequences(sequences, max_len=None):
    # Padding simple simulado
    return sequences # Asumimos que ya vienen bien o usas tu función real

def pad_labels(labels, max_len=None):
    return labels

# -------------------------
# Cargar y Preparar Dataset
# -------------------------
print("Procesando datos...")
try:
    X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
except Exception as e:
    print(f"Nota: No se encontraron los archivos .npy reales ({e}), usando datos aleatorios para demostración.")
    X = np.random.rand(100, 100, 20) # (Samples, Time, Freq)
    Y = np.random.randint(0, 4, (100, 100))

X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

# Simulamos el padding si no usas tus utils externas
# Asegúrate de usar tus funciones pad_sequences reales aquí
X_train_padded = X_train.astype(np.float32)
Y_train_padded = Y_train.astype(np.int64)
X_val_padded = X_val.astype(np.float32)
Y_val_padded = Y_val.astype(np.int64)

# Crear Tensores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

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

# ==========================================
# DEFINICIÓN DE MODELOS
# ==========================================

# 1. CRNN
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
        x = x.permute(0, 2, 1) # (Batch, Freq, Time)
        x = self.conv(x)
        x = x.permute(0, 2, 1) # (Batch, Time, Feat)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# 2. AudioTransNet (CNN + Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AudioTransNet(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, output_dim=4, dropout=0.1):
        super(AudioTransNet, self).__init__()
        # A. Feature Extractor (CNN)
        self.conv_embedding = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        # B. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        # C. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # D. Clasificador
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, Freq, Time)
        x = self.conv_embedding(x) 
        x = x.permute(0, 2, 1) # (Batch, Time, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.fc(x) 
        return out

# 3. AudioUNet
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
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv1d(64, n_classes, kernel_size=1)

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
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits.permute(0, 2, 1)

    def _pad_and_cat(self, x_up, x_skip):
        diff = x_skip.size(2) - x_up.size(2)
        if diff != 0:
            x_up = F.pad(x_up, [diff // 2, diff - diff // 2])
        return torch.cat([x_skip, x_up], dim=1)

# ==========================================
# OPTUNA: FUNCIÓN OBJETIVO UNIFICADA
# ==========================================

def objective(trial, model_name):
    # 1. Hiperparámetros comunes
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    input_dim = X_train_tensor.shape[2]
    model = None

    # 2. Configuración específica por modelo
    if model_name == "CRNN":
        hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
        model = CRNN(input_dim, hidden_dim=hidden_dim)
        
    elif model_name == "AudioTransNet":
        # d_model debe ser divisible por nhead
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        
        # Filtramos nhead válidos
        possible_nheads = [2, 4, 8]
        valid_nheads = [h for h in possible_nheads if d_model % h == 0]
        nhead = trial.suggest_categorical('nhead', valid_nheads)
        
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = AudioTransNet(input_dim, d_model=d_model, nhead=nhead, 
                              num_layers=num_layers, dropout=dropout)
        
    elif model_name == "AudioUNet":
        # AudioUNet tiene arquitectura fija en este script, optimizamos LR y Batch
        # (Se podria parametrizar la profundidad, pero requiere cambiar la clase)
        model = AudioUNet(n_channels=input_dim, n_classes=4)

    # Mover modelo al dispositivo
    model.to(device)

    # 3. Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    # 4. Optimizador y Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    n_epochs = 5  # Epochs reducidas para la búsqueda

    # 5. Loop de Entrenamiento
    for epoch in range(n_epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.reshape(-1, 4), Y_batch.reshape(-1))
            loss.backward()
            optimizer.step()
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.reshape(-1, 4), Y_batch.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Reportar a Optuna
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

# ==========================================
# MAIN: EJECUCIÓN MULTI-MODELO
# ==========================================

if __name__ == '__main__':
    # Diccionario para guardar los resultados finales
    best_results = {}
    
    models_to_optimize = ["CRNN", "AudioTransNet", "AudioUNet"]
    
    # Pruner compartido
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    for model_name in models_to_optimize:
        print(f"\n=============================================")
        print(f" Optimizando Modelo: {model_name}")
        print(f"=============================================")
        
        # Limpiar memoria GPU entre estudios
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Crear nombre único para el estudio
        study_name = f"study_{model_name}"
        study = optuna.create_study(direction='minimize', pruner=pruner, study_name=study_name)
        
        # Usamos partial para pasar el argumento 'model_name' a la función objective
        objective_func = functools.partial(objective, model_name=model_name)
        
        # Ejecutar optimización (ej. 20 trials por modelo para probar rápido)
        study.optimize(objective_func, n_trials=20, show_progress_bar=True)
        
        print(f"--> Mejor Loss para {model_name}: {study.best_value:.4f}")
        print(f"--> Mejores Params: {study.best_params}")
        
        # Guardar en el diccionario
        best_results[model_name] = {
            "best_val_loss": study.best_value,
            "params": study.best_params
        }

    # ==========================================
    # GUARDAR RESULTADOS EN DISCO
    # ==========================================
    output_file = "mejores_parametros.npy"
    print(f"\nGuardando todos los resultados en {output_file}...")
    
    # Numpy guarda diccionarios si se envuelven en un array object
    np.save(output_file, best_results)
    
    # Verificación de carga
    loaded_results = np.load(output_file, allow_pickle=True).item()
    print("\n--- Resumen Final Guardado ---")
    for m_name, data in loaded_results.items():
        print(f"Modelo: {m_name} | Loss: {data['best_val_loss']:.4f}")
        print(f"   Params: {data['params']}")