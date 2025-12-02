import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt 
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

# -------------------------------------------------------------------------
# 0. CONFIGURACIÓN E IMPORTACIONES
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels
except ImportError:
    print("ADVERTENCIA: No se pudo importar utils.dataset_utils. Asegúrate de que la ruta sea correcta.")
    # Mock functions para que el linter no falle si no tienes los archivos locales
    def load_full_dataset(x, y): return np.zeros((100, 300, 252)), np.zeros((100, 666))
    def split_dataset(x, y): return x, y, x, y, x, y
    def pad_sequences(x, max_len=None): return x
    def pad_labels(y, max_len=None): return y

# Configuración global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 5   # Intentos de Optuna
N_EPOCHS = 15  # Épocas por trial para búsqueda rápida
OUTPUT_DIM = 4 # Clases: 0, 1, 2, 3

print(f"--- Usando dispositivo: {DEVICE} ---")

# -------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS (GLOBAL)
# -------------------------------------------------------------------------
print("--- 1. Cargando Dataset Globalmente ---")

if os.path.exists("dataset_X.npy") and os.path.exists("dataset_Y.npy"):
    X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    # Padding
    X_train_padded = pad_sequences(X_train)
    Y_train_padded = pad_labels(Y_train)
    
    max_len = X_train_padded.shape[1]
    X_val_padded = pad_sequences(X_val, max_len=max_len)
    Y_val_padded = pad_labels(Y_val, max_len=max_len)

    # Conversión a tensores
    X_train_tensor = torch.tensor(X_train_padded.astype(np.float32))
    Y_train_tensor = torch.tensor(Y_train_padded.astype(np.int64))
    X_val_tensor = torch.tensor(X_val_padded.astype(np.float32))
    Y_val_tensor = torch.tensor(Y_val_padded.astype(np.int64))
    
    INPUT_DIM = X_train_tensor.shape[2] 
else:
    print("Error: No se encuentran dataset_X.npy o dataset_Y.npy")
    sys.exit()

# --- PRO-TIP: SUAVIZADO DE PESOS ---
# Elevamos a la potencia 0.75 para "suavizar" la agresividad contra la clase mayoritaria (0).
# Esto reduce los Falsos Positivos en silencio sin ignorar las notas.
def calculate_class_weights_soft(y_tensor, power=0.75):
    y_flat = y_tensor.flatten().numpy()
    classes, counts = np.unique(y_flat, return_counts=True)
    
    # Frecuencia inversa suavizada
    weights = 1.0 / (counts ** power)
    
    # Normalizar para que el peso medio sea 1.0 (estabilidad numérica)
    weights = weights / weights.mean()
    
    print("\n--- Pesos Suavizados (Power 0.75) ---")
    for c, w, count in zip(classes, weights, counts):
        print(f"Clase {c}: {count} muestras -> Peso: {w:.4f}")
    
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

CLASS_WEIGHTS = calculate_class_weights_soft(Y_train_tensor, power=0.75)

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

# -------------------------------------------------------------------------
# 2. ARQUITECTURA PRO: SE-BLOCK + FOCAL LOSS
# -------------------------------------------------------------------------

# --- SQUEEZE-AND-EXCITATION BLOCK (SE-Block) ---
# Permite a la red "atender" a los canales más importantes y suprimir el ruido.
# Es ideal para datasets pequeños porque añade pocos parámetros pero mucha "inteligencia".
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
    """Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> ReLU -> SE-Block"""
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
        self.se = SEBlock(out_channels) # Añadimos atención aquí

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x) # Recalibración de canales
        return x

class AudioUNetWithBiLSTMEnd(nn.Module):
    def __init__(self, n_channels, n_classes, lstm_hidden_size=64, dropout_rate=0.1):
        super(AudioUNetWithBiLSTMEnd, self).__init__()
        
        # Encoder con SE-Blocks
        self.inc = DoubleConvSE(n_channels, 64, dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConvSE(64, 128, dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConvSE(128, 256, dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConvSE(256, 512, dropout_rate))
        
        # Decoder
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConvSE(512, 256, dropout_rate)
        
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConvSE(256, 128, dropout_rate)
        
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConvSE(128, 64, dropout_rate)
        
        # LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=64, 
                            hidden_size=self.lstm_hidden_size, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout_rate if dropout_rate < 1 else 0)
        
        self.outc = nn.Conv1d(self.lstm_hidden_size * 2, n_classes, kernel_size=1)

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
        x_final = x_lstm_out.permute(0, 2, 1)
        
        logits = self.outc(x_final)
        return logits.permute(0, 2, 1)

    def _pad_and_cat(self, x_up, x_skip):
        diff = x_skip.size(2) - x_up.size(2)
        if diff != 0:
            x_up = F.pad(x_up, [diff // 2, diff - diff // 2])
        return torch.cat([x_skip, x_up], dim=1)

# --- FOCAL LOSS (Reemplazo PRO de CrossEntropy) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss) # Probabilidad de acierto
        # (1 - pt)^gamma penaliza mucho los ejemplos difíciles (que el modelo duda)
        # y poco los fáciles (silencios claros).
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

# -------------------------------------------------------------------------
# 3. FUNCIÓN OBJETIVO DE OPTUNA
# -------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    acc = accuracy_score(y_true_flat, y_pred_flat)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    return acc, f1_macro

def objective(trial):
    # Sugerencias Optuna
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128])
    batch_size = trial.suggest_categorical("batch_size", [4, 8]) # Pequeño por memoria
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
    gamma = trial.suggest_float("focal_gamma", 1.0, 3.0) # Ajustar agresividad del Focal Loss

    t_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    v_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = AudioUNetWithBiLSTMEnd(INPUT_DIM, OUTPUT_DIM, lstm_hidden, dropout_rate).to(DEVICE)
    
    # Usamos Focal Loss en lugar de CE normal
    criterion = FocalLoss(alpha=CLASS_WEIGHTS, gamma=gamma)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    epochs_bar = tqdm(range(N_EPOCHS), desc=f"Trial {trial.number}", leave=False)
    
    for epoch in epochs_bar:
        model.train()
        for X_b, Y_b in t_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs.reshape(-1, OUTPUT_DIM), Y_b.view(-1))
            loss.backward()
            optimizer.step()
        
        # Validación rápida para Optuna
        model.eval()
        all_preds = []
        all_targets = []
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

        epochs_bar.set_postfix(f1_macro=f"{f1_m:.4f}")
        trial.report(f1_m, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return f1_m

# -------------------------------------------------------------------------
# 4. EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n--- Iniciando Búsqueda Profesional ({N_TRIALS} intentos) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    with tqdm(total=N_TRIALS, desc="Optuna Progress") as trials_bar:
        def tqdm_callback(study, trial):
            trials_bar.update(1)
            try:
                if study.best_value:
                    trials_bar.set_postfix(best_f1=f"{study.best_value:.4f}")
            except: pass
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[tqdm_callback])

    print("\n------------------------------------------------")
    print("MEJORES PARÁMETROS:")
    best_params = study.best_params
    print(best_params)
    print("------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 5. RE-ENTRENAMIENTO FINAL
    # -------------------------------------------------------------------------
    print("--- Entrenando Modelo Final (SE-UNet + BiLSTM + FocalLoss) ---")
    
    final_bs = best_params["batch_size"]
    final_loader_train = DataLoader(train_dataset, batch_size=final_bs, shuffle=True)
    final_loader_val = DataLoader(val_dataset, batch_size=final_bs)
    
    final_model = AudioUNetWithBiLSTMEnd(
        INPUT_DIM, OUTPUT_DIM, 
        lstm_hidden_size=best_params["lstm_hidden"], 
        dropout_rate=best_params["dropout"]
    ).to(DEVICE)
    
    # Usar Focal Loss con el gamma optimizado
    final_criterion = FocalLoss(alpha=CLASS_WEIGHTS, gamma=best_params["focal_gamma"])
    
    final_optimizer = optim.Adam(
        final_model.parameters(), 
        lr=best_params["lr"], 
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='max', factor=0.5, patience=5)

    best_f1_macro = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    FINAL_EPOCHS = 100 
    
    # Listas para almacenar pérdidas para la gráfica
    train_losses = []
    val_losses = []

    for epoch in range(FINAL_EPOCHS):
        final_model.train()
        train_loss = 0
        pbar = tqdm(final_loader_train, desc=f"Final Epoch {epoch+1}/{FINAL_EPOCHS}", leave=False)
        for X_b, Y_b in pbar:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            final_optimizer.zero_grad()
            out = final_model(X_b)
            loss = final_criterion(out.reshape(-1, OUTPUT_DIM), Y_b.view(-1))
            loss.backward()
            final_optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Guardar loss promedio de entrenamiento
        avg_train_loss = train_loss / len(final_loader_train)
        train_losses.append(avg_train_loss)
        
        # Validación
        final_model.eval()
        val_loss_accum = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_b, Y_b in final_loader_val:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                out = final_model(X_b)
                
                # Calcular loss de validación para la gráfica
                loss = final_criterion(out.reshape(-1, OUTPUT_DIM), Y_b.view(-1))
                val_loss_accum += loss.item()
                
                preds = torch.argmax(out, dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(Y_b.cpu().numpy())
        
        # Guardar loss promedio de validación
        avg_val_loss = val_loss_accum / len(final_loader_val)
        val_losses.append(avg_val_loss)
        
        y_p = np.concatenate(all_preds).flatten()
        y_t = np.concatenate(all_targets).flatten()
        acc, f1_m = compute_metrics(y_t, y_p)
        
        # Imprimir info completa
        print(f"Ep {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | F1-Macro: {f1_m:.4f}")
        scheduler.step(f1_m)
        
        if f1_m > best_f1_macro:
            best_f1_macro = f1_m
            print(f"  ✅ Nuevo Mejor F1-Macro! Guardando...")
            torch.save(final_model.state_dict(), f"best_model_pro_{timestamp}.pth")
            with open(f"report_{timestamp}.txt", "w") as f:
                f.write(classification_report(y_t, y_p, target_names=['Sil', 'C3', 'E3', 'G3']))
    
    # --- GRÁFICA DE LOSS ---
    print("\n--- Generando Gráfica de Error (Loss) ---")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Training vs Validation Loss (Best F1: {best_f1_macro:.3f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Focal)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_plot_{timestamp}.png")
    print(f"Gráfica guardada: loss_plot_{timestamp}.png")

    print("\n--- Generando Matriz de Confusión Final ---")
    cm = confusion_matrix(y_t, y_p)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix PRO (F1-Macro: {best_f1_macro:.3f})')
    plt.colorbar()
    tick_marks = np.arange(OUTPUT_DIM)
    plt.xticks(tick_marks, ['0 (Sil)', '1 (C3)', '2 (E3)', '3 (G3)'])
    plt.yticks(tick_marks, ['0 (Sil)', '1 (C3)', '2 (E3)', '3 (G3)'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(f"final_confusion_matrix_pro_{timestamp}.png")