import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna 
import datetime

# Actualización de sintaxis para evitar warnings de deprecación
from torch.amp import GradScaler, autocast 

# -------------------------------------------------------------------------
# 0. CONFIGURACIÓN E IMPORTACIONES
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset_utils import load_full_dataset

# Configuración Fija del Dataset
FIXED_LENGTH = 666       
INPUT_FREQ = 252         
NUM_CLASSES = 4          

# Semillas
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------------------------------
# 1. CARGA DE DATOS (Una sola vez para todo el proceso)
# -------------------------------------------------------------------------
print(f"--- 1. Cargando Dataset en Memoria ---")
X_raw, Y_raw = load_full_dataset("dataset_X.npy", "dataset_Y.npy")

print("Convirtiendo tipos de datos (Corrección Object -> Number)...")
# Conversión explícita para evitar errores de numpy
X_stack = np.array([np.array(x, dtype=np.float32) for x in X_raw], dtype=np.float32)
Y_stack = np.array([np.array(y, dtype=np.int64) for y in Y_raw], dtype=np.int64)

# División 70/15/15
indices = np.arange(len(X_stack))
np.random.shuffle(indices)
n_train = int(0.7 * len(X_stack))
n_val = int(0.15 * len(X_stack))

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]

X_train_np, Y_train_np = X_stack[train_idx], Y_stack[train_idx]
X_val_np, Y_val_np     = X_stack[val_idx], Y_stack[val_idx]
X_test_np, Y_test_np   = X_stack[test_idx], Y_stack[test_idx]

class FixedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# -------------------------------------------------------------------------
# 2. ARQUITECTURA HÍBRIDA (Modificada para Logits)
# -------------------------------------------------------------------------

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class LightUNetExtractor(nn.Module):
    def __init__(self, in_channels=1, feature_dim=32):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(32, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = UNetBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = UNetBlock(32, feature_dim)
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        if u2.shape[2:] != e2.shape[2:]: u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape[2:] != e1.shape[2:]: u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return d1 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1), :].unsqueeze(0)

class HybridTranscriber(nn.Module):
    def __init__(self, input_freq_bins, num_classes=4, cnn_features=32, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Feature Extractor
        self.unet = LightUNetExtractor(in_channels=1, feature_dim=cnn_features)
        self.linear_emb = nn.Linear(cnn_features, d_model)
        
        # 2. Freq Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True, dropout=dropout)
        self.freq_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.freq_pos = PositionalEncoding(d_model, max_len=input_freq_bins)
        
        # 3. Converter
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True, dropout=dropout)
        self.freq_converter = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.class_query = nn.Parameter(torch.randn(1, num_classes, d_model))
        
        # Cabezales 1
        self.head1_frame = nn.Linear(d_model, 1)
        self.head1_onset = nn.Linear(d_model, 1)
        self.head1_offset = nn.Linear(d_model, 1)
        
        # 4. Time Transformer
        time_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True, dropout=dropout)
        self.time_transformer = nn.TransformerEncoder(time_layer, num_layers=1)
        self.time_pos = PositionalEncoding(d_model, max_len=FIXED_LENGTH + 50)
        
        # Cabezales 2
        self.head2_frame = nn.Linear(d_model, 1)
        self.head2_onset = nn.Linear(d_model, 1)
        self.head2_offset = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, F = x.shape
        x_img = x.unsqueeze(1) 
        x_feat = self.unet(x_img) 
        
        x_feat = x_feat.permute(0, 2, 3, 1).reshape(B*T, F, -1)
        x_emb = self.linear_emb(x_feat) 
        x_emb = self.freq_pos(x_emb)
        freq_enc = self.freq_transformer(x_emb) 
        
        query = self.class_query.repeat(B*T, 1, 1) 
        class_repr = self.freq_converter(query, freq_enc) 
        
        # Outputs 1 (LOGITS, sin sigmoid)
        h1_frame = self.head1_frame(class_repr)
        h1_onset = self.head1_onset(class_repr)
        h1_offset = self.head1_offset(class_repr)
        
        C = class_repr.shape[1]
        time_in = class_repr.view(B, T, C, -1).permute(0, 2, 1, 3).reshape(B*C, T, -1)
        time_in = self.time_pos(time_in)
        time_enc = self.time_transformer(time_in)
        
        # Outputs 2 (LOGITS, sin sigmoid)
        h2_frame = self.head2_frame(time_enc)
        h2_onset = self.head2_onset(time_enc)
        h2_offset = self.head2_offset(time_enc)
        
        # Devolvemos Logits crudos para BCEWithLogitsLoss
        out1 = {
            'frame': h1_frame.view(B, T, C),
            'onset': h1_onset.view(B, T, C),
            'offset': h1_offset.view(B, T, C)
        }
        out2 = {
            'frame': h2_frame.view(B, T, C),
            'onset': h2_onset.view(B, T, C),
            'offset': h2_offset.view(B, T, C)
        }
        return out1, out2

# -------------------------------------------------------------------------
# 3. UTILS AUXILIARES
# -------------------------------------------------------------------------
def get_binary_targets(Y_batch, num_classes=4):
    target_frame = F.one_hot(Y_batch, num_classes=num_classes).float()
    padded = F.pad(target_frame, (0,0,1,0,0,0)) 
    diff = padded[:, 1:, :] - padded[:, :-1, :]
    target_onset = (diff > 0).float()
    target_offset = (diff < 0).float()
    return target_frame, target_onset, target_offset

def extract_notes_simple(onset_probs, frame_probs, threshold=0.5):
    notes = []
    num_classes = onset_probs.shape[1]
    for c in range(1, num_classes): 
        onsets = np.where(onset_probs[:, c] > threshold)[0]
        for start in onsets:
            future = frame_probs[start:, c]
            ends = np.where(future < threshold)[0]
            if len(ends) > 0: end = start + ends[0]
            else: end = len(frame_probs)
            if end > start + 1: notes.append((c, start, end))
    return notes

def compute_note_metrics(pred_notes, true_notes, tolerance=3):
    tp = 0; fp = 0; matched = set()
    gt = list(true_notes)
    for p in pred_notes:
        found = False
        for i, g in enumerate(gt):
            if i in matched: continue
            if p[0] == g[0] and abs(p[1] - g[1]) <= tolerance:
                found = True; matched.add(i); tp += 1; break
        if not found: fp += 1
    fn = len(gt) - len(matched)
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    return p, r, f1

# -------------------------------------------------------------------------
# 4. OPTUNA OBJECTIVE
# -------------------------------------------------------------------------
def objective(trial):
    # --- CONFIGURACIÓN DE SEGURIDAD PARA GTX 1060 (6GB) ---
    # Solo dejamos que Optuna optimice el Learning Rate
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # FIJAMOS los parámetros de memoria para que no explote
    batch_size = 4          # Batch 8 explotó, nos quedamos en 4
    d_model = 64            # 32 o 64 es seguro
    n_heads = 2             # 2 heads gastan menos memoria que 4
    cnn_feat = 16           # 16 filtros en U-Net es ligero
    
    # 2. Setup Modelo y Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # IMPORTANTE: Liberar memoria antes de empezar cada trial
    torch.cuda.empty_cache()
    
    train_loader = DataLoader(FixedDataset(X_train_np, Y_train_np), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(FixedDataset(X_val_np, Y_val_np), batch_size=batch_size, shuffle=False)
    
    model = HybridTranscriber(
        input_freq_bins=INPUT_FREQ, 
        num_classes=NUM_CLASSES, 
        cnn_features=cnn_feat, 
        d_model=d_model,
        n_heads=n_heads
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() 
    scaler = GradScaler('cuda')

    print(f"\n[Trial {trial.number}] Params: {trial.params}")
    
    try:
        for epoch in range(15):
            model.train()
            loop = tqdm(train_loader, desc=f"Trial {trial.number} Ep {epoch+1}/15", leave=False)
            
            for X_b, Y_b in loop:
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                t_frame, t_onset, t_offset = get_binary_targets(Y_b, NUM_CLASSES)
                
                with autocast('cuda'):
                    out1, out2 = model(X_b)
                    l1 = criterion(out1['frame'], t_frame) + criterion(out1['onset'], t_onset)
                    l2 = criterion(out2['frame'], t_frame) + criterion(out2['onset'], t_onset)
                    loss = l1 + l2

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loop.set_postfix(loss=loss.item())

            # Validación
            model.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for X_b, Y_b in val_loader:
                    X_b = X_b.to(device)
                    _, out2 = model(X_b)
                    
                    p_fr = torch.sigmoid(out2['frame']).cpu().numpy()
                    p_on = torch.sigmoid(out2['onset']).cpu().numpy()
                    t_fr = F.one_hot(Y_b, NUM_CLASSES).float().numpy()
                    padded = np.pad(t_fr, ((0,0),(1,0),(0,0)))
                    t_on = (padded[:, 1:, :] - padded[:, :-1, :] > 0)
                    
                    for i in range(X_b.shape[0]):
                        all_pred.extend(extract_notes_simple(p_on[i], p_fr[i]))
                        all_true.extend(extract_notes_simple(t_on[i], t_fr[i]))
            
            _, _, f1 = compute_note_metrics(all_pred, all_true)
            trial.report(f1, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Trial {trial.number} explotó por memoria. Saltando...")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned() # Cortamos el trial limpiamente
        else:
            raise e

    return f1

# -------------------------------------------------------------------------
# 5. EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------
if __name__ == "__main__":
    
    print("--- 2. Buscando Hiperparámetros con Optuna ---")
    study = optuna.create_study(direction="maximize")
    # Ejecutamos 20 trials (intentos)
    study.optimize(objective, n_trials=5) 

    print("\n" + "="*50)
    print("MEJORES HIPERPARÁMETROS ENCONTRADOS:")
    best_params = study.best_params
    print(best_params)
    print("="*50 + "\n")

    # -------------------------------------------------------------------------
    # 6. ENTRENAMIENTO FINAL (Con los mejores parámetros)
    # -------------------------------------------------------------------------
    print("--- 3. Iniciando Entrenamiento Final (50 Épocas) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear Loaders finales
    final_batch = best_params['batch_size']
    train_loader = DataLoader(FixedDataset(X_train_np, Y_train_np), batch_size=final_batch, shuffle=True)
    val_loader   = DataLoader(FixedDataset(X_val_np, Y_val_np), batch_size=final_batch, shuffle=False)
    
    # Instanciar Modelo Ganador
    model = HybridTranscriber(
        input_freq_bins=INPUT_FREQ, 
        num_classes=NUM_CLASSES, 
        cnn_features=best_params['cnn_features'], 
        d_model=best_params['d_model'],
        n_heads=best_params['n_heads']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.BCEWithLogitsLoss() # Estable
    scaler = GradScaler('cuda')
    
    history = {'loss': [], 'val_f1': []}
    BEST_FINAL_F1 = 0.0
    
    # Bucle final (50 Epochs)
    for epoch in range(50):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Final Ep {epoch+1}", leave=False)
        
        for X_b, Y_b in pbar:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            t_frame, t_onset, t_offset = get_binary_targets(Y_b, NUM_CLASSES)
            
            with autocast('cuda'):
                out1, out2 = model(X_b)
                l1 = criterion(out1['frame'], t_frame) + criterion(out1['onset'], t_onset) + criterion(out1['offset'], t_offset)
                l2 = criterion(out2['frame'], t_frame) + criterion(out2['onset'], t_onset) + criterion(out2['offset'], t_offset)
                loss = l1 + l2

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Validación Final
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device)
                _, out2 = model(X_b)
                p_fr = torch.sigmoid(out2['frame']).cpu().numpy()
                p_on = torch.sigmoid(out2['onset']).cpu().numpy()
                
                t_fr = F.one_hot(Y_b, NUM_CLASSES).float().numpy()
                padded = np.pad(t_fr, ((0,0),(1,0),(0,0)))
                t_on = (padded[:, 1:, :] - padded[:, :-1, :] > 0)
                
                for i in range(X_b.shape[0]):
                    all_pred.extend(extract_notes_simple(p_on[i], p_fr[i]))
                    all_true.extend(extract_notes_simple(t_on[i], t_fr[i]))
        
        prec, rec, f1 = compute_note_metrics(all_pred, all_true)
        history['val_f1'].append(f1)
        print(f"Ep {epoch+1} | Loss: {avg_loss:.4f} | F1: {f1:.4f}")
        
        if f1 > BEST_FINAL_F1:
            BEST_FINAL_F1 = f1
            torch.save(model.state_dict(), "best_hybrid_optuna_optimized.pth")

    # Gráfica Final
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title(f'Optimized Training (Best F1: {BEST_FINAL_F1:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig("optuna_results.png")
    print("Proceso completo. Gráfica guardada.")