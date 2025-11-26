import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import mido
import os
import sys
import glob
import math
import matplotlib.pyplot as plt
import pretty_midi as librosa_midi 

# =========================================================================
# I. CONFIGURACIÓN Y UTILIDADES
# =========================================================================

# --- Configuración de CQT ---
SR = 16000
HOP = 96
BINS_PER_OCTAVE = 36
N_BINS = 7 * BINS_PER_OCTAVE # 252

# --- Mapeo Inverso ---
INVERSE_MIDI_MAP = {
    0: -1,   # Silencio
    1: 60,   # C3
    2: 64,   # E3
    3: 67    # G3
}

# --- Parámetros Generales ---
NORMALIZATION_PARAMS_PATH = 'cqt_mean_std.npy' 
INPUT_DIM = N_BINS 
OUTPUT_DIM = len(INVERSE_MIDI_MAP) 

# =========================================================================
# II. DEFINICIÓN DE ARQUITECTURAS (TODAS)
# =========================================================================

# --- A. Modelo CRNN (Clásico) ---
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

# --- B. Modelo Transformer (AudioTransNet) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe
        else:
            x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class AudioTransNet(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, output_dim=4, dropout=0.1):
        super(AudioTransNet, self).__init__()
        self.conv_embedding = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=2000, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv_embedding(x) 
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.fc(x)
        return out

# --- C. Modelo U-Net (CORREGIDO) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Aquí estaba el problema. Debemos incluir el Dropout para que la estructura
        # coincida con el archivo guardado, aunque en inferencia no actúe.
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.1), # <--- ESTA LÍNEA ES CRUCIAL PARA CARGAR EL MODELO
            
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
        if diff != 0: x_up = F.pad(x_up, [diff // 2, diff - diff // 2])
        return torch.cat([x_skip, x_up], dim=1)

# =========================================================================
# III. FUNCIONES AUXILIARES
# =========================================================================

def load_audio_cqt(path):
    try: y, sr = librosa.load(path, sr=SR)
    except: return None, None
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max) 
    return cqt_db.T, len(y) / sr

def predictions_to_midi(predictions, output_path, tempo=120):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    ticks_per_beat = mid.ticks_per_beat 
    micro_s_per_beat = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=micro_s_per_beat, time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track.append(mido.Message('program_change', program=0, time=0, channel=0)) 
    
    frame_time_sec = HOP / SR 
    frame_time_ticks = int(round(mido.second2tick(frame_time_sec, ticks_per_beat, micro_s_per_beat)))
    current_note = -1          
    last_event_frame = 0        
    
    for i, label in enumerate(predictions):
        predicted_note = INVERSE_MIDI_MAP.get(label, -1)
        if predicted_note != current_note:
            delta_frames = i - last_event_frame
            delta_ticks = delta_frames * frame_time_ticks
            if current_note != -1:
                track.append(mido.Message('note_off', note=current_note, velocity=0, time=delta_ticks))
                last_event_frame = i 
                delta_ticks = 0      
            if predicted_note != -1:
                track.append(mido.Message('note_on', note=predicted_note, velocity=100, time=delta_ticks))
                last_event_frame = i 
            current_note = predicted_note
            
    delta_frames = len(predictions) - last_event_frame
    final_delta_ticks = delta_frames * frame_time_ticks
    if current_note != -1:
        track.append(mido.Message('note_off', note=current_note, velocity=0, time=final_delta_ticks))

    mid.save(output_path)
    print(f"MIDI guardado: {output_path}")

def visualize_midi_piano_roll(midi_path, sr=SR, hop_length=HOP):
    MIDI_IMAGE_PATH = midi_path.replace(".mid", "_pianoroll.png")
    try:
        midi_data = librosa_midi.PrettyMIDI(midi_path)
        fs = sr / hop_length
        piano_roll = midi_data.get_piano_roll(fs=fs) 
        active_notes = np.where(np.sum(piano_roll, axis=1) > 0)[0]
        if len(active_notes) == 0: return
        min_note = max(0, active_notes.min() - 2)
        max_note = min(127, active_notes.max() + 2)
        piano_roll_visual = piano_roll[min_note:max_note, :]
        plt.figure(figsize=(12, 6))
        plt.imshow(piano_roll_visual, aspect='auto', origin='lower', cmap='plasma')
        y_ticks = np.arange(0, max_note - min_note)
        y_labels = [librosa.midi_to_note(n) for n in np.arange(min_note, max_note)]
        plt.yticks(y_ticks, y_labels)
        plt.title(f'Piano Roll: {os.path.basename(midi_path)}')
        plt.savefig(MIDI_IMAGE_PATH)
        plt.close() 
    except: pass

# =========================================================================
# IV. LÓGICA DE SELECCIÓN DE MODELO
# =========================================================================

def select_model_file():
    models = glob.glob("*.pth")
    if not models:
        print("❌ No se encontraron archivos .pth en el directorio actual.")
        return None
    print("\n--- MODELOS DISPONIBLES ---")
    for i, m in enumerate(models):
        print(f"[{i+1}] {m}")
    while True:
        try:
            choice = int(input("\nElige el número del modelo a usar: ")) - 1
            if 0 <= choice < len(models): return models[choice]
            else: print("Número inválido.")
        except ValueError: print("Por favor, introduce un número.")

def load_architecture(model_path, device):
    filename = model_path.lower()
    
    if "transformer" in filename:
        print(">> Detectado arquitectura: TRANSFORMER")
        model = AudioTransNet(input_dim=INPUT_DIM, d_model=128, nhead=4, num_layers=3, output_dim=OUTPUT_DIM)
    
    elif "unet" in filename:
        print(">> Detectado arquitectura: U-NET")
        model = AudioUNet(n_channels=INPUT_DIM, n_classes=OUTPUT_DIM)
        
    elif "crnn" in filename:
        print(">> Detectado arquitectura: CRNN Clásico")
        model = CRNN(input_dim=INPUT_DIM, hidden_dim=128, output_dim=OUTPUT_DIM)
        
    else:
        print("\n⚠️ No se reconoce el nombre.")
        print("[1] Transformer / [2] U-Net / [3] CRNN")
        opt = input("¿Cuál es?: ")
        if opt == "1": model = AudioTransNet(input_dim=INPUT_DIM, d_model=128, nhead=4, num_layers=3, output_dim=OUTPUT_DIM)
        elif opt == "2": model = AudioUNet(n_channels=INPUT_DIM, n_classes=OUTPUT_DIM)
        else: model = CRNN(input_dim=INPUT_DIM, hidden_dim=128, output_dim=OUTPUT_DIM)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except RuntimeError as e:
        print(f"\n❌ Error cargando pesos. Detalle:\n{e}")
        return None

# =========================================================================
# V. EJECUCIÓN
# =========================================================================

def run_inference(audio_path):
    model_path = select_model_file()
    if not model_path: return

    base_name = os.path.basename(audio_path).split('.')[0]
    model_name = os.path.basename(model_path).split('.')[0]
    output_midi_path = f"{base_name}_pred_{model_name}.mid"

    try:
        norm_params = np.load(NORMALIZATION_PARAMS_PATH)
        global_mean, global_std = norm_params[0], norm_params[1]
    except:
        print(f"❌ Falta {NORMALIZATION_PARAMS_PATH}.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = load_architecture(model_path, device)
    if model is None: return

    print(f"Procesando audio: {audio_path}")
    cqt_features, duration = load_audio_cqt(audio_path)
    if cqt_features is None: return
    
    # Normalizar
    cqt_features_norm = (cqt_features - global_mean) / global_std
    X_test_tensor = torch.tensor(cqt_features_norm, dtype=torch.float32).unsqueeze(0).to(device)

    print("Ejecutando red neuronal...")
    with torch.no_grad():
        outputs = model(X_test_tensor) 
    
    predictions = torch.argmax(outputs.squeeze(0), dim=-1).cpu().numpy()

    if np.all(predictions == 0): print("⚠️ AVISO: Solo silencio detectado.")
    else: print("✅ Predicción con notas generada.")

    predictions_to_midi(predictions, output_midi_path)
    visualize_midi_piano_roll(output_midi_path)

if __name__ == '__main__':
    # RUTA DEL AUDIO A PROBAR
    INPUT_AUDIO = 'C:/Users/franc/Desktop/Tecnologia_del_habla/Audio_to_Midi/Prueba.wav' 
    
    if os.path.exists(INPUT_AUDIO):
        run_inference(INPUT_AUDIO)
    else:
        print(f"❌ No encuentro el archivo: {INPUT_AUDIO}")