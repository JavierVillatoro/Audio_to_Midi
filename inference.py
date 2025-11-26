import torch
import torch.nn as nn
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
# II. DEFINICIÓN DE ARQUITECTURAS (AMBOS MODELOS)
# =========================================================================

# --- 1. Modelo CRNN (Clásico) ---
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

# --- 2. Modelo Transformer (CORREGIDO PARA AUDIOS LARGOS) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model # Guardamos d_model para recalcular si hace falta

        # Generar buffer inicial (el que se guarda en el .pth)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        seq_len = x.size(1)
        
        # --- FIX: Lógica dinámica para audios largos ---
        if seq_len > self.pe.size(1):
            # Si el audio es más largo que el buffer guardado (ej. 4000 > 2000)
            # Generamos un PE nuevo al vuelo para este tamaño específico
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0) # [1, Seq_Len, d_model]
            
            x = x + pe
        else:
            # Caso normal: usamos el buffer rápido guardado
            x = x + self.pe[:, :seq_len, :]
            
        return self.dropout(x)

class AudioTransNet(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, output_dim=4, dropout=0.1):
        super(AudioTransNet, self).__init__()
        self.conv_embedding = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        # Inicializamos con max_len 2000 para ser compatible con los pesos guardados,
        # pero el forward ahora soporta cualquier longitud.
        self.pos_encoder = PositionalEncoding(d_model, max_len=2000, dropout=dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv_embedding(x) 
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x) # Aquí es donde ocurría el error antes
        x = self.transformer_encoder(x)
        out = self.fc(x)
        return out

# =========================================================================
# III. FUNCIONES AUXILIARES (CQT, MIDI, PIANO ROLL)
# =========================================================================

def load_audio_cqt(path):
    try:
        y, sr = librosa.load(path, sr=SR)
    except Exception as e:
        print(f"Error al cargar el audio: {e}")
        return None, None
        
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
    track.append(mido.MetaMessage('end_of_track', time=1))
    mid.save(output_path)
    print(f"MIDI guardado: {output_path}")

def visualize_midi_piano_roll(midi_path, sr=SR, hop_length=HOP):
    MIDI_IMAGE_PATH = midi_path.replace(".mid", "_pianoroll.png")
    try:
        midi_data = librosa_midi.PrettyMIDI(midi_path)
        fs = sr / hop_length
        piano_roll = midi_data.get_piano_roll(fs=fs) 
        active_notes = np.where(np.sum(piano_roll, axis=1) > 0)[0]
        if len(active_notes) == 0:
             print("Advertencia: El piano roll está vacío.")
             return
        min_note = max(0, active_notes.min() - 2)
        max_note = min(127, active_notes.max() + 2)
        piano_roll_visual = piano_roll[min_note:max_note, :]
        
        plt.figure(figsize=(12, 6))
        plt.imshow(piano_roll_visual, aspect='auto', origin='lower', cmap='plasma')
        y_ticks = np.arange(0, max_note - min_note)
        y_labels = [librosa.midi_to_note(n) for n in np.arange(min_note, max_note)]
        plt.yticks(y_ticks, y_labels)
        plt.xlabel('Tiempo (Frames)')
        plt.ylabel('Nota Musical')
        plt.title(f'Piano Roll: {os.path.basename(midi_path)}')
        plt.savefig(MIDI_IMAGE_PATH)
        print(f"Imagen guardada: {MIDI_IMAGE_PATH}")
        plt.close() 
    except Exception as e:
        print(f"No se pudo generar la imagen: {e}")

# =========================================================================
# IV. LÓGICA DE SELECCIÓN DE MODELO
# =========================================================================

def select_model_file():
    # Buscar todos los archivos .pth en el directorio actual
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
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print("Número inválido.")
        except ValueError:
            print("Por favor, introduce un número.")

def load_architecture(model_path, device):
    """
    Instancia la clase correcta basándose en el nombre del archivo.
    """
    filename = model_path.lower()
    
    if "transformer" in filename:
        print(">> Detectado arquitectura: TRANSFORMER (AudioTransNet)")
        # Importante: mantenemos los mismos hyperparámetros
        model = AudioTransNet(input_dim=INPUT_DIM, d_model=128, nhead=4, num_layers=3, output_dim=OUTPUT_DIM)
    elif "crnn" in filename:
        print(">> Detectado arquitectura: CRNN Clásico")
        model = CRNN(input_dim=INPUT_DIM, hidden_dim=128, output_dim=OUTPUT_DIM)
    elif "unet" in filename:
        print(">> Detectado arquitectura: U-Net")
        model = AudioUNet(n_channels=INPUT_DIM, n_classes=OUTPUT_DIM)
    else:
        # Fallback manual
        print("\n⚠️ No se puede determinar la arquitectura por el nombre del archivo.")
        print("[1] Transformer")
        print("[2] CRNN")
        opt = input("¿Qué arquitectura es?: ")
        if opt == "1":
            model = AudioTransNet(input_dim=INPUT_DIM, d_model=128, nhead=4, num_layers=3, output_dim=OUTPUT_DIM)
        else:
            model = CRNN(input_dim=INPUT_DIM, hidden_dim=128, output_dim=OUTPUT_DIM)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except RuntimeError as e:
        print(f"\n❌ Error cargando pesos: {e}")
        return None

# =========================================================================
# V. FUNCIÓN PRINCIPAL DE INFERENCIA
# =========================================================================

def run_inference(audio_path):
    
    # 1. Seleccionar Modelo
    model_path = select_model_file()
    if not model_path: return

    # 2. Configurar Salida
    base_name = os.path.basename(audio_path).split('.')[0]
    model_name = os.path.basename(model_path).split('.')[0]
    output_midi_path = f"{base_name}_pred_{model_name}.mid"

    # 3. Cargar Normalización
    try:
        norm_params = np.load(NORMALIZATION_PARAMS_PATH)
        global_mean, global_std = norm_params[0], norm_params[1]
    except FileNotFoundError:
        print(f"❌ Falta {NORMALIZATION_PARAMS_PATH}. Ejecuta el entrenamiento primero.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 4. Cargar Modelo
    model = load_architecture(model_path, device)
    if model is None: return

    # 5. Preprocesar Audio
    print(f"Procesando audio: {audio_path}")
    cqt_features, duration = load_audio_cqt(audio_path)
    if cqt_features is None: return
    
    print(f"Dimensiones de entrada (Frames): {cqt_features.shape[0]}")
    
    # Normalizar
    cqt_features_norm = (cqt_features - global_mean) / global_std
    X_test_tensor = torch.tensor(cqt_features_norm, dtype=torch.float32).unsqueeze(0).to(device)

    # 6. Inferencia
    print("Ejecutando red neuronal...")
    with torch.no_grad():
        outputs = model(X_test_tensor) 
    predictions = torch.argmax(outputs.squeeze(0), dim=-1).cpu().numpy()

    if np.all(predictions == 0):
        print("⚠️ AVISO: El modelo ha predicho solo silencio.")
    else:
        print("✅ Predicción finalizada con éxito.")

    # 7. Generar Resultados
    predictions_to_midi(predictions, output_midi_path)
    visualize_midi_piano_roll(output_midi_path)

if __name__ == '__main__':
    # RUTA DEL AUDIO A PROBAR
    INPUT_AUDIO = 'C:/Users/franc/Desktop/Tecnologia_del_habla/Audio_to_Midi/Prueba.wav' 
    
    if os.path.exists(INPUT_AUDIO):
        run_inference(INPUT_AUDIO)
    else:
        print(f"❌ No encuentro el archivo: {INPUT_AUDIO}")