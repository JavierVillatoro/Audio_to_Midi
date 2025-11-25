import torch
import torch.nn as nn
import numpy as np
import librosa
import mido
import os
import sys
import matplotlib.pyplot as plt
import pretty_midi as librosa_midi 

# =========================================================================
# I. CONFIGURACIÓN Y UTILIDADES
# =========================================================================

# --- Configuración de CQT (Debe coincidir con el entrenamiento) ---
SR = 16000
HOP = 96
BINS_PER_OCTAVE = 36
N_BINS = 7 * BINS_PER_OCTAVE # 252

# --- Mapeo Inverso de Clases a Notas MIDI ---
INVERSE_MIDI_MAP = {
    0: -1,   # Silencio
    1: 60,   # C3
    2: 64,   # E3
    3: 67    # G3
}

# --- Parámetros del Modelo CRNN ---
MODEL_PATH = 'best_crnn_model.pth'
NORMALIZATION_PARAMS_PATH = 'cqt_mean_std.npy' # <-- RUTA A LOS PARÁMETROS GUARDADOS
BEST_HIDDEN_DIM = 128 
INPUT_DIM = N_BINS 
OUTPUT_DIM = len(INVERSE_MIDI_MAP) 

# Asegurarse de que el script pueda encontrar las clases y utilidades
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------
# A. Clase Modelo (CRNN)
# ---------------------------------
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


# ---------------------------------
# B. Función de Extracción CQT
# ---------------------------------
def load_audio_cqt(path):
    """Carga audio y aplica CQT con la misma configuración de entrenamiento."""
    try:
        y, sr = librosa.load(path, sr=SR)
    except Exception as e:
        print(f"Error al cargar el audio: {e}")
        return None, None
        
    cqt = np.abs(librosa.cqt(
        y, sr=sr,
        hop_length=HOP,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE
    ))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max) 
    
    return cqt_db.T, len(y) / sr


# ---------------------------------
# C. Función de Conversión a MIDI 
# ---------------------------------
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
    frame_time_ticks = mido.second2tick(frame_time_sec, ticks_per_beat, micro_s_per_beat)
    frame_time_ticks = int(round(frame_time_ticks))
    
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
    print(f"\nArchivo MIDI guardado exitosamente en: {output_path}")


# ----------------------------------------------------
# D. Función de Visualización del MIDI (Piano Roll)
# ----------------------------------------------------
def visualize_midi_piano_roll(midi_path, sr=SR, hop_length=HOP):
    
    MIDI_IMAGE_PATH = midi_path.replace(".mid", "_pianoroll.png")
    
    try:
        midi_data = librosa_midi.PrettyMIDI(midi_path)
        
        fs = sr / hop_length
        piano_roll = midi_data.get_piano_roll(fs=fs) 
        
        active_notes = np.where(np.sum(piano_roll, axis=1) > 0)[0]
        
        if len(active_notes) == 0:
             print("Advertencia: El piano roll está vacío, no se detectaron notas para graficar.")
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
        plt.title('Piano Roll del MIDI Generado')
        
        plt.savefig(MIDI_IMAGE_PATH)
        print(f"✅ Imagen del Piano Roll guardada en: {MIDI_IMAGE_PATH}")
        plt.close() 

    except ImportError:
        print("ERROR: La librería 'pretty_midi' no está instalada.")
        print("Instala con: pip install pretty-midi")
    except Exception as e:
        print(f"ERROR durante la visualización del MIDI: {e}")


# =========================================================================
# II. FUNCIÓN PRINCIPAL DE INFERENCIA (CORREGIDA)
# =========================================================================

def run_inference(audio_path, output_midi_path):
    """
    Ejecuta el pipeline Audio -> CQT -> NORMALIZACIÓN -> CRNN -> MIDI -> Visualización.
    """
    
    # --- CARGAR PARÁMETROS DE NORMALIZACIÓN ---
    try:
        norm_params = np.load(NORMALIZATION_PARAMS_PATH)
        global_mean = norm_params[0]
        global_std = norm_params[1]
        print(f"Parámetros de Normalización cargados: Media={global_mean:.4f}, Std={global_std:.4f}")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de parámetros de normalización en {NORMALIZATION_PARAMS_PATH}.")
        print("Asegúrate de ejecutar el script de preprocesamiento para crear cqt_mean_std.npy.")
        return
    
    # 1. Cargar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Inicializar y Cargar Modelo
    model = CRNN(input_dim=INPUT_DIM, hidden_dim=BEST_HIDDEN_DIM, output_dim=OUTPUT_DIM)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Modelo CRNN cargado y listo para inferencia.")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo del modelo en {MODEL_PATH}.")
        return

    # 3. Preprocesar Audio (CQT)
    print(f"Procesando audio: {audio_path}")
    cqt_features, duration = load_audio_cqt(audio_path)
    
    if cqt_features is None:
        return
        
    T = cqt_features.shape[0]
    print(f"Duración del audio: {duration:.2f}s ({T} frames)")
    
    # 4. APLICAR NORMALIZACIÓN Z-SCORE (¡PASO CRÍTICO!)
    # Se aplica la misma media y std global usados en el entrenamiento.
    cqt_features_norm = (cqt_features - global_mean) / global_std
    
    # 5. Convertir a Tensor e inserta la dimensión del batch (1, T, 252)
    X_test_tensor = torch.tensor(cqt_features_norm, dtype=torch.float32).unsqueeze(0).to(device)

    # 6. Predicción
    with torch.no_grad():
        outputs = model(X_test_tensor) 
    predictions = torch.argmax(outputs.squeeze(0), dim=-1).cpu().numpy()

    # ** Diagnóstico Añadido: Verificar si hay notas activas predichas **
    if np.all(predictions == 0):
        print("\n** DIAGNÓSTICO: La predicción es 100% Clase 0 (Silencio). **")
        print("Esto podría indicar que el audio de prueba es silencioso o que la normalización aún no es la solución.")
    else:
        print("\n✅ Notas activas predichas. Generando MIDI...")


    # 7. Generar el Archivo MIDI
    predictions_to_midi(predictions, output_midi_path)
    
    # 8. VISUALIZACIÓN
    visualize_midi_piano_roll(output_midi_path) 
    

# =========================================================================
# III. EJECUCIÓN
# =========================================================================

if __name__ == '__main__':
    
    # --- ¡ATENCIÓN! CONFIGURA ESTA RUTA ---
    INPUT_AUDIO = 'C:/Users/franc/Desktop/Tecnologia_del_habla/Audio_to_Midi/Prueba.wav' 
    OUTPUT_MIDI = 'output_prediction.mid'      
    
    # --- Asegúrate de que el archivo existe ---
    if not os.path.exists(INPUT_AUDIO):
        print(f"Error: El archivo de audio de entrada no se encontró en '{INPUT_AUDIO}'.")
        print("Por favor, actualiza la variable INPUT_AUDIO con la ruta correcta.")
    else:
        run_inference(INPUT_AUDIO, OUTPUT_MIDI)