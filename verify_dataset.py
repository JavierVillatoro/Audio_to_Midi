#Verifico el tamaño del dataset es correcto 

import os
import librosa
import numpy as np
from utils.audio_utils import SR
from utils.midi_utils import midi_to_sequence_window

DATASET = "data/"

# --------------------------------------
# 1. Verificar duración de los WAV
# --------------------------------------
print("=== Verificando duración de archivos WAV ===")

wav_files = sorted([f for f in os.listdir(DATASET) if f.endswith(".wav")])

dur_errors = []

for f in wav_files:
    path = os.path.join(DATASET, f)
    y, sr = librosa.load(path, sr=SR)
    dur = librosa.get_duration(y=y, sr=sr)

    if abs(dur - 4.0) > 0.02:  # margen de 20 ms
        dur_errors.append((f, dur))

print(f"Total WAV encontrados: {len(wav_files)}")
if len(dur_errors) == 0:
    print("✔ Todas las grabaciones duran ~4.0 s")
else:
    print("❌ Archivos con duración incorrecta:")
    for f, d in dur_errors:
        print(f"  {f}: {d:.4f} s")


# --------------------------------------
# 2. Verificar longitud de labels MIDI
# --------------------------------------
print("\n=== Verificando longitud de vectores MIDI ===")

mid_errors = []
mid_files_missing = []

for f in wav_files:
    midi_name = f.replace("voice_", "midi_").replace(".wav", ".mid")
    midi_path = os.path.join(DATASET, midi_name)

    if not os.path.exists(midi_path):
        mid_files_missing.append(midi_name)
        continue

    labels = midi_to_sequence_window(midi_path)

    if len(labels) != 666:
        mid_errors.append((midi_name, len(labels)))

print(f"Total MID encontrados: {len(wav_files) - len(mid_files_missing)}")
if len(mid_files_missing) > 0:
    print("❌ Faltan archivos MIDI:")
    for f in mid_files_missing:
        print("  ", f)

if len(mid_errors) == 0:
    print("✔ Todos los vectores MIDI tienen longitud 666")
else:
    print("❌ Archivos MIDI con longitud incorrecta:")
    for f, l in mid_errors:
        print(f"  {f}: {l} frames")
