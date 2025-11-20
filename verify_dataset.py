# Script para comprobar que el dataset se ha hecho correctamente

import os
import re
import librosa
import numpy as np
from utils.audio_utils import SR
from utils.midi_utils import midi_to_sequence_window

DATASET = "data/"

# ================================================================
# 0. VERIFICAR NOMBRES DE ARCHIVOS
# ================================================================
print("=== Verificando formato de nombres ===")

# Regex esperados
wav_regex = re.compile(r"voice_(\d+)\.wav$")
mid_regex = re.compile(r"midi_(\d+)\.mid$")

wav_files = sorted([f for f in os.listdir(DATASET) if f.endswith(".wav")])
mid_files = sorted([f for f in os.listdir(DATASET) if f.endswith(".mid")])

bad_wav_names = [f for f in wav_files if wav_regex.match(f) is None]
bad_mid_names = [f for f in mid_files if mid_regex.match(f) is None]

# Archivos con número correlativo
wav_ids = sorted([int(wav_regex.match(f).group(1)) for f in wav_files if wav_regex.match(f)])
mid_ids = sorted([int(mid_regex.match(f).group(1)) for f in mid_files if mid_regex.match(f)])

# Comprobación de nombres válidos
if len(bad_wav_names) == 0:
    print("✔ Todos los WAV tienen nombre válido (voice_XXX.wav)")
else:
    print("❌ WAV con nombres incorrectos:")
    for f in bad_wav_names:
        print("  ", f)

if len(bad_mid_names) == 0:
    print("✔ Todos los MID tienen nombre válido (midi_XXX.mid)")
else:
    print("❌ MID con nombres incorrectos:")
    for f in bad_mid_names:
        print("  ", f)

# Comprobación de coincidencia de IDs
missing_mid = []
for wid in wav_ids:
    if wid not in mid_ids:
        missing_mid.append(wid)

if len(missing_mid) == 0:
    print("✔ Cada voice_XXX.wav tiene su correspondiente midi_XXX.mid")
else:
    print("❌ Faltan MID para los siguientes IDs:")
    for mid in missing_mid:
        print(f"  midi_{mid:03d}.mid")


# ================================================================
# 1. VERIFICAR DURACIÓN Y SR DE ARCHIVOS WAV
# ================================================================
print("\n=== Verificando duración y frecuencia de muestreo de archivos WAV ===")

dur_errors = []
sr_errors = []

for f in wav_files:
    match = wav_regex.match(f)
    if match is None:
        continue  # evitar errores si el nombre no coincide

    path = os.path.join(DATASET, f)
    y, sr = librosa.load(path, sr=None)  # sr=None para leer sr real
    dur = librosa.get_duration(y=y, sr=sr)

    # Duración
    if abs(dur - 4.0) > 0.02:  # margen razonable
        dur_errors.append((f, dur))

    # Frecuencia de muestreo
    if sr != 16000:
        sr_errors.append((f, sr))


print(f"Total WAV encontrados: {len(wav_files)}")

if len(dur_errors) == 0:
    print("✔ Todas las grabaciones duran ~4.0 s")
else:
    print("❌ Archivos con duración incorrecta:")
    for f, d in dur_errors:
        print(f"  {f}: {d:.4f} s")

if len(sr_errors) == 0:
    print("✔ Todos los WAV tienen frecuencia de muestreo 16000 Hz")
else:
    print("❌ Archivos con SR diferente a 16000 Hz:")
    for f, s in sr_errors:
        print(f"  {f}: {s} Hz")


# ================================================================
# 2. VERIFICAR LONGITUD LABELS MIDI
# ================================================================
print("\n=== Verificando longitud de vectores MIDI ===")

mid_errors = []
mid_files_missing = []

for f in wav_files:
    m = wav_regex.match(f)
    if m is None:
        continue

    idx = int(m.group(1))
    midi_name = f"midi_{idx:03d}.mid"
    midi_path = os.path.join(DATASET, midi_name)

    if not os.path.exists(midi_path):
        mid_files_missing.append(midi_name)
        continue

    labels = midi_to_sequence_window(midi_path)

    if len(labels) != 666:
        mid_errors.append((midi_name, len(labels)))

print(f"Total MID encontrados: {len(mid_files) - len(mid_files_missing)}")

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


