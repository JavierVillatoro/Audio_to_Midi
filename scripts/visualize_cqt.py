import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.audio_utils import SR, load_audio_cqt


DATASET = "data/"
AUDIO_FILE = "voice_001.wav"  # Cambia el archivo que quieras visualizar

path = os.path.join(DATASET, AUDIO_FILE)

# -------------------------------
# Cargar audio y calcular CQT
# -------------------------------
cqt_db, y, sr = load_audio_cqt(path)

# cqt_db: frames x bins
print(f"CQT shape: {cqt_db.shape}")
print(f"Audio duration: {librosa.get_duration(y=y, sr=sr):.2f} s")

# -------------------------------
# Visualización
# -------------------------------
plt.figure(figsize=(12, 6))
librosa.display.specshow(
    cqt_db.T,  # Transponemos para que eje x sea tiempo
    sr=sr,
    hop_length=96,             # igual que en tu CQT
    x_axis='time',
    y_axis='cqt_hz',
    bins_per_octave=36
)
plt.colorbar(format='%+2.0f dB')
plt.title(f"CQT - {AUDIO_FILE}")
plt.show()
