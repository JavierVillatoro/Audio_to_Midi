import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Añadir el path de utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.audio_utils import SR, load_audio_cqt

# -------------------------------
# Configuración
# -------------------------------
DATASET = "data/"
AUDIO_FILE = "voice_002.wav"  # Cambia el archivo que quieras visualizar
path = os.path.join(DATASET, AUDIO_FILE)

# -------------------------------
# Cargar audio y calcular CQT
# -------------------------------
cqt_db, y, sr = load_audio_cqt(path)

# Información del audio
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
    y_axis='cqt_note',         # Puedes cambiar a 'hz' si quieres
    bins_per_octave=36
)
plt.colorbar(format='%+2.0f dB')
plt.title(f"CQT - {AUDIO_FILE}")

# -------------------------------
# Guardar la imagen
# -------------------------------
output_image = os.path.join("results", AUDIO_FILE.replace(".wav", "_cqt.png"))
os.makedirs(os.path.dirname(output_image), exist_ok=True)
plt.savefig(output_image, dpi=300, bbox_inches='tight')

# Mostrar la imagen
plt.show()
print(f"Imagen guardada en: {output_image}")


