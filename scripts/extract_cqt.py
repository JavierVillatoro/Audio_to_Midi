import sys
import os
import numpy as np

# Asegura que el path sea correcto para encontrar utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.audio_utils import load_audio_cqt

DATASET = "data/"
OUTPUT = "features/cqt/"
os.makedirs(OUTPUT, exist_ok=True)

files = [f for f in os.listdir(DATASET) if f.endswith(".wav")]

for f in files:
    # cqt es devuelto en forma [n_frames, n_bins] -> [667, 252]
    cqt, _, _ = load_audio_cqt(os.path.join(DATASET, f)) 
    
    # 🌟 CORRECCIÓN DE ALINEACIÓN: Recortar el último frame (el 667)
    # cqt[:-1, :] toma todos los frames EXCEPTO el último
    # Esto asegura que la forma final sea [666, 252]
    cqt_aligned = cqt[:-1, :] 
    
    np.save(os.path.join(OUTPUT, f.replace(".wav", ".npy")), cqt_aligned)
    
    # Opcional: imprimir la forma para verificar
    # print(f"CQT guardado: {f}, Forma: {cqt_aligned.shape}")
    
    print("CQT guardado:", f)
