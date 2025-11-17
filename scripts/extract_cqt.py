import os
from utils.audio_utils import load_audio_cqt
import numpy as np

DATASET = "data/"
OUTPUT = "features/cqt/"
os.makedirs(OUTPUT, exist_ok=True)

files = [f for f in os.listdir(DATASET) if f.endswith(".wav")]

for f in files:
    cqt, _, _ = load_audio_cqt(os.path.join(DATASET, f))
    np.save(os.path.join(OUTPUT, f.replace(".wav", ".npy")), cqt)
    print("CQT guardado:", f)
