import os
import librosa
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.audio_utils import SR
from utils.midi_utils import midi_to_sequence_window

DATASET = "data/"
OUTPUT = "features/labels/"
os.makedirs(OUTPUT, exist_ok=True)

files = [f for f in os.listdir(DATASET) if f.endswith(".wav")]

for audio_name in files:
    midi_name = audio_name.replace("voice_", "midi_").replace(".wav", ".mid")

    #Para caso en el que no haya fijado el tamaño (ver midi_utils)
    #y, sr = librosa.load(os.path.join(DATASET, audio_name), sr=SR)
    #dur = librosa.get_duration(y=y, sr=sr)

    labels = midi_to_sequence_window(os.path.join(DATASET, midi_name))
    
    # Para el caso en el que no haya fijado el tamaño 
    #labels = midi_to_sequence_window(os.path.join(DATASET, midi_name), dur) 
    np.save(os.path.join(OUTPUT, audio_name.replace(".wav", ".npy")), labels)

    print("Etiquetas guardadas:", audio_name)
