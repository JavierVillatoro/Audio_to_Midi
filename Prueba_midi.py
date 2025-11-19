#Script para probar que los midi han sido vectorizados correctamente con sus etiquetas

from utils.midi_utils import midi_to_sequence_window
from utils.inv_midi import sequence_to_midi

resultado = midi_to_sequence_window("data/midi_201.mid")  
sequence_to_midi(resultado, output_path="reconstruir_7.mid",
                 sr=16000, hop_length=96, velocity=64,
                 tempo_bpm=120, ticks_per_beat=480, stretch=1.0)
print(resultado.size,resultado)