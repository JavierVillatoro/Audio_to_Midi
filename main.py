from utils.midi_utils import midi_to_sequence_window
from utils.inv_midi import sequence_to_midi

resultado = midi_to_sequence_window("data/Midi_5.mid")
sequence_to_midi(resultado, output_path="reconstruir.mid",
                 sr=16000, hop_length=96, velocity=64,
                 tempo_bpm=120, ticks_per_beat=480, stretch=1.0)