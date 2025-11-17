import mido
import numpy as np

# Mapeo
MIDI_MAP = {-1: 0, 60: 1, 64: 2, 67: 3}  # Silencio, C4, E4, G4

def midi_to_sequence_window(midi_path, sr=8000, hop_length=48):  # Probar diferentes
    mid = mido.MidiFile(midi_path)

    # Calcular duración total en segundos
    audio_length = sum(msg.time for msg in mid)
    total_frames = int(audio_length * sr / hop_length)
    
    # Array de frames inicializado a silencio
    midi_seq = np.full(total_frames, -1, dtype=int)

    # Diccionario para trackear notas activas
    active_notes = {}

    current_time = 0.0
    for msg in mid:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            # Guardar tiempo de inicio de la nota
            active_notes[msg.note] = current_time
        elif msg.type == "note_off" or (msg.type=="note_on" and msg.velocity==0):
            if msg.note in active_notes:
                start_time = active_notes.pop(msg.note)
                # Convertir tiempo a frames
                start_frame = int(start_time * sr / hop_length)
                end_frame = int(current_time * sr / hop_length)
                midi_seq[start_frame:end_frame] = msg.note

    # Mapear a etiquetas finales
    labels = np.array([MIDI_MAP.get(n, 0) for n in midi_seq])
    return labels

