import mido
import numpy as np

# Mapeo
MIDI_MAP = {-1: 0, 60: 1, 64: 2, 67: 3}  # Silencio, C3, E3, G3

def midi_to_sequence_window(midi_path, sr=16000, hop_length=96, duration=4.0):

    mid = mido.MidiFile(midi_path)

    # Duración real del MIDI
    midi_duration = sum(msg.time for msg in mid)

    # Usamos SIEMPRE 4 segundos como duración objetivo
    total_frames = int(duration * sr / hop_length)

    # Vector inicial lleno de silencio (-1)
    midi_seq = np.full(total_frames, -1, dtype=int)

    # Trackeo de notas activas
    active_notes = {}
    current_time = 0.0

    for msg in mid:
        current_time += msg.time

        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = current_time

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active_notes:
                start_time = active_notes.pop(msg.note)

                start_frame = int(start_time * sr / hop_length)
                end_frame = int(current_time * sr / hop_length)

                # Limitar a los 4 segundos
                start_frame = max(0, min(start_frame, total_frames))
                end_frame = max(0, min(end_frame, total_frames))

                midi_seq[start_frame:end_frame] = msg.note

    # Convertimos notas→labels
    labels = np.array([MIDI_MAP.get(n, 0) for n in midi_seq])

    return labels


