import mido
import numpy as np

# Mapeo inverso: etiquetas -> notas MIDI
INV_MIDI_MAP = {0: -1, 1: 60, 2: 64, 3: 67}

def sequence_to_midi(labels,
                     output_path="reconstruir.mid",
                     sr=8000,
                     hop_length=48,
                     velocity=64,
                     tempo_bpm=120,
                     ticks_per_beat=480,
                     stretch=1.0):
    """
    Convierte un array de etiquetas en un archivo MIDI.
    stretch: factor para aumentar la duración de cada frame (por si necesitas alargar).
    """

    # Crear MIDI y pista
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Fijar tempo explícito (microsegundos por negra)
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # ticks por segundo: ticks/seg = ticks_per_beat / (segundos_por_negra)
    # segundos_por_negra = tempo / 1_000_000
    ticks_per_second = ticks_per_beat * 1_000_000.0 / tempo

    # duración real por frame (en segundos), escalada por stretch
    frame_duration = ((hop_length / sr) * stretch)
     

    prev_note = -1
    last_event_time_sec = 0.0  # tiempo absoluto del último evento añadido (en segundos)

    for i, label in enumerate(labels):
        note = INV_MIDI_MAP.get(int(label), -1)
        event_time_sec = i * frame_duration

        # solo actuamos cuando cambia la etiqueta (nota diferente)
        if note != prev_note:
            # delta real entre este evento y el último evento añadido
            delta_sec = event_time_sec - last_event_time_sec
            delta_ticks = int(round(delta_sec * ticks_per_second))
            if delta_ticks < 0:
                delta_ticks = 0

            # Si hay una nota previa abierta, la cerramos primero.
            # La primera de las dos acciones debe tomar delta_ticks; la siguiente tiempo=0.
            first_time = delta_ticks

            if prev_note != -1:
                # note_off para la nota previa (usa first_time)
                track.append(mido.Message('note_off', note=prev_note, velocity=0, time=first_time))
                first_time = 0  # siguiente mensaje (note_on) ocurre instantáneamente después

            if note != -1:
                # note_on para la nueva nota (usa first_time, que puede ser delta_ticks o 0)
                track.append(mido.Message('note_on', note=note, velocity=velocity, time=first_time))

            # actualizamos marker de tiempo del último evento añadido a event_time_sec
            last_event_time_sec = event_time_sec
            prev_note = note

    # cerrar última nota si quedó abierta (hasta el final de la secuencia)
    if prev_note != -1:
        end_time_sec = len(labels) * frame_duration
        delta_sec = end_time_sec - last_event_time_sec
        delta_ticks = int(round(delta_sec * ticks_per_second))
        if delta_ticks < 0:
            delta_ticks = 0
        track.append(mido.Message('note_off', note=prev_note, velocity=0, time=delta_ticks))

    mid.save(output_path)
    print(f"✔ MIDI guardado en: {output_path}")
    print(f"  tempo={tempo_bpm} BPM, ticks_per_beat={ticks_per_beat}, frame_duration={frame_duration:.6f}s, ticks_per_second={ticks_per_second:.2f}")



