import librosa
import numpy as np

SR = 16000
HOP = 96
BINS_PER_OCTAVE = 36
N_BINS = 7 * BINS_PER_OCTAVE

## C3 (130.8 Hz)
# F_MIN = librosa.note_to_hz('C3') 
## Necesito solo 2 octavas (Octava 3 y 4) para cubrir C4, E4, G4.
# N_BINS_REDUCIDO = 2 * BINS_PER_OCTAVE



def load_audio_cqt(path):
    y, sr = librosa.load(path, sr=SR)
    cqt = np.abs(librosa.cqt(
        y, sr=sr,
        hop_length=HOP,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE
        #fmin=F_MIN # Uso la nueva frecuencia de inicio
    ))
    cqt_db = librosa.amplitude_to_db(cqt)
    return cqt_db.T, y, sr  # frames × bins
