import librosa
import numpy as np

SR = 16000
HOP = 96
BINS_PER_OCTAVE = 36
N_BINS = 7 * BINS_PER_OCTAVE


def load_audio_cqt(path):
    y, sr = librosa.load(path, sr=SR)
    cqt = np.abs(librosa.cqt(
        y, sr=sr,
        hop_length=HOP,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE
    ))
    cqt_db = librosa.amplitude_to_db(cqt)
    
    # cqt_db tiene la forma [bins, frames], por defecto [252, 667]
    # CORRECCIÓN DE ALINEACIÓN: Recortar el último frame (el 667)
    # cqt_aligned ahora tiene la forma [252, 666]
    cqt_aligned = cqt_db[:, :-1] 
    
    # Retornar la transpuesta: [frames, bins] -> [666, 252]
    return cqt_aligned.T, y, sr
