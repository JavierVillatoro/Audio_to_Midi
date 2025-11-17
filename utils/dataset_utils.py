#Este archivo incluye funciones útiles para preparar datasets

import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# Normalización (muy importante para modelos de audio)
# ----------------------------------------------------
def normalize_cqt(cqt):
    """
    Normaliza un CQT (por archivo) a media 0 y varianza 1.
    """
    mean = np.mean(cqt)
    std = np.std(cqt) + 1e-8
    return (cqt - mean) / std


# ----------------------------------------------------
# Padding de secuencias (si se quiere modelo batch)
# ----------------------------------------------------
def pad_sequences(sequences, max_len=None, pad_value=0):
    """
    Padding para secuencias variables.
    sequences: lista de arrays (frames, features)
    """
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)

    padded = []
    for seq in sequences:
        length = seq.shape[0]
        if length < max_len:
            pad_width = ((0, max_len - length), (0, 0))
            seq_padded = np.pad(seq, pad_width, mode="constant", constant_values=pad_value)
        else:
            seq_padded = seq[:max_len]
        padded.append(seq_padded)

    return np.array(padded)


def pad_labels(labels, max_len=None, pad_value=0):
    """
    Padding para secuencias de etiquetas.
    """
    if max_len is None:
        max_len = max(len(l) for l in labels)

    padded = []
    for l in labels:
        length = len(l)
        if length < max_len:
            l_padded = np.concatenate([l, np.full(max_len - length, pad_value)])
        else:
            l_padded = l[:max_len]
        padded.append(l_padded)

    return np.array(padded)


# ----------------------------------------------------
# División del dataset
# ----------------------------------------------------
def split_dataset(X, Y, test_size=0.15, val_size=0.15):
    """
    Divide X, Y en train, val y test.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    val_ratio_adjusted = val_size / (1 - test_size)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=val_ratio_adjusted, random_state=42
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ----------------------------------------------------
# Cargar dataset ya procesado
# ----------------------------------------------------
def load_full_dataset(X_path="dataset_X.npy", Y_path="dataset_Y.npy"):
    X = np.load(X_path, allow_pickle=True)
    Y = np.load(Y_path, allow_pickle=True)
    return X, Y
