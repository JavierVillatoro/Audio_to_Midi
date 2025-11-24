import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset_utils import normalize_cqt

CQT_PATH = "features/cqt/"
LBL_PATH = "features/labels/"

X = []
Y = []


## NORMALIZAR CQT????


files = sorted([f for f in os.listdir(CQT_PATH) if f.endswith(".npy")])

for f in files:
    cqt = np.load(os.path.join(CQT_PATH, f), allow_pickle=True)
    labels = np.load(os.path.join(LBL_PATH, f), allow_pickle=True)

    min_len = min(len(cqt), len(labels))
    X.append(cqt[:min_len])
    Y.append(labels[:min_len])

X = np.array(X, dtype=object)
Y = np.array(Y, dtype=object)

np.save("dataset_X.npy", X)
np.save("dataset_Y.npy", Y)

print("Dataset creado: dataset_X.npy, dataset_Y.npy")
