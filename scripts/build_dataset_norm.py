import numpy as np
import os

CQT_PATH = "features/cqt/"
LBL_PATH = "features/labels/"

X = []
Y = []

files = sorted([f for f in os.listdir(CQT_PATH) if f.endswith(".npy")])

# 1. CONSOLIDAR DATOS 
for f in files:
    cqt = np.load(os.path.join(CQT_PATH, f), allow_pickle=True)
    labels = np.load(os.path.join(LBL_PATH, f), allow_pickle=True)

    min_len = min(len(cqt), len(labels))
    X.append(cqt[:min_len])
    Y.append(labels[:min_len])
    
# Convertir a array de objetos para manejar longitudes variables (si las hubiera)
X = np.array(X, dtype=object) 

# 2. CALCULAR PARÁMETROS GLOBALES
# Aplanamos todos los CQTs en un solo vector para calcular la media y std de todo el dataset.
all_cqt = np.concatenate(X) 
global_mean = np.mean(all_cqt)
global_std = np.std(all_cqt) + 1e-8 # Añadir epsilon para evitar división por cero

# 3. APLICAR NORMALIZACIÓN GLOBAL
X_normalized = []
for cqt_file in X:
    cqt_norm = (cqt_file - global_mean) / global_std
    X_normalized.append(cqt_norm)

# 4. GUARDAR DATASET CON NORMALIZACIÓN
X_final = np.array(X_normalized, dtype=object)
Y_final = np.array(Y, dtype=object)

np.save("dataset_X.npy", X_final)
np.save("dataset_Y.npy", Y_final)

# OPCIONAL: Guardar los parámetros de normalización para usarlos en el test set o inferencia
np.save("cqt_mean_std.npy", np.array([global_mean, global_std]))

print("Dataset creado y normalizado: dataset_X.npy, dataset_Y.npy")
print(f"Parámetros de Normalización guardados: Media={global_mean:.4f}, Std={global_std:.4f}")