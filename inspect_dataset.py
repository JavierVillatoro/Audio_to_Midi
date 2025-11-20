import numpy as np
import os

def inspect_consolidated_dataset(X_path="dataset_X.npy", Y_path="dataset_Y.npy"):
    """
    Carga e inspecciona los arrays consolidados dataset_X y dataset_Y.
    """
    print("-" * 60)
    print("ANALIZANDO DATASETS CONSOLIDADOS")
    print("-" * 60)

    try:
        # 1. Cargar datos
        X = np.load(X_path, allow_pickle=True)
        Y = np.load(Y_path, allow_pickle=True)
        
        # 2. Información General
        num_files = X.shape[0]
        print(f"Número total de archivos (muestras): {num_files}")
        print(f"Tipo de datos de la matriz contenedora: {X.dtype}") # Debe ser object
        
        # 3. Inspeccionar el primer elemento (Ej: voice_001)
        print("\n--- INSPECCIÓN DEL PRIMER ARCHIVO (Índice 0) ---")
        
        # --- INPUT (X): CQT ---
        X0 = X[0]
        print("▶️ INPUT (CQT - X[0])")
        print(f"  - Forma (Frames, Bins): {X0.shape}")
        
        # Verificar la alineación y la dimensionalidad
        if X0.shape[0] != X[0].shape[0]:
            print("  🚨 ADVERTENCIA: Las longitudes de las secuencias X e Y no coinciden.")
        
        # Muestra una porción para verificar el contenido (normalizado)
        print(f"  - Rango de valores: [{np.min(X0):.2f}] a [{np.max(X0):.2f}]")
        print("  - Muestra (Primeros 5 Frames x 5 Bins - Normalizados):")
        with np.printoptions(precision=2, suppress=True):
            print(X0[:5, :5])

        # --- OUTPUT (Y): LABELS ---
        Y0 = Y[0]
        print("\n▶️ OUTPUT (LABELS - Y[0])")
        print(f"  - Forma: {Y0.shape}")
        print(f"  - Valores Únicos (Clases): {np.unique(Y0)}")
        
        # Muestra la secuencia de etiquetas del inicio
        print(f"  - Muestra (Primeros 10 Frames de Etiquetas):")
        print(Y0[:10].flatten())
        
        # 4. Inspeccionar un elemento intermedio (Ej: mitad del dataset)
        mid_index = num_files // 2
        X_mid = X[mid_index]
        Y_mid = Y[mid_index]

        print(f"\n--- INSPECCIÓN DE UN ARCHIVO INTERMEDIO (Índice {mid_index}) ---")
        print(f"▶️ INPUT (CQT - X[{mid_index}]) Forma: {X_mid.shape}")
        print(f"▶️ OUTPUT (LABELS - Y[{mid_index}]) Forma: {Y_mid.shape}")

    except FileNotFoundError as e:
        print(f"❌ ERROR: No se encontró el archivo: {e.filename}")
        print("Asegúrate de ejecutar primero built_dataset.py.")

if __name__ == "__main__":
    inspect_consolidated_dataset()