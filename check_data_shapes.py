import numpy as np
import os

# --- Configuración de Rutas ---
CQT_DIR = "features/cqt/"
LABELS_DIR = "features/labels/"

def check_full_data(file_name_base):
    """
    Carga y muestra el contenido completo de las etiquetas y una
    muestra amplia de la matriz CQT.
    """
    
    # Construir nombres de archivo
    cqt_file = os.path.join(CQT_DIR, f"{file_name_base}.npy")
    labels_file = os.path.join(LABELS_DIR, f"{file_name_base}.npy")
    
    # ----------------------------------------------------
    print("-" * 60)
    print(f"👁️ INSPECCIONANDO ARCHIVO BASE: {file_name_base}.npy")
    print("-" * 60)
    
    # --- CQT (Input): Muestra Detallada ---
    print("## 📊 MATRIZ CQT (Características de Entrada)")
    try:
        cqt_data = np.load(cqt_file)
        
        frames, bins = cqt_data.shape
        print(f"✅ SHAPE: ({frames}, {bins})")
        print(f"   (Debería ser (666, 252) si está alineado)")
        print(f"   Rango de valores (dB): [{np.min(cqt_data):.2f} dB] a [{np.max(cqt_data):.2f} dB]")
        
        # Mostraremos una sección para verificar la continuidad temporal
        # y la frecuencia. Mostramos los primeros 10 frames y los 10 primeros bins.
        print("\n   --- MUESTRA (Primeros 10 Frames x 10 Bins) ---")
        # numpy.set_printoptions(threshold=...) nos permite ver más
        with np.printoptions(precision=2, suppress=True):
            print(cqt_data[:10, :10])
        
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo CQT no encontrado en {cqt_file}")
    
    print("\n" + "=" * 60 + "\n")
    
    # --- LABELS (Output): Secuencia Completa ---
    print("## 📝 VECTOR DE ETIQUETAS (Salida Deseada)")
    try:
        labels_data = np.load(labels_file)
        
        # Aplanamos el vector para imprimirlo fácilmente
        labels_flat = labels_data.flatten()
        num_frames = labels_flat.shape[0]

        print(f"✅ LONGITUD TOTAL DE FRAMES: {num_frames}")
        print(f"   (Debería ser 666)")
        print(f"   Valores Únicos (Clases): {np.unique(labels_flat)}")
        
        print("\n   --- SECUENCIA COMPLETA DE ETIQUETAS (666 Valores) ---")
        
        # Ajustamos las opciones de impresión temporalmente para mostrar el vector completo
        with np.printoptions(threshold=np.inf):
            print(labels_flat)
        
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo LABELS no encontrado en {labels_file}")
    
    print("\n" + "-" * 60)


if __name__ == "__main__":
    # --- MODIFICA ESTA LÍNEA ---
    FILE_TO_CHECK = "voice_002" 
    
    check_full_data(FILE_TO_CHECK)