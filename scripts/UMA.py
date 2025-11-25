import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- Configuración de Rutas (Se mantiene igual) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from utils.dataset_utils import load_full_dataset
except ImportError:
    print("ADVERTENCIA: No se pudo importar 'load_full_dataset'.")
    print("Asegúrate de que tus archivos .npy estén en la misma carpeta o ajusta la ruta.")
    def load_full_dataset(X_path, Y_path):
        return np.load(X_path), np.load(Y_path)


# -----------------------------------------------------
# 1. Carga de Datos y Aplanamiento (Flattening)
# -----------------------------------------------------

# Cargar los datasets completos.
try:
    X_full, Y_full = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
    print(f"Dataset X cargado. Forma original: {X_full.shape}")
    print(f"Dataset Y cargado. Forma original: {Y_full.shape}\n")
except Exception as e:
    print(f"ERROR FATAL al cargar archivos: {e}")
    sys.exit(1)


# --- CAMBIO CLAVE: APLANAR EL DATASET COMPLETO ---

# X_full shape: (300, 666, 252) -> Queremos (300 * 666, 252) = (199800, 252)
X_total = X_full.reshape(-1, X_full.shape[-1])

# Y_full shape: (300, 666) -> Queremos (300 * 666,) = (199800,)
Y_total = Y_full.flatten()

print(f"Dataset aplanado para UMAP: X_total.shape = {X_total.shape}")
print(f"Etiquetas aplanadas para UMAP: Y_total.shape = {Y_total.shape}")


# Comprobar la distribución de clases en el dataset total
unique_labels = np.unique(Y_total)
N_TOTAL_CLASSES = len(unique_labels)
print(f"Clases Únicas detectadas en el dataset total: {unique_labels}")
print(f"Total de Clases para visualización: {N_TOTAL_CLASSES}\n")

if N_TOTAL_CLASSES < 4:
    print("🚨 ADVERTENCIA: El dataset total tiene menos de 4 clases únicas. Revisar etiquetas.")

# -----------------------------------------------------
# 2. Reducción de Dimensionalidad con UMAP
# -----------------------------------------------------
print(f"Iniciando UMAP: Reduciendo {X_total.shape[0]} frames de {X_total.shape[1]} a 2 dimensiones...")

# UMAP es computacionalmente intensivo con 199,800 puntos.
# Puedes reducir 'n_neighbors' o aumentar 'n_components' si tarda mucho.

reducer = umap.UMAP(
    n_neighbors=10,        # Configuracion inicual n_neighbors = 15
    min_dist=0.01,          # Configuración inicial min_dist = 0.1
    n_components=2,        # n_components = 2,
    random_state=42,        #42
    # Sugerencia: Puedes limitar el número de frames para que sea más rápido
    # max_samples=10000 
)

# Aplicar la reducción al dataset completo aplanado
umap_data_total = reducer.fit_transform(X_total)

print("Reducción completada. Forma del resultado:", umap_data_total.shape)


# -----------------------------------------------------
# 3. Visualización con Seaborn y Matplotlib
# -----------------------------------------------------

plt.figure(figsize=(12, 10))

# Definimos la paleta de colores fija para las 4 clases (0, 1, 2, 3)
fixed_palette = sns.color_palette("tab10", N_TOTAL_CLASSES) 
palette_map = {str(label): fixed_palette[i] for i, label in enumerate(unique_labels)}

# Usar Seaborn para crear el gráfico de dispersión
sns.scatterplot(
    x=umap_data_total[:, 0],
    y=umap_data_total[:, 1],
    hue=Y_total.astype(str),
    palette=palette_map,
    hue_order=[str(label) for label in unique_labels],
    s=5, # Reducimos el tamaño del punto (s) para tantos datos
    alpha=0.5
)

plt.title('Visualización UMAP del Dataset CQT Completo (Todos los Frames)')
plt.xlabel('Componente UMAP 1')
plt.ylabel('Componente UMAP 2')
plt.legend(title='Clase (Etiqueta)', loc='best')
plt.grid(True, linestyle='--', alpha=0.3)

# Guardar la imagen para inspección
OUTPUT_FILENAME = 'umap_cqt_visualization_TOTAL_DATASET.png'
plt.savefig(OUTPUT_FILENAME)
plt.close()

print(f"\n✅ Gráfico del Dataset Total guardado con éxito en: '{OUTPUT_FILENAME}'")