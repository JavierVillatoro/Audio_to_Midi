import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# --- Configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CAMBIO PRINCIPAL: Reducimos la lista a solo las dos imágenes requeridas
DATA_CONFIG = [
    ('Midi_Original.PNG', r'$\bf{}$ MIDI Original'),
    ('Midi_Unet.PNG', r'$\bf{}$ U-Net'),
]

# Dimensiones objetivo
TARGET_WIDTH = 1600
TARGET_HEIGHT = 250
DPI = 300 

def load_and_process_image(path, target_w, target_h):
    """Carga, convierte a RGB y redimensiona."""
    if not os.path.exists(path):
        print(f"Error: Archivo no encontrado en la ruta: {path}")
        # Retorna imagen negra si falla
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
    try:
        img = Image.open(path).convert('RGB')
        # Usamos LANCZOS para mejor calidad al redimensionar
        img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return np.array(img_resized)
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

def generate_comparison_plot():
    num_plots = len(DATA_CONFIG)
    
    # Configuración de la figura
    # La altura se ajusta automáticamente según num_plots (2.5 * 2 = 5 pulgadas de alto)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, 
                             figsize=(12, 2.5 * num_plots),
                             sharex=True, sharey=False,
                             gridspec_kw={'hspace': 0.05}) 

    # Si por alguna razón num_plots fuera 1, axes no es una lista, lo convertimos
    if num_plots == 1:
        axes = [axes]

    print("--- Iniciando proceso de imágenes (Comparativa Original vs U-Net) ---")
    print(f"Directorio base: {BASE_DIR}")

    for i, (ax, (img_filename, label)) in enumerate(zip(axes, DATA_CONFIG)):
        
        full_img_path = os.path.join(BASE_DIR, img_filename)
        print(f"Procesando: {img_filename}...")
        
        # 1. Cargar y procesar
        img_data = load_and_process_image(full_img_path, TARGET_WIDTH, TARGET_HEIGHT)
        
        # 2. Visualizar
        ax.imshow(img_data, aspect='auto', interpolation='nearest')
        
        # 3. Estilizado
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=12, labelpad=15, fontweight='bold')
        ax.set_yticks([]) # Sin ticks en Y
        
        # Bordes finos
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#333333')

        # Configuración eje X
        if i == num_plots - 1:
            ax.tick_params(axis='x', labelsize=10)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(axis='x', length=0)

    # Guardar con un nombre específico para esta comparativa
    output_filename = 'comparacion_original_vs_unet.png'
    output_path = os.path.join(BASE_DIR, output_filename)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\n¡Éxito! Gráfica generada en: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    generate_comparison_plot()