import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# --- Configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_CONFIG = [
    ('Pitch_estimado.PNG', r'$\bf{}$ CQT'), 
    ('Midi_Unet.PNG', r'$\bf{}$ U-Net'),
]

TARGET_WIDTH = 1600
TARGET_HEIGHT = 250
DPI = 300 

def load_and_process_image(path, target_w, target_h):
    if not os.path.exists(path):
        print(f"Error: Archivo no encontrado: {path}")
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    try:
        img = Image.open(path).convert('RGB')
        img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return np.array(img_resized)
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

def generate_comparison_plot():
    num_plots = len(DATA_CONFIG)
    
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, 
                             figsize=(12, 2.5 * num_plots),
                             sharex=True, sharey=False,
                             gridspec_kw={'hspace': 0.05}) 

    if num_plots == 1: axes = [axes]

    print("--- Generando gráfica limpia (Sin ejes X) ---")

    for i, (ax, (img_filename, label)) in enumerate(zip(axes, DATA_CONFIG)):
        
        full_img_path = os.path.join(BASE_DIR, img_filename)
        
        # 1. Cargar
        img_data = load_and_process_image(full_img_path, TARGET_WIDTH, TARGET_HEIGHT)
        
        # --- CAMBIO DE COLOR (Mismo que antes) ---
        if img_filename == 'Midi_Unet.PNG':
            grayscale = img_data.mean(axis=2)
            mask_notes = grayscale > 50 
            new_img = np.ones_like(img_data) * 255 # Fondo Blanco
            new_img[mask_notes] = [255, 0, 0]      # Notas Rojas
            img_data = new_img
        # -----------------------------------------

        # 2. Visualizar
        ax.imshow(img_data, aspect='auto', interpolation='nearest')
        
        # 3. Etiquetas laterales
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=12, labelpad=15, fontweight='bold')
        
        # 4. LIMPIEZA TOTAL DE EJES
        ax.set_yticks([])     # Quitar números eje Y
        ax.set_xticks([])     # Quitar números eje X
        ax.tick_params(axis='both', which='both', length=0) # Quitar las "rayitas" de los ticks
        
        # Bordes finos
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#333333')

    output_filename = 'comparacion_spectrogram_vs_unet_clean.png'
    output_path = os.path.join(BASE_DIR, output_filename)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\n¡Listo! Imagen limpia guardada en: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    generate_comparison_plot()