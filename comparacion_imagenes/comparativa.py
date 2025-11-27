import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# --- Configuración ---
# Obtenemos la ruta del directorio donde se encuentra ESTE script.
# Esto asegura que encuentre las imágenes si están en la misma carpeta.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Definimos el orden exacto de las imágenes usando los NOMBRES REALES
# que aparecen en tu explorador de archivos.
DATA_CONFIG = [
    # (Nombre del archivo real, Etiqueta profesional para el paper)
    ('Pitch_estimado.PNG', r'$\bf{a)}$ Pitch'),
    ('Midi_Original.PNG', r'$\bf{b)}$ MIDI Original'),
    ('Midi_crnn.PNG', r'$\bf{c)}$ CRNN'),
    # Nota: He mantenido el nombre tal cual aparece en tu captura ("Transforemer")
    ('Midi_Transforemer.PNG', r'$\bf{d)}$ Transformer'), 
    ('Midi_Transformer_Overfitting.PNG', r'$\bf{e)}$ Transformer'),
    ('Midi_Unet.PNG', r'$\bf{f)}$ Model: U-Net'),
]

# Dimensiones objetivo para estandarizar todas las imágenes (extender y aplanar).
TARGET_WIDTH = 1600
TARGET_HEIGHT = 250
DPI = 300  # Alta resolución para publicación

def load_and_process_image(path, target_w, target_h):
    """
    Carga una imagen desde una ruta absoluta, la convierte a RGB y 
    la redimensiona a las dimensiones objetivo.
    """
    if not os.path.exists(path):
        print(f"Error: Archivo no encontrado en la ruta: {path}")
        # Devuelve una imagen negra con un mensaje si falta el archivo
        placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return placeholder
        
    try:
        img = Image.open(path).convert('RGB')
        # Redimensionado de alta calidad (LANCZOS) para ajustar al tamaño objetivo
        img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return np.array(img_resized)
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

def generate_comparison_plot():
    num_plots = len(DATA_CONFIG)
    
    # Configuración de la figura para estilo "Research Paper"
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, 
                             figsize=(12, 2.5 * num_plots),
                             sharex=True, sharey=False,
                             gridspec_kw={'hspace': 0.05}) # Poco espacio entre plots

    print("--- Iniciando proceso de imágenes ---")
    print(f"Directorio base del script: {BASE_DIR}")

    # Iterar sobre las configuraciones y los ejes
    for i, (ax, (img_filename, label)) in enumerate(zip(axes, DATA_CONFIG)):
        
        # Construimos la ruta COMPLETA a la imagen
        full_img_path = os.path.join(BASE_DIR, img_filename)
        print(f"Procesando: {img_filename}...")
        
        # 1. Cargar y procesar
        img_data = load_and_process_image(full_img_path, TARGET_WIDTH, TARGET_HEIGHT)
        
        # 2. Visualizar
        # 'aspect="auto"' estira la imagen para llenar el recuadro
        ax.imshow(img_data, aspect='auto', interpolation='nearest')
        
        # 3. Estilizado Profesional
        # Etiqueta en el lado izquierdo rotada
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=11, labelpad=20)
        
        # Eliminar ticks y etiquetas del eje Y
        ax.set_yticks([])
        
        # Bordes finos y oscuros
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#333333')

        # Configuración para el eje X solo en el último plot
        if i == num_plots - 1:
            ax.set_xlabel('Time Frames', fontsize=12, fontweight='bold', labelpad=10)
            ax.tick_params(axis='x', labelsize=10)
        else:
            # Ocultar eje X en los intermedios
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(axis='x', length=0)

    # Título general
    fig.suptitle('Comparative Analysis: Pitch Estimation vs. Deep Learning Models', 
                 fontsize=16, y=0.995, fontweight='bold')
    
    # Guardar en alta resolución en la MISMA carpeta que el script
    output_path = os.path.join(BASE_DIR, 'figure_model_comparison_fixed.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\n¡Éxito! Gráfica generada en: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    # Solo necesitas ejecutar este script. Él se encargará de buscar
    # las imágenes en su misma carpeta.
    generate_comparison_plot()