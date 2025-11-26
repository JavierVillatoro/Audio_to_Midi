import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os 
import sys # Añadido para debugging


def recortar_bordes_blancos(ruta_imagen, umbral=240):
    """
    Carga una imagen y recorta automáticamente los márgenes blancos/claros 
    para aislar el gráfico.
    """
    img = Image.open(ruta_imagen).convert("RGB")
    np_img = np.array(img)

    # Crear una máscara de todo lo que NO es blanco (o casi blanco)
    mascara = np_img < umbral
    
    # Verificamos si hay algún pixel en cualquiera de los 3 canales (R, G, B)
    hay_datos = np.any(mascara, axis=2)
    
    # Encontrar las coordenadas del cuadro delimitador (bounding box)
    filas = np.any(hay_datos, axis=1)
    cols = np.any(hay_datos, axis=0)
    
    if not np.any(filas) or not np.any(cols):
        return np_img

    ymin, ymax = np.where(filas)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Recortar la imagen usando esos índices
    img_recortada = np_img[ymin:ymax+1, xmin:xmax+1]
    return img_recortada


def visualizar_comparacion(imagenes_info):
    n_imgs = len(imagenes_info)
    fig, axes = plt.subplots(n_imgs, 1, figsize=(15, 4 * n_imgs), sharex=True)
    plt.subplots_adjust(hspace=0.05)

    # *** FIX DE RUTA: Obtener la ruta del directorio del script ***
    # Esto asegura que la búsqueda de archivos siempre inicie donde está el .py
    directorio_script = os.path.dirname(os.path.abspath(__file__))

    for i, item in enumerate(imagenes_info):
        ax = axes[i]
        nombre_archivo = item['archivo']
        titulo = item['titulo']
        
        # *** FIX DE RUTA: Construir la ruta completa ***
        ruta_completa = os.path.join(directorio_script, nombre_archivo)
        
        try:
            # Mensaje de depuración: Muestra la ruta que está intentando abrir
            print(f"Intentando abrir: {ruta_completa}") 
            
            img_data = recortar_bordes_blancos(ruta_completa)
            
            ax.imshow(img_data, aspect='auto')
            ax.set_ylabel(titulo, rotation=0, labelpad=60, fontsize=10, weight='bold', ha='right')
            ax.set_yticks([])
            if i < n_imgs - 1:
                ax.set_xticks([])
            
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"ARCHIVO NO ENCONTRADO:\n{nombre_archivo}", 
                    ha='center', va='center', color='red')
            print(f"ERROR: No se encuentra el archivo en la ruta:\n{ruta_completa}", file=sys.stderr)

    axes[-1].set_xlabel("Tiempo (Alineado visualmente)", fontsize=12)
    plt.suptitle("Comparación: Espectrograma vs Modelos (Alineados)", fontsize=16, y=0.92)
    plt.show()

# --- CONFIGURACIÓN DE LOS ARCHIVOS (TODOS EN .PNG) ---
if __name__ == "__main__":
    lista_imagenes = [
        {
            "titulo": "Espectrograma\n(Referencia)", 
            "archivo": "Prueba_spectrogram_bw.png" # Cambiado a .png
        },
        {
            "titulo": "CRNN Model\n(Piano Roll)", 
            "archivo": "Prueba_pred_best_model_crnn_pianoroll.png"
        },
        {
            "titulo": "Transformer\n(23:18)", 
            "archivo": "Prueba_pred_best_model_transformer_20251125_2318_pianoroll.png"
        },
        {
            "titulo": "U-Net\n(00:14)", 
            "archivo": "Prueba_pred_best_unet_20251126_0014_pianoroll.png"
        },
        {
            "titulo": "Transformer\n(23:30)", 
            "archivo": "Prueba_pred_best_model_transformer_20251125_2330_pianoroll.png"
        }
    ]

    visualizar_comparacion(lista_imagenes)