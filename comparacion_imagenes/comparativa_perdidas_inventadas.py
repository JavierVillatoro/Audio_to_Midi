import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2  # Necesitarás opencv-python

def compare_piano_rolls(path_ground_truth, path_prediction, output_filename="comparacion_final.png"):
    """
    Carga dos imágenes de Piano Roll, las binariza y las superpone.
    Ground Truth = Verde
    Prediction = Rojo/Magenta
    Coincidencia = Mezcla (Amarillento/Blanco)
    """
    
    # 1. Cargar imágenes
    # Asegúrate de que las rutas sean correctas
    img_gt = cv2.imread(path_ground_truth, cv2.IMREAD_GRAYSCALE)
    img_pred = cv2.imread(path_prediction, cv2.IMREAD_GRAYSCALE)

    if img_gt is None or img_pred is None:
        print("Error: No se pudieron cargar las imágenes. Revisa las rutas.")
        return

    # 2. Asegurar que tengan el mismo tamaño
    # Si la predicción tiene un tamaño ligeramente distinto, la redimensionamos al GT
    if img_gt.shape != img_pred.shape:
        print(f"Redimensionando predicción... {img_pred.shape} -> {img_gt.shape}")
        img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))

    # 3. Binarizar (Convertir a 0 y 1 puro)
    # Asumimos que las notas son claras (blancas/amarillas) y el fondo oscuro
    _, binary_gt = cv2.threshold(img_gt, 127, 255, cv2.THRESH_BINARY)
    _, binary_pred = cv2.threshold(img_pred, 127, 255, cv2.THRESH_BINARY)

    # 4. Crear imagen RGB para la superposición
    # Altura x Anchura x 3 canales (R, G, B)
    h, w = binary_gt.shape
    comparison_img = np.zeros((h, w, 3), dtype=np.uint8)

    # --- ASIGNACIÓN DE COLORES ---
    
    # CANAL VERDE (G): Asignamos el Ground Truth
    # Si hay nota en el original, se verá verde.
    comparison_img[:, :, 1] = binary_gt 

    # CANAL ROJO (R) y AZUL (B): Asignamos la Predicción (formará color Magenta/Rosa)
    # Esto ayuda a distinguir mejor sobre fondo negro.
    # Si quieres solo Rojo, comenta la línea del canal 2.
    comparison_img[:, :, 0] = binary_pred 
    comparison_img[:, :, 2] = binary_pred

    # --- LEYENDA DE COLORES ---
    # Verde puro: Nota que existía pero NO fue predicha (False Negative / Nota Perdida)
    # Magenta puro: Nota que NO existía pero FUE predicha (False Positive / Alucinación)
    # Blanco/Gris claro (Mezcla): Acierto (True Positive)

    # 5. Visualizar
    plt.figure(figsize=(12, 6))
    plt.title("Comparativa: Verde=Original(Perdidas) | Magenta=Predicción(Inventadas) | Blanco=Aciertos")
    plt.imshow(comparison_img)
    plt.axis('off') # Ocultar ejes de píxeles
    
    # Guardar resultado
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Imagen guardada como: {output_filename}")

# --- USO DEL SCRIPT ---
# Reemplaza estos nombres con los archivos que tengas en tu carpeta
file_gt = "C:/Users/franc/Desktop/Tecnologia_del_habla/Audio_to_Midi/comparacion_imagenes/Prueba_2_pianoroll.png" # <--- REEMPLAZA ESTO
file_pred = "C:/Users/franc/Desktop/Tecnologia_del_habla/Audio_to_Midi/comparacion_imagenes/Prueba_2_pred_best_unet_cleaned_pianoroll_10_10.png" # <--- REEMPLAZA ESTO

# Asegúrate de usar barras normales (/) o doble barra invertida (\\) en Windows.

compare_piano_rolls(file_gt, file_pred)