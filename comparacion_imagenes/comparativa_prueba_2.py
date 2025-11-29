import matplotlib.pyplot as plt
import numpy as np
import cv2

def create_professional_comparison_2row(path_gt, path_prediction, output_filename="Paper_Comparison_Final.png"):
    
    # --- 1. CARGA Y LIMPIEZA (Recorte de texto antiguo) ---
    # Altura a recortar desde arriba para quitar metadatos sucios
    CROP_HEIGHT = 40 
    
    img_gt_raw = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
    img_pred_raw = cv2.imread(path_prediction, cv2.IMREAD_GRAYSCALE)

    if img_gt_raw is None or img_pred_raw is None:
        print("Error: Revisa las rutas.")
        return

    # Aplicar el recorte de limpieza
    img_gt = img_gt_raw[CROP_HEIGHT:, :]
    img_pred = img_pred_raw[CROP_HEIGHT:, :]

    # --- 2. PROCESAMIENTO ---
    # Asegurar mismas dimensiones
    h, w = img_gt.shape
    img_pred = cv2.resize(img_pred, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binarizar con umbral alto para asegurar negro puro
    _, bin_gt = cv2.threshold(img_gt, 200, 255, cv2.THRESH_BINARY)
    _, bin_pred = cv2.threshold(img_pred, 200, 255, cv2.THRESH_BINARY)

    # Generar imagen comparativa (Verde vs Magenta)
    # Blanco = Acierto (Coincidencia)
    comp_img = np.zeros((h, w, 3), dtype=np.uint8)
    comp_img[:, :, 1] = bin_gt        # Canal Verde = Original (lo que debió estar)
    comp_img[:, :, 0] = bin_pred      # Canal Rojo \
    comp_img[:, :, 2] = bin_pred      # Canal Azul / = Magenta (lo que se predijo)

    # --- 3. GRAFICADO PROFESIONAL "APLANADO" ---
    
    # Configuración de fuentes académicas (opcional, limpia el estilo)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # figsize=(20, 8) crea una figura muy ancha (aplanada) y de altura media para 2 filas.
    # dpi=300 es el estándar para publicaciones científicas.
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), dpi=300)

    # --- Fila 1: Original (Referencia) ---
    # vmin/vmax aseguran contraste máximo
    axes[0].imshow(bin_gt, cmap='gray', aspect='auto', interpolation='nearest', vmin=0, vmax=255)
    axes[0].set_title("(a) Ground Truth (Midi Original)", fontsize=14, fontweight='bold', pad=15)
    axes[0].axis('off')

    # --- Fila 2: Comparativa de Errores ---
    axes[1].imshow(comp_img, aspect='auto', interpolation='nearest')
    # Título descriptivo y profesional
    axes[1].set_title("(b) Análisis de Errores: Notas Perdidas (Verde) vs. Notas Inventadas (Magenta)", fontsize=14, fontweight='bold', pad=15)
    axes[1].axis('off')

    # --- AJUSTE FINAL DE MÁRGENES ---
    plt.tight_layout()
    # Ajuste fino: top=0.92 asegura que el título superior no roce el borde.
    # hspace=0.3 da un respiro elegante entre las dos gráficas.
    plt.subplots_adjust(top=0.92, hspace=0.3)

    plt.savefig(output_filename, bbox_inches='tight')
    # plt.show() # Comentado para ejecución rápida. Descomentar si quieres verla al momento.
    print(f"Figura profesional guardada en alta resolución: {output_filename}")

# --- RUTAS DE ENTRADA ---
base_path = "C:/Users/franc/Desktop/Tecnologia_del_habla/Audio_to_Midi/comparacion_imagenes/"
f_original = base_path + "Prueba_2_pianoroll.png"
# Usamos tu mejor resultado (Unet Cleaned) como la predicción a comparar
f_pred_final = base_path + "Prueba_2_pred_best_unet_cleaned_pianoroll_10_10.png"

create_professional_comparison_2row(f_original, f_pred_final)