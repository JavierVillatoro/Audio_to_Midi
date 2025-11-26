import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Asegúrate de que esta ruta sea correcta en tu PC
AUDIO_PATH = 'C:/Users/franc/Desktop/Tecnologia_del_habla/Audio_to_Midi/Prueba.wav'

def plot_spectrogram_color(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ Error: No se encuentra el archivo {audio_path}")
        return

    print(f"🔄 Generando espectrograma en color (Azul->Amarillo)...")
    
    # 1. Cargar Audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # 2. Calcular STFT (Transformada de Fourier)
    D = librosa.stft(y)
    # Convertir a dB para visualizar
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # =========================================================================
    # VISUALIZACIÓN
    # =========================================================================
    plt.figure(figsize=(14, 6))

    # --- CAMBIO AQUÍ ---
    # Usamos cmap='plasma'. 
    # Los valores bajos (dB negativos) se verán azules/morados.
    # Los valores altos (cercanos a 0 dB) se verán amarillos brillantes.
    librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr, cmap='plasma')

    # Estética
    # La barra de color ahora mostrará el gradiente azul-amarillo
    plt.colorbar(format='%+2.0f dB', label='Energía (dB)')
    
    plt.title(f'Espectrograma (Energía): {os.path.basename(audio_path)}')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    plt.grid(False) # Quitamos la rejilla para que se vea más limpio el color
    
    # Guardar con un nuevo nombre para no sobrescribir el de blanco y negro
    output_img = audio_path.replace(".wav", "_spectrogram_color.png")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"✅ Imagen guardada en: {output_img}")
    plt.show()

if __name__ == '__main__':
    plot_spectrogram_color(AUDIO_PATH)