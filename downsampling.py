import os
import librosa
import soundfile as sf
import numpy as np

# --- ⚙️ Configuración ---
# Directorio donde se encuentran tus archivos de audio originales
INPUT_DIR = "data"
# Directorio donde se guardarán los archivos con downsampling
OUTPUT_DIR = "data_16000"
# Frecuencia de muestreo original (ya la mencionaste, 22050 Hz)
SR_ORIGINAL = 22050
# Nueva frecuencia de muestreo deseada (16000 Hz)
SR_TARGET = 16000
# Duración de las muestras (4 segundos)
DURACION = 4
# Número total de archivos que esperas (001 a 300)
NUM_ARCHIVOS = 300

## ----------------------------------------------------------------
## 📂 Crear el directorio de salida si no existe
## ----------------------------------------------------------------
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directorio creado: {OUTPUT_DIR}")

## ----------------------------------------------------------------
## 🔄 Procesamiento de archivos
## ----------------------------------------------------------------
print(f"Iniciando el downsampling de archivos de {INPUT_DIR} a {OUTPUT_DIR}...")

archivos_procesados = 0

for i in range(1, NUM_ARCHIVOS + 1):
    # Formato del nombre de archivo (voice_001.wav, voice_002.wav, etc.)
    nombre_archivo = f"voice_{i:03d}.wav"
    ruta_entrada = os.path.join(INPUT_DIR, nombre_archivo)
    ruta_salida = os.path.join(OUTPUT_DIR, nombre_archivo)

    # Solo intentamos procesar si el archivo de entrada existe
    if os.path.exists(ruta_entrada):
        try:
            # 1. Cargar el archivo de audio con librosa
            # NO especificamos sr=None para que librosa use la frecuencia de muestreo por defecto 
            # o la que le pasamos. Aquí vamos a leerla tal cual.
            # Sin embargo, si sabemos que la frecuencia es 22050, es más eficiente usar eso como target 
            # inicial y luego hacer el resample.
            # Para este caso, vamos a cargarla y resamplearla a la vez.

            # Cargar y realizar el resampleo (downsampling) directamente a 16000 Hz
            # librosa.load se encarga automáticamente de resamplear si el sr especificado es diferente
            # al del archivo.
            audio_data, sr = librosa.load(
                ruta_entrada, 
                sr=SR_TARGET, 
                mono=True  # Asumiendo que quieres que sea mono, es común en procesamiento de voz
            )
            
            # 2. Guardar el archivo downsampleado
            # Usamos soundfile (sf) para guardar, que es la recomendación de librosa para WAV.
            sf.write(ruta_salida, audio_data, SR_TARGET)
            
            archivos_procesados += 1
            # print(f"  -> Procesado: {nombre_archivo}")
            
        except Exception as e:
            print(f"⚠️ Error al procesar {nombre_archivo}: {e}")
            
    # else:
    #     print(f"  -> Omitido: {nombre_archivo} (No encontrado)")


print("\n--- ✅ Proceso Finalizado ---")
print(f"Total de archivos procesados y guardados en {OUTPUT_DIR}: {archivos_procesados} de {NUM_ARCHIVOS} esperados.")