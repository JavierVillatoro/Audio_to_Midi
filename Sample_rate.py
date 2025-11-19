import librosa
import os

def obtener_frecuencia_muestreo(ruta_archivo):
    """
    Carga un archivo de audio con librosa y devuelve su frecuencia de muestreo (sampling rate).
    
    Args:
        ruta_archivo (str): La ruta completa al archivo de audio.
        
    Returns:
        int: La frecuencia de muestreo del audio, o None si hay un error.
    """
    if not os.path.exists(ruta_archivo):
        print(f"⚠️ Error: El archivo no se encontró en la ruta: {ruta_archivo}")
        return None
        
    try:
        # 1. Cargar el audio con librosa.load
        # Usamos sr=None para asegurar que librosa NO resamplee y devuelva la frecuencia original.
        # No nos importa el contenido del audio (audio_data), solo necesitamos el 'sr'.
        audio_data, sr = librosa.load(ruta_archivo, sr=None)
        
        return sr
        
    except Exception as e:
        print(f"⚠️ Error al intentar cargar el archivo: {e}")
        return None

# --- 💡 Ejemplo de Uso ---
# IMPORTANTE: Reemplaza 'ruta/a/tu/audio.wav' con la ruta real de uno de tus archivos.
ruta_ejemplo = "data/voice_001.wav" 

frecuencia = obtener_frecuencia_muestreo(ruta_ejemplo)

if frecuencia is not None:
    print(f"\n--- Resultado ---")
    print(f"El archivo: '{os.path.basename(ruta_ejemplo)}'")
    print(f"Tiene una **Frecuencia de Muestreo (Sampling Rate)** de: **{frecuencia} Hz**")
    print(f"({frecuencia / 1000} kHz)")