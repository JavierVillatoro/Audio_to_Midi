import pretty_midi as librosa_midi
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# Asume que estas constantes están definidas globalmente o las pasas a la función
# SR (Sample Rate) y HOP_LENGTH (Hop Length)
SR = 44100  # Ejemplo común
HOP_LENGTH = 512 # Ejemplo común

def visualize_midi_piano_roll_improved(midi_path, sr=SR, hop_length=HOP_LENGTH):
    """
    Genera y guarda una imagen del piano roll de un archivo MIDI,
    con un esquema de color similar al ejemplo (azul oscuro/amarillo).

    Args:
        midi_path (str): Ruta al archivo MIDI.
        sr (int): Tasa de muestreo (sample rate).
        hop_length (int): Longitud de salto (hop length).
    """
    # 1. Definir la ruta de la imagen de salida
    MIDI_IMAGE_PATH = midi_path.replace(".mid", "_pianoroll.png")
    
    try:
        # 2. Cargar el archivo MIDI
        midi_data = librosa_midi.PrettyMIDI(midi_path)
        
        # 3. Calcular la tasa de muestreo de los frames
        fs = sr / hop_length
        
        # 4. Generar el piano roll
        # Se obtiene un array con las notas (filas) y el tiempo (columnas).
        # Los valores representan la velocidad (velocity) de la nota en ese momento.
        piano_roll = midi_data.get_piano_roll(fs=fs)
        
        # 5. Determinar el rango de notas activas
        # `np.sum(piano_roll, axis=1)` suma las velocidades de cada nota a lo largo del tiempo.
        # `active_notes` son los índices (MIDI) de las notas que se tocaron.
        active_notes = np.where(np.sum(piano_roll, axis=1) > 0)[0]
        
        if len(active_notes) == 0:
            print("Advertencia: El piano roll está vacío (no hay notas tocadas).")
            return
            
        # 6. Recortar el piano roll para mostrar solo las notas activas + un buffer
        min_note = max(0, active_notes.min() - 2)
        max_note = min(127, active_notes.max() + 2)
        piano_roll_visual = piano_roll[min_note:max_note, :]
        
        # 7. Visualización con Matplotlib
        plt.figure(figsize=(12, 6))
        
        # *** Clave para el estilo: Usar 'jet' o 'inferno' y ajustar, o crear un mapa de color personalizado.
        # 'jet' o 'inferno' con fondo oscuro funcionan bien para destacar el amarillo.
        # **Usaremos 'gist_stern' o 'hot' que da un buen contraste con fondo oscuro**
        
        plt.imshow(
            piano_roll_visual, 
            aspect='auto', 
            origin='lower', 
            cmap='gist_stern', # Prueba con 'gist_stern' o 'hot' para obtener el contraste
            interpolation='nearest' # 'nearest' para bloques definidos, otros para suavizado
        )
        
        # 8. Configurar los ejes Y (Notas Musicales)
        # Se genera una etiqueta para cada fila visible en el piano roll.
        y_ticks = np.arange(0, max_note - min_note)
        y_labels = [librosa.midi_to_note(n) for n in np.arange(min_note, max_note)]
        
        plt.yticks(y_ticks, y_labels)
        
        # 9. Configurar Títulos y Ejes X
        plt.xlabel('Tiempo (Frames)')
        plt.ylabel('Nota Musical')
        plt.title(f'Piano Roll del MIDI Generado: {os.path.basename(midi_path)}')
        
        # 10. Guardar la imagen
        plt.savefig(MIDI_IMAGE_PATH, dpi=300) # Usar un DPI alto para mejor calidad
        print(f"✅ Imagen guardada: {MIDI_IMAGE_PATH}")
        plt.close() 
        
    except FileNotFoundError:
        print(f"❌ Error: Archivo MIDI no encontrado en la ruta: {midi_path}")
    except Exception as e:
        print(f"❌ No se pudo generar la imagen debido a un error: {e}")

# Ejemplo de uso (asegúrate de que Prueba.mid existe y las librerías están instaladas)
visualize_midi_piano_roll_improved("Prueba_2_pred_best_unet_cleaned.mid")