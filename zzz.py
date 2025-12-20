import numpy as np
import sys

def count_classes(file_path):
    print(f"--- Analizando archivo: {file_path} ---")
    
    try:
        # CORRECCIÓN: Se añade allow_pickle=True para permitir cargar el array
        Y_full = np.load(file_path, allow_pickle=True)
        print(f"Forma original (Shape): {Y_full.shape}")
        
        # 2. Aplanar los datos (300 * 666)
        Y_flat = Y_full.flatten()
        total_samples = Y_flat.size
        print(f"Total de muestras (frames): {total_samples}")
        print("-" * 30)

        # 3. Contar clases únicas
        classes, counts = np.unique(Y_flat, return_counts=True)

        # 4. Mostrar resultados
        print(f"{'Clase':<10} | {'Cantidad':<15} | {'Porcentaje':<10}")
        print("-" * 40)
        
        for cls, count in zip(classes, counts):
            percentage = (count / total_samples) * 100
            print(f"{int(cls):<10} | {count:<15} | {percentage:.2f}%")
            
        print("-" * 40)

        # 5. Advertencia de desbalance
        for cls, count in zip(classes, counts):
            percentage = (count / total_samples) * 100
            if percentage < 10:
                print(f"⚠️  ADVERTENCIA: La clase {int(cls)} está sub-representada (<10%).")
                
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'. Verifica la ruta.")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    count_classes("dataset_Y.npy")