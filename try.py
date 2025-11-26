import torch
print(f"¿GPU Disponible?: {torch.cuda.is_available()}")
print(f"Nombre: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")