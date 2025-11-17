import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils.dataset_utils import load_full_dataset, split_dataset, pad_sequences, pad_labels

# -------------------------
# Cargar dataset
# -------------------------
X, Y = load_full_dataset("dataset_X.npy", "dataset_Y.npy")
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

# Padding para batches
X_train_padded = pad_sequences(X_train)
Y_train_padded = pad_labels(Y_train)

X_val_padded = pad_sequences(X_val, max_len=X_train_padded.shape[1])
Y_val_padded = pad_labels(Y_val, max_len=Y_train_padded.shape[1])

# Convertir a tensores
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_padded, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_padded, dtype=torch.long)

# -------------------------
# Dataset y DataLoader
# -------------------------
class AudioMIDIDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = AudioMIDIDataset(X_train_tensor, Y_train_tensor)
val_dataset   = AudioMIDIDataset(X_val_tensor, Y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4)

# -------------------------
# Modelo CRNN simple
# -------------------------
class CRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super(CRNN, self).__init__()
        # CNN para extraer features locales
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # LSTM bidireccional para secuencia temporal
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        # Capa final por frame
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        # x: (batch, frames, bins)
        x = x.permute(0, 2, 1)  # (batch, bins, frames)
        x = self.conv(x)        # (batch, channels, frames)
        x = x.permute(0, 2, 1)  # (batch, frames, channels)
        out, _ = self.lstm(x)   # (batch, frames, hidden*2)
        out = self.fc(out)       # (batch, frames, output_dim)
        return out

# -------------------------
# Inicializar modelo
# -------------------------
input_dim = X_train_tensor.shape[2]  # número de bins CQT
model = CRNN(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Loop de entrenamiento simple
# -------------------------
n_epochs = 10

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)  # (batch, frames, output_dim)
        # CrossEntropyLoss espera (batch*frames, classes)
        loss = criterion(outputs.view(-1, 4), Y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, 4), Y_batch.view(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

print("Entrenamiento finalizado")
