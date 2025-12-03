import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_pipeline_viz():
    # Configuración del lienzo (3 paneles)
    fig = plt.figure(figsize=(24, 10))
    fig.patch.set_facecolor('#f4f4f4')
    
    ax_pipe = plt.subplot2grid((2, 3), (0, 0), colspan=3) # Panel Superior (Pipeline)
    ax_arch = plt.subplot2grid((2, 3), (1, 0), colspan=2) # Panel Inferior Izq (Arquitectura)
    ax_se   = plt.subplot2grid((2, 3), (1, 2))            # Panel Inferior Der (SE-Block)

    for ax in [ax_pipe, ax_arch, ax_se]:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

    # ==========================================
    # 1. PANEL SUPERIOR: DATA & TRAINING PIPELINE
    # ==========================================
    ax_pipe.set_title("1. Training Pipeline & Data Flow", fontsize=16, fontweight='bold', pad=20)
    
    # Estilos de cajas
    def add_box(ax, x, y, w, h, text, color='#dddddd', edge='#333333', shape='rect'):
        if shape == 'rect':
            box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1", 
                                         linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')

    def add_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))

    # Nodos
    add_box(ax_pipe, 2, 70, 10, 10, "Dataset X\n(CQT Audio)", color='#bbdefb')
    add_box(ax_pipe, 2, 50, 10, 10, "Dataset Y\n(MIDI Labels)\n[Sil, C3, E3, G3]", color='#ffccbc')
    
    add_box(ax_pipe, 18, 60, 12, 10, "Preprocessing\nSplit & Pad\nTo Tensor", color='#e0e0e0')
    
    add_box(ax_pipe, 35, 60, 12, 10, "Class Weights\n(Soft Power 0.75)", color='#fff9c4')
    
    add_box(ax_pipe, 52, 55, 15, 20, "OPTUNA LOOP\n(5 Trials)\n\nVars: LR, Batch,\nLSTM Hidden,\nGamma, Dropout", color='#d1c4e9')
    
    add_box(ax_pipe, 72, 55, 12, 20, "Training\n(Adam Optimizer)", color='#c8e6c9')
    
    add_box(ax_pipe, 88, 70, 10, 8, "Loss Function\n(FOCAL LOSS)", color='#ffcdd2', edge='red')
    add_box(ax_pipe, 88, 50, 10, 8, "Final Model\n(.pth)", color='#b2dfdb')

    # Flechas Pipeline
    add_arrow(ax_pipe, 13, 75, 17, 70) # X -> Prep
    add_arrow(ax_pipe, 13, 55, 17, 65) # Y -> Prep
    add_arrow(ax_pipe, 31, 66, 34, 66) # Prep -> Weights
    add_arrow(ax_pipe, 48, 66, 51, 66) # Weights -> Optuna
    add_arrow(ax_pipe, 68, 66, 71, 66) # Optuna -> Train
    add_arrow(ax_pipe, 85, 66, 88, 74) # Train -> Loss
    add_arrow(ax_pipe, 85, 64, 88, 54) # Train -> Model

    # ==========================================
    # 2. PANEL: ARQUITECTURA (U-NET + BiLSTM)
    # ==========================================
    ax_arch.set_title("2. Architecture: SE-UNet + BiLSTM", fontsize=14, fontweight='bold')

    # Coordenadas relativas para la U
    layer_pos = {
        'in': (10, 80), 'd1': (25, 65), 'd2': (40, 50), 'd3': (55, 35), # Encoder
        'u1': (70, 50), 'u2': (85, 65), 'u3': (100, 80), # Decoder
        'lstm': (115, 80), 'out': (130, 80) # Head
    }
    
    # Ajuste de escala para este panel
    ax_arch.set_xlim(0, 140)

    # Dibujar Bloques
    # Encoder
    add_box(ax_arch, 5, 80, 10, 8, "Input\n(CQT)", color='#eeeeee')
    add_box(ax_arch, 20, 65, 10, 8, "Down 1\n(SE-Block)", color='#bbdefb')
    add_box(ax_arch, 35, 50, 10, 8, "Down 2\n(SE-Block)", color='#90caf9')
    add_box(ax_arch, 50, 35, 10, 8, "Down 3\n(SE-Block)", color='#64b5f6') # Bottleneck

    # Decoder
    add_box(ax_arch, 65, 50, 10, 8, "Up 1\n(Concat)", color='#ffcc80')
    add_box(ax_arch, 80, 65, 10, 8, "Up 2\n(Concat)", color='#ffb74d')
    add_box(ax_arch, 95, 80, 10, 8, "Up 3\n(Concat)", color='#ffa726')

    # Head
    add_box(ax_arch, 110, 80, 12, 8, "BiLSTM\n(Contexto)", color='#ce93d8')
    add_box(ax_arch, 128, 80, 8, 8, "FC\n(4 Clases)", color='#ef9a9a')

    # Flechas Flujo Principal
    ax_arch.plot([10, 10, 25], [79, 73, 73], color='#555', lw=2) # In -> D1
    ax_arch.plot([25, 25, 40], [64, 58, 58], color='#555', lw=2) # D1 -> D2
    ax_arch.plot([40, 40, 55], [49, 43, 43], color='#555', lw=2) # D2 -> D3 (Bottom)
    
    ax_arch.plot([61, 70], [39, 50], color='#555', lw=2) # D3 -> U1
    ax_arch.plot([75, 85], [59, 65], color='#555', lw=2) # U1 -> U2
    ax_arch.plot([90, 100], [74, 80], color='#555', lw=2) # U2 -> U3
    
    add_arrow(ax_arch, 106, 84, 109, 84) # U3 -> LSTM
    add_arrow(ax_arch, 123, 84, 127, 84) # LSTM -> FC

    # Skip Connections (Las flechas horizontales de la U-Net)
    ax_arch.annotate("Skip", xy=(70, 54), xytext=(46, 54), arrowprops=dict(arrowstyle="->", color="green", ls="--"))
    ax_arch.annotate("Skip", xy=(85, 69), xytext=(31, 69), arrowprops=dict(arrowstyle="->", color="green", ls="--"))
    ax_arch.annotate("Skip", xy=(100, 84), xytext=(16, 84), arrowprops=dict(arrowstyle="->", color="green", ls="--"))

    # ==========================================
    # 3. PANEL: SE-BLOCK DETAIL
    # ==========================================
    ax_se.set_title("3. Detail: SE-Block Logic", fontsize=14, fontweight='bold')
    
    # Input tensor
    add_box(ax_se, 40, 85, 20, 10, "Input Feature\n(C x L)", color='#e0e0e0')
    
    # Squeeze Path (Left)
    ax_se.text(25, 75, "Squeeze", ha='center', color='blue', fontweight='bold')
    add_box(ax_se, 10, 60, 20, 10, "Global Avg Pool\n(1x1)", color='#bbdefb')
    add_box(ax_se, 10, 45, 20, 10, "FC Reduce\n(ReLU)", color='#90caf9')
    add_box(ax_se, 10, 30, 20, 10, "FC Expand\n(Sigmoid)", color='#64b5f6')
    
    # Original Path (Right)
    ax_se.plot([60, 60], [84, 25], color='#999', lw=2, linestyle='--')
    ax_se.text(75, 55, "Identity Pass", color='#999')

    # Scale Operation (Multiply)
    add_box(ax_se, 40, 10, 20, 10, "Scale\n(X * Weights)", color='#ffcc80')

    # Flechas SE
    add_arrow(ax_se, 50, 84, 20, 71) # Input -> Squeeze
    add_arrow(ax_se, 20, 59, 20, 56)
    add_arrow(ax_se, 20, 44, 20, 41)
    add_arrow(ax_se, 20, 29, 40, 15) # Sigmoid -> Scale
    add_arrow(ax_se, 60, 25, 60, 15) # Identity -> Scale

    # Add text explaining the MIDI Classes
    plt.figtext(0.5, 0.02, 
                "Classes: 0=Silencio, 1=Do3 (C3), 2=Mi3 (E3), 3=Sol3 (G3) | Loss: Focal Loss (Gamma Tuned) | Context: BiLSTM", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.tight_layout()
    plt.show()

# Ejecutar la visualización
if __name__ == "__main__":
    print("Generando diagrama de arquitectura...")
    draw_pipeline_viz()