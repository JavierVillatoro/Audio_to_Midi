# Audio to MIDI: Voice-Controlled Note Classification

This project explores the conversion of raw audio (voice) into MIDI files using Deep Learning. The goal is to classify audio segments into four specific outputs: Silence, C4, E4, and G4.

The project compares multiple architectures to find the best performance on a "Small Data" custom dataset, using hyperparameter optimization and class balancing techniques.

## Dataset

The dataset was custom-recorded to test model robustness across different recording qualities. It consists of 300 samples in total, split into three distinct categories:

- Studio Quality (100 samples): Recorded with a condenser microphone.
- Noisy Environment (100 samples): Recorded with a condenser microphone mixed with background ambience recorded via a Zoom H1n Essential.
- Low Quality (100 samples): Recorded with a standard laptop integrated microphone.

## Methodology

### Preprocessing
- Input: Audio files are processed using the Constant-Q Transform (CQT).
- Windowing: The audio is divided into windows (e.g., 4 seconds) for processing.
- Balancing: Weighted Cross-Entropy was implemented to solve class imbalance (specifically the bias toward class 0/Silence). Weights were calculated based on the frequency of each class in the dataset to penalize errors on minority classes more heavily.

### Models Evaluated
Several architectures were tested to compare performance on this specific task:

- CRNN (Convolutional Recurrent Neural Network): Used as the industry standard baseline.
- AudioTransNet: A Transformer-based architecture.
- AudioUNet: A U-Net based architecture commonly used for segmentation.
- U-Net + BiLSTM: A hybrid approach combining spatial feature extraction with temporal sequence modeling.

### Optimization
- Optuna: Used for automated hyperparameter tuning (Learning Rate, Batch Size, Hidden Dimensions).
- Training Strategy: Implemented Early Stopping and ReduceLROnPlateau to prevent overfitting and optimize convergence.

## Results

- Best Model: Currently, the U-Net architecture provides the most accurate results for this specific dataset and voice profile.
- Analysis: Performance was evaluated using Confusion Matrices and Loss Curves. The Transformer model showed higher overfitting risks compared to the U-Net on this small dataset, although it occasionally captured specific pitches well.

## Future Work

- Onset and Frames: Improve note segmentation by separating the task into Onset detection (when a note starts) and Frame classification (note duration), similar to piano transcription models.
- New Architectures: Test Harmonic Res U-Net to better capture harmonic structures.
- Real-time: Optimize the best-performing model for low-latency inference.
- Inference Visualization: Improve the visual comparison between ground truth MIDI and generated MIDI.