# Seizure Classification using Vision Transformers and Tensor Decomposition

This project presents an end-to-end framework for epileptic seizure classification by transforming multi-channel EEG signals into feature-rich images and classifying them using a Vision Transformer (ViT). By leveraging Canonical Polyadic (CP) Tensor Decomposition, we transform spatio-temporal-spectral EEG data into a compact 2D representation, enabling the application of state-of-the-art vision models to a traditionally time-series based problem.

The project is inspired by the work in the following research papers:
- [Vision Transformer for End-to-End Seizure Detection and Prediction](https://ieeexplore.ieee.org/document/9734546)
- [Tensor-Based Epileptic Seizure Prediction](https://ieeexplore.ieee.org/document/8918754)

## 1. Project Overview

Epilepsy is a neurological disorder where recurrent, unpredictable seizures pose significant risks to patient safety and quality of life. An automated system capable of accurately predicting seizures could enable timely intervention. This project tackles the seizure prediction challenge by framing it as an image classification task.

The core workflow is as follows:
1.  **EEG Signal Preprocessing**: Raw multi-channel EEG data from the CHB-MIT Scalp EEG Database is loaded, filtered, and segmented into labeled windows.
2.  **Time-Frequency Analysis**: The preprocessed signals are transformed into the time-frequency domain using the Short-Time Fourier Transform (STFT) to generate spectrograms for each EEG channel.
3.  **Tensor Construction and Decomposition**: The 2D spectrograms are stacked to form a 3rd-order tensor (`channel × frequency × time`). Canonical Polyadic (CP) Tensor Decomposition is applied to reduce dimensionality and extract features across the three modes.
4.  **Vision Transformer Classification**: The decomposed, low-rank feature maps are treated as images and fed into a Vision Transformer (ViT) model for binary classification into **ictal** (seizure) or **inter-ictal** (non-seizure) states.

This approach combines the signal processing strength of tensor analysis with the powerful feature extraction capabilities of the Transformer architecture, which has demonstrated superior performance in capturing global dependencies in image data as compared to more traditional Convolutional Neural Networks (CNNs).

## 2. Technical Methodology

The pipeline is implemented as a series of modular steps, from data loading to feature extraction and classification.

### 2.1. Data Source and Preparation
The project uses the **CHB-MIT Scalp EEG Database**, a public dataset containing continuous EEG recordings from 22 unique subjects, with each patient having 23 channels.
-   **Data Loading**: EEG recordings are loaded from `.edf` files using the MNE-Python library. Seizure annotations are parsed from the accompanying `summary.txt` files.
-   **Filtering**: A band-pass filter (0.5-40 Hz) is applied to each channel to remove DC offset and noise. 
-   **Segmentation**: The continuous EEG signal is divided into overlapping 30-second segments (15 second overlap between segments).
-   **Labeling**: Each segment is labeled based on being inside of a seizure event.

### 2.2. Time-Frequency Representation
To convert the 1D time-series EEG data into a 2D format suitable for image-based models, we generate time-frequency (TF) representations using the **Short-Time Fourier Transform (STFT)**.

For an EEG segment `x(t)` from a single channel, the STFT is computed, and its squared magnitude, `|STFT(τ, ω)|²`, gives the spectrogram. This gives a frequency/amplitude plot of the signal over time.

### 2.3. Tensor-Based Feature Extraction
A key aspect of this project is the use of tensor decomposition to condense information in all EEG channels and extract spectral-temporal features.

-   **Tensor Construction**: The spectrograms from all `N` channels are stacked to form a 3rd-order tensor `X` of size `N × F × T`, where `F` is the number of frequency bins and `T` is the number of time steps.
-   **Canonical Polyadic (CP) Decomposition**: The tensor `X` is decomposed into a sum of `R` rank-one tensors. For a 3rd-order tensor, this is:
    ```
    X ≈ Σ_{r=1 to R} λ_r * a_r ∘ b_r ∘ c_r
    ```
    where `a_r`, `b_r`, and `c_r` are vectors representing the salient features along the **channel (spatial)**, **frequency (spectral)**, and **time (temporal)** modes, respectively. The outer product `∘` creates the rank-one tensor.
-   **Feature Image Generation**: The factor matrices `B` (from `b_r` vectors) and `C` (from `c_r` vectors) capture the dominant frequency and temporal patterns across all channels. Their outer product, `B @ C.T`, creates a low-rank `F × T` matrix that serves as a compact, feature-rich "image." This image distills the essential information from the original multi-channel segment and serves as the input to the ViT. It is quite similar to SVD with 2D matrices.

### 2.4. Classification with Vision Transformer (ViT)
The ViT architecture is used to classify the generated feature images.
-   **Patching and Embedding**: The input image is split into a sequence of fixed-size patches (e.g., 16x16 pixels). Each patch is flattened and linearly projected into an embedding vector. A learnable `[CLS]` token is prepended to the sequence, and positional embeddings are added to retain spatial information.
-   **Transformer Encoder**: The sequence of embeddings is processed by a standard Transformer encoder. This consists of alternating layers of Multi-Head Self-Attention (MHSA) and MLP blocks. The self-attention mechanism allows the model to weigh the importance of all other patches when processing a given patch, enabling it to learn global relationships across the entire time-frequency plane.
-   **Classification Head**: The final output embedding corresponding to the `[CLS]` token is passed to a final MLP head for classification into "pre-ictal" or "inter-ictal" classes.

## 3. Getting Started

### 3.1. Prerequisites
- Python 3.8+
- PyTorch
- NumPy, SciPy, MNE-Python
- TensorLy
- A local copy of the **CHB-MIT Scalp EEG Database**.

### 3.2. Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/v1ppu/Seizure-Prediction-using-ViT.git
    cd Seizure-Prediction-using-ViT
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 3.3. Usage

The end-to-end pipeline is executed through a series of python commands. The typical workflow is as follows:

1.  **Create Training and Testing Datasets**
    This script partitions the processed data into training and testing sets. We can change the edf_files array to whatever specified files from the CHB-MIT dataset.
    ```bash
    python create_train_test.py
    ```

2.  **Train the Vision Transformer Model**
    This script executes the training loop using the training dataset.
    ```bash
    python train.py
    ```

3.  **Evaluate Model Performance**
    This script evaluates the trained model against the test set.
    ```bash
    python evaluate.py
    ```

## 4. Project Structure
```
.
├── data/                    # Placeholder for train and test data
├── src/
│   ├── dataset.py           # PyTorch dataset class with loading, resizing, tensor format
│   ├── model.py             # Custom ViT architecture for EEG classification
│   ├── preprocess.py        # Functions for loading/filtering/segmenting/labeling data
│   ├── tensor_reduction.py  # Functions for tensor decomp. on spectrogram images 
│   ├── tf_methods.py        # Functions for generating spectrograms via STFT
│   └── train.py             # Script for ViT training loop
├── create_train_test.py     # Script to create training and testing sets
├── data_prep.py             # Functions to create data using prev. src files
├── evaluate.py              # Script to evaluate the model on test data
└── requirements.txt         # Python dependencies
```

## 5. Evaluation and Future Work

### 5.1. Evaluation Metrics
The model's performance will be evaluated using standard metrics for seizure prediction:
-   **Test Accuracy**: Overall accuracy percentage across all test samples.
-   **Average Loss**: Mean loss value across all test batches
-   **Classification Report**: Precision, Recall, F1-score for each class ("No Seizure", "Seizure").


### 5.2. Possible Future Work
- **Model Training and Hyperparameter Tuning**: Train the ViT model and systematically tune hyperparameters (e.g., learning rate, decomposition rank `R`, patch size, number of Transformer layers).
- **Experimentation**:
    -   Explore alternative TF representations (e.g., Continuous Wavelet Transform).
    -   Investigate other tensor decomposition methods like Tucker decomposition for different feature interactions.
- **Real-time Implementation**: Adapt the framework for a simulated real-time prediction scenario, focusing on computational efficiency.
