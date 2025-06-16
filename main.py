import numpy as np
import re
import mne
import os  
import torch      
from src.preprocess import load_file, filter_eeg, segment_eeg, label_segments
from src.tf_methods import generate_tf_rep, plot_tf_rep
from src.tensor_reduction import generate_segment_tensor, tensor_decomp, process_all_segments

# === Configuration ===
FILE_PATH = 'chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_03.edf' #download from kaggle
SUMMARY_PATH = 'chb-mit-scalp-eeg-database-1.0.0/chb01/chb01-summary.txt' #download from kaggle
SEGMENT_LENGTH = 30  # seconds
FS = 256
METHOD = 'SPEC'  # Options: 'GK', 'SPEC', etc.

SEGMENT_IDX = 50        # Which segment to plot
CHANNEL_IDX = 0         # Which channel in that segment
DURATION_SEC = 30     # How much of the segment to visualize (in seconds)


# === Main Process ===
if __name__ == "__main__":
    print(f"Loading file: {FILE_PATH}")
    raw = load_file(FILE_PATH)
    raw = filter_eeg(raw)
    #print("hi", raw.annotations)
    
    print("Segmenting signal...")
    segments = segment_eeg(raw, segment_length=SEGMENT_LENGTH, fs=FS)
    labels = label_segments(segments, FILE_PATH, SUMMARY_PATH, segment_length=SEGMENT_LENGTH, stepsize=15, fs=FS)

    print(f"Total segments: {len(segments)}")
    print(f"Selected Segment: {SEGMENT_IDX}, Label: {'Seizure' if labels[SEGMENT_IDX] else 'Non-Seizure'}")

    print(f"Testing the tensor dims")
    # Generate tensor representation for the selected segment
    segment_tensor = generate_segment_tensor(segments[SEGMENT_IDX], method=METHOD, fs=FS)
    print(f"original tensor shape {segment_tensor.shape}")

    segment_tensor = torch.tensor(segment_tensor, dtype=torch.float32)
    reduced_tensor = tensor_decomp(segment_tensor, rank=10)
    print(f"Reduced tensor shape {reduced_tensor.shape}")

    # Extract the desired EEG channel and trim
    eeg_signal = segments[SEGMENT_IDX][CHANNEL_IDX]
    eeg_signal = eeg_signal[:DURATION_SEC * FS]

    print(f"Generating TF representation using method: {METHOD}")
    tfr, t, f = generate_tf_rep(eeg_signal, method=METHOD, fs=FS)

    title = f"{METHOD} TF Rep | Segment {SEGMENT_IDX} | Ch {CHANNEL_IDX} | {DURATION_SEC}s"
    plot_tf_rep(tfr, t, f, title=title)
    
    # Show reduced tensor slice for the selected channel
    component_idx = 0  # First component, change if needed
    reduced_slice = reduced_tensor[:, :, component_idx].numpy()
    title_reduced = f"Decomposed Component {component_idx} | Segment {SEGMENT_IDX} | Ch {CHANNEL_IDX}"
    plot_tf_rep(reduced_slice, t, f, title=title_reduced)

    #print one of the actual matrices from the reduced tensor
    print(reduced_tensor[:, :, component_idx].numpy()) 
    #print its shape too
    print(f"Shape of reduced slice: {reduced_slice.shape}")

    #print each image shape in the reduced tensor
    for i in range(reduced_tensor.shape[2]):
        print(f"Image {i} shape: {reduced_tensor[:, :, i].shape}")



    
