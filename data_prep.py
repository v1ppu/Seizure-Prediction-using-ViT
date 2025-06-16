import os
import torch
import time
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from src.preprocess import load_file, filter_eeg, segment_eeg, label_segments
from src.tensor_reduction import generate_segment_tensor, tensor_decomp


def preprocess_and_save(edf_path, summary_path, output_dir, rank=10, resize_to=(224,224), method='SPEC'):
    print(f"Processing {edf_path}")
    start_time = time.time()


    raw = load_file(edf_path)
    raw = filter_eeg(raw)
    segments = segment_eeg(raw)
    labels = label_segments(segments, edf_path, summary_path)

    X = []
    Y = []

    for i, segment in enumerate(tqdm(segments, desc="Segments")):
        tensor = generate_segment_tensor(segment, method=method)
        tensor = torch.tensor(tensor, dtype=torch.float32)
        reduced = tensor_decomp(tensor, rank=rank)

        for r in range(reduced.shape[2]):
            image = reduced[:, :, r].unsqueeze(0).unsqueeze(0)
            image = F.interpolate(image, size=resize_to, mode='bilinear', align_corners=False)
            X.append(image.squeeze(0).numpy())
            Y.append(labels[i])
    
    X = np.stack(X)
    Y = np.array(Y)

    os.makedirs(output_dir, exist_ok=True)
    X_path = os.path.join(output_dir, 'X.npy')
    Y_path = os.path.join(output_dir, 'Y.npy')

    if(os.path.exists(X_path)):
        X_existing = np.load(X_path)
        Y_existing = np.load(Y_path)
        X = np.concatenate((X_existing, X), axis=0)
        Y = np.concatenate((Y_existing, Y), axis=0)

    np.save(X_path, X)
    np.save(Y_path, Y)
    print(f"Saved {len(X)} images to {output_dir}")
    print(f"Finished {edf_path} in {time.time() - start_time:.2f} seconds")
