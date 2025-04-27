import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import mode_dot
from src.tf_methods import generate_tf_rep

tl.set_backend('pytorch')

#this does the whole ed file so (3841, 241, 23) -> (3841, 241, 10)
def generate_segment_tensor(segment, method='SPEC', fs=256):

    # generate 3D tensor (T,F,K (number of channels))
    tf_slices = []
    for signal in segment:
        tfr, _, _ = generate_tf_rep(signal, method, fs)
        tf_slices.append(tfr)
    return np.stack(tf_slices, axis=-1)

def tensor_decomp(X, rank=10):
    factors = parafac(X, rank=rank)
    _, _, C = factors.factors

    P = torch.linalg.pinv(C)
    X_reduced = mode_dot(X, P, mode=2)

    return X_reduced

def process_all_segments(segments, method='SPEC', fs=256, rank=10):
    # converts all segments to reduced tensors of shape (T,F,R)
    #returns list of tensors and labels
    reduced_tensors = []

    for i, segment in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}")
        segment_tensor = generate_segment_tensor(segment, method, fs)
        tensor = generate_segment_tensor(segment, method, fs)
        tensor = torch.tensor(tensor, dtype=torch.float32)
        reduced_tensor = tensor_decomp(tensor, rank=rank)
        reduced_tensors.append(reduced_tensor.numpy())
    return reduced_tensors

def prepare_dataset(reduced_tensors, labels):
    #flatten tensor in image label pairs
    X = []
    Y = []
    for tensor, label in zip(reduced_tensors, labels):
        for r in range(tensor.shape[2]):
            X.append(tensor[:, :, r])
            Y.append(label)

    return np.array(X), np.array(Y)



