import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import tftb.processing as tp


def spectrogram_tf(eeg_signal, fs=256):
    n = len(eeg_signal)
    t = np.arange(n) / fs

    window_length = int(fs/4)
    window = signal.windows.hann(window_length)

    f, t, Zxx = signal.stft(eeg_signal, fs=fs, window=window, nperseg=window_length, noverlap=window_length//2, nfft=n)
    tfr = np.abs(Zxx) ** 2

    return tfr, t, f

def generate_tf_rep(eeg_segment, method, fs=256):
    method_map = {
        # 'GK': gaussian_kernel_tf,
        # 'WV': wigner_ville_tf,
        'SPEC': spectrogram_tf,
        # 'MB': modified_b_tf,
        # 'SWV': smoothed_wv_tf,
        # 'SPEK': separable_kernel_tf,
    }

    if method not in method_map:
        raise ValueError(f"Invalid method '{method}'. Choose from {list(method_map.keys())}.")
    
    tfr, t, f = method_map[method](eeg_segment, fs=fs)
    return tfr, t, f

def plot_tf_rep(tfr, t, f, title='TF Representation', vmin=0, vmax=8):
    plt.figure(figsize=(8, 6))
    f_max_idx = np.searchsorted(f, 30.0)
    plt.pcolormesh(t, f[:f_max_idx], tfr[:f_max_idx], shading='gouraud', cmap='viridis', vmin=vmin, vmax=0.000000005)  
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (S)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.yticks([0, 10, 20, 30])
    plt.xticks(np.arange(0, t[-1], 0.2))
    plt.tight_layout()
    plt.show()

