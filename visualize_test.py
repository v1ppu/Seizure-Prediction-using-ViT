import os
import numpy as np
import matplotlib.pyplot as plt

def visualize(data_dir, num_samples=5):
    X_path = os.path.join(data_dir, 'X.npy')
    Y_path = os.path.join(data_dir, 'Y.npy')

    if not (os.path.exists(X_path) and os.path.exists(Y_path)):
        print(f"Data files not found in {data_dir}.")
        return
    
    X = np.load(X_path)
    Y = np.load(Y_path)

    print(f"Loaded data: X shape:{X.shape}, Y shape:{Y.shape}")
    Y_int = Y.astype(int)
    print(f"Label distribution: {np.bincount(Y_int)}")

    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    seizure_indices = np.where(Y == 1)[0]
    if(len(seizure_indices) > 0):
        seizure_samples = np.random.choice(seizure_indices, min(num_samples, len(seizure_indices)), replace=False)
        for i, ax in enumerate(axes[0, :min(num_samples, len(seizure_samples))]):
            ax.imshow(X[seizure_samples[i]].squeeze(), cmap='gray')
            ax.set_title(f"Seizures (Y=1)")
            ax.axis('off')

    nonseizure_indices = np.where(Y == 0)[0]
    if(len(nonseizure_indices) > 0):
        nonseizure_samples = np.random.choice(nonseizure_indices, min(num_samples, len(nonseizure_indices)), replace=False)
        for i, ax in enumerate(axes[1, :min(num_samples, len(nonseizure_samples))]):
            ax.imshow(X[nonseizure_samples[i]].squeeze(), cmap='viridis')
            ax.set_title(f"Non-Seizures (Y=0)")
            ax.axis('off')


    plt.tight_layout()
    plt.show()

    print(f"Value range: {X.min()} to {X.max()}")
    print(f"Mean: {X.mean()}, Std:: {X.std()}")

if __name__ == "__main__":
    data_dir = 'data/train'
    visualize(data_dir, num_samples=5)