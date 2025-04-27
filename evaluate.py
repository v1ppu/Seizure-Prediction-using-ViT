import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from src.dataset import EEGDataset
from src.model import EEGViTClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.ndimage

X_test = np.load('data/test/X.npy')
Y_test = np.load('data/test/Y.npy')

test_dataset = EEGDataset(X_test, Y_test, resize_to=(224, 224))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print(f"Test dataset contains {len(test_dataset)} samples with {len(test_loader)} batches")


model = EEGViTClassifier(img_size=(224, 224), num_classes=2) 
model.load_state_dict(torch.load('model.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
correct = 0
total = 0
batch_losses = []
batch_accuracies = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batch_losses.append(loss.item())
        batch_accuracy = (predicted == labels).sum().item() / labels.size(0)
        batch_accuracies.append(batch_accuracy)

print(f'Test Accuracy: {100 * correct / total:.2f}%')
avg_loss = np.mean(batch_losses)
print(f'Average Loss: {avg_loss:.4f}')

smooth_losses = scipy.ndimage.gaussian_filter1d(batch_losses, sigma=2)
smooth_accuracies = scipy.ndimage.gaussian_filter1d(batch_accuracies, sigma=2)

# Plot Loss
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(batch_losses, 'o-', alpha=0.3, color='blue', markersize=3, label='Raw data')
plt.plot(smooth_losses, '-', linewidth=2.5, color='darkblue', label='Smoothed')
plt.title(f'Test Loss per Batch (Avg: {avg_loss:.4f})', fontsize=12)
plt.xlabel('Batch')
plt.ylabel('Loss')
min_loss, max_loss = min(batch_losses), max(batch_losses)
plt.ylim(max(0, min_loss - (max_loss - min_loss) * 0.1), 
         max_loss + (max_loss - min_loss) * 0.1)  # Fixed this line
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(batch_accuracies, 'o-', alpha=0.3, color='green', markersize=3, label='Raw data')
plt.plot(smooth_accuracies, '-', linewidth=2.5, color='darkgreen', label='Smoothed')
plt.title(f'Test Accuracy per Batch (Overall: {100 * correct / total:.2f}%)', fontsize=12)
plt.xlabel('Batch')
plt.ylabel('Accuracy')
min_acc, max_acc = min(batch_accuracies), max(batch_accuracies)
padding = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 0.05
plt.ylim(max(0, min_acc - padding), min(1, max_acc + padding))
plt.grid(True, alpha=0.3)
plt.legend()

for ax in plt.gcf().get_axes():
    if len(batch_losses) > 20:
        step = len(batch_losses) // 10
        ax.set_xticks(range(0, len(batch_losses), step))

plt.tight_layout()
plt.savefig('test_performance.png')
plt.show()