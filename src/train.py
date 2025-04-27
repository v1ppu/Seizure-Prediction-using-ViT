import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import EEGDataset
from model import EEGViTClassifier
from tqdm import tqdm
import numpy as np

#load preprocessed data
X = np.load('data/train/X.npy') # shape (N, 1, 256, 256)
Y = np.load('data/train/Y.npy')

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Number of classes: {np.unique(Y)}")

train_dataset = EEGDataset(X, Y, resize_to=(224, 224))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = EEGViTClassifier(img_size=(224,224), num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#training loop:

scaler = torch.amp.GradScaler('cuda')

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()   
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), 'model.pth')