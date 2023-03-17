import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.hub import load
from torchvision import transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
import pandas as pd
from PIL import Image

SCALING_SIZE = (224, 224)
BATCH_SIZE = 128
MEAN = [0.485, 0.456, 0.406] # Required by resnet
SD = [0.229, 0.224, 0.225] # Required by resnet
MODEL = load('pytorch/vision:v0.10.0', 'resnet50', pretrained = True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
EPOCHS = 25

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = MEAN, std = SD)
])

train_img = datasets.ImageFolder('../data/facebook/img/train', transform = preprocess)
val_img = datasets.ImageFolder('../data/facebook/img/val', transform = preprocess)
train_loader = DataLoader(train_img, batch_size=BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_img, batch_size=BATCH_SIZE, shuffle = True)

num_features = MODEL.fc.in_features
MODEL.fc = nn.Linear(num_features, len(train_img.classes))
MODEL.to(DEVICE)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(MODEL.parameters(), lr = LEARNING_RATE)

for epoch in range(EPOCHS):
    MODEL.train()
    train_loss = 0
    train_correct = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = MODEL(images)
        _loss = loss(outputs, labels)
        _loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
    train_loss /= len(train_img)
    train_accuracy = 100*train_correct / len(train_img)


    MODEL.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = MODEL(images)
            _loss = loss(outputs, labels)
            val_loss += _loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
    val_loss /= len(val_img)
    val_accuracy = 100 * val_correct / len(val_img)

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train Loss: {train_loss:.4f} - "
          f"Train Acc: {train_accuracy:.2f}% - "
          f"Val Loss: {val_loss:.4f} - "
          f"Val Acc: {val_accuracy:.2f}%")