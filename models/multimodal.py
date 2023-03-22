import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torchvision import *
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os


class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id = self.data.loc[index, 'id']
        image_file = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_file).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        text = self.data.loc[index, 'text']
        label = self.data.loc[index, 'label']

        return image, text, label


class LateFusionModel(nn.Module):
    def __init__(self, visual_input_dim, text_input_dim, hidden_dim, num_classes, dropout):
        super().__init__()

        # Resnet50 for visual feature extraction
        self.visual_model = models.resnet50(pretrained=True)
        self.visual_fc = nn.Linear(visual_input_dim, hidden_dim)

        # Pre-trained BERT for textual feature extraction
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(text_input_dim, hidden_dim)

        # Late fusion
        self.fusion_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_input, text_input):
        # Visual feature extraction
        visual_output = self.visual_model(visual_input)
        visual_output = visual_output.view(visual_output.size(0), -1)
        visual_output = self.dropout(F.relu(self.visual_fc(visual_output)))

        # Textual feature extraction
        text_output = self.text_model(text_input)[1]
        text_output = self.dropout(F.relu(self.text_fc(text_output)))

        # Late fusion
        fusion_output = torch.cat((visual_output, text_output), dim=1)
        fusion_output = self.dropout(F.relu(self.fusion_fc1(fusion_output)))
        fusion_output = self.fusion_fc2(fusion_output)

        return fusion_output


# Define data directories and files
image_dir = '../data/facebook/img/train'
train_csv = '../data/facebook/train.csv'
val_csv = '../data/facebook/val.csv'

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Define data loaders for training and testing
train_data = CustomDataset(image_dir, train_csv, transform)
test_data = CustomDataset(image_dir, val_csv, transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Define the model
model = LateFusionModel(visual_input_dim=2048, text_input_dim=768, hidden_dim=512, num_classes=2, dropout=0.5)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for images, texts, labels in train_loader:
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_acc += torch.sum(torch.max(outputs, dim=1)[1] == labels)

    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}")

    # Evaluate the model
    test_loss = 0.0
    test_acc = 0.0
    model.eval()
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            test_acc += torch.sum(torch.max(outputs, dim=1)[1] == labels)

    test_loss = test_loss / len(test_data)
    test_acc = test_acc / len(test_data)

    print(f"Epoch {epoch+1}/{num_epochs}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")