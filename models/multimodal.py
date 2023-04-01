import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_json("../data/facebook/train.json")
test_df = pd.read_json("../data/facebook/test.json")
dev_df = pd.read_json("../data/facebook/dev.json")

# Vectorie textual input
vectorizer = TfidfVectorizer(min_df = 0.05, sublinear_tf = True)
tfidf_scores_train = vectorizer.fit_transform(train_df['text'])
tfidf_scores_test = vectorizer.transform(test_df['text'])
tfidf_scores_dev = vectorizer.fit_transform(dev_df['text'])

# Global Variable
BATCH_SIZE = 128
EPOCHS = 10
ROOT_PATH = '../data/facebook'
IMAGE_SIZE = 224*224
NUM_CLASSES = 2
TEXTUAL_DIMENSION = tfidf_scores_train.shape[1] # No. of Unique words in tfidf score

# Initialize the dataset and maintain the dataloader
class DynamicDataset(Dataset):
    def __init__(self, json_path, transform = None):
        self.df = pd.read_json(json_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.loc[index, 'img']
        img_file = os.path.join(ROOT_PATH, img_path)
        image = Image.open(img_file).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        text = self.df.loc[index, 'text']
        label = self.df.loc[index, 'label']

        return image ,text, label

# Create objects of each set of data
train_data = DynamicDataset(os.path.join(ROOT_PATH, 'train.json'), transform = transforms.ToTensor())
dev_data = DynamicDataset(os.path.join(ROOT_PATH, 'dev.json'), transform = transforms.ToTensor())
test_data = DynamicDataset(os.path.join(ROOT_PATH, 'test.json'), transform = transforms.ToTensor())

# Create a dataloader
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
dev_loader = DataLoader(dev_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

# Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create Model
class MultiModal(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        # ResNet50 architecture
        self.visual_model = models.resnet50(pretrained = True)
        self.visual_fc = nn.Linear(IMAGE_SIZE, hidden_dim)

        # BERT
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(TEXTUAL_DIMENSION, hidden_dim)

        # Late Fusion
        self.fusion_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, NUM_CLASSES)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

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
    
model = MultiModal(hidden_dim = 512)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # Adaptive learning rate left

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
model.to(device)

for epoch in range(EPOCHS):
    train_loss = 0
    train_acc = 0
    dev_loss = 0
    dev_acc = 0

    model.train()

    for images, texts, labels in train_loader:
        images = images.to(device)
        texts = tokenizer(texts).to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_acc += torch.sum(torch.max(outputs, dim = 1)[1] == labels)

    model.eval()
    for images, text, labels in dev_loader:
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        dev_los = loss.item() * images.size(0)
        dev_acc += torch.sum(torch.max(outputs, dim = 1)[1] == labels)

    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    dev_loss = dev_loss / len(dev_data)
    dev_acc = dev_acc / len(dev_data)

    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}")
    print(f"Epoch {epoch+1}/{EPOCHS}: Dev Loss = {dev_loss:.4f}, Dev Accuracy = {dev_acc:.4f}")

    # Evaluate the model
    test_loss = 0
    test_acc = 0
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

    print(f"Epoch {epoch+1}/{EPOCHS}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")