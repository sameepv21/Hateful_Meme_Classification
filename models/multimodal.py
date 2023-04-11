import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tqdm import tqdm

train_df = pd.read_json("../data/facebook/train.json")
dev_df = pd.read_json("../data/facebook/dev.json")

# Global Variable
BATCH_SIZE = 128
EPOCHS = 10
ROOT_PATH = '../data/facebook'
IMAGE_SIZE = 224*224
NUM_CLASSES = 2
TEXTUAL_DIMENSION = 512
VISUAL_DIMENSION = 512
CHECKPOINT = './model.pt'
train_loss = 0
train_acc = 0
dev_loss = 0
dev_acc = 0
highest_dev_acc = 0
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

# Define the transformation for preprocessing the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
        if 'label' not in self.df.columns:
            return image, text
        label = self.df.loc[index, 'label']

        return image ,text, label

# Create objects of each set of data
train_data = DynamicDataset(os.path.join(ROOT_PATH, 'train.json'), transform = transform)
dev_data = DynamicDataset(os.path.join(ROOT_PATH, 'dev.json'), transform = transform)

# Create a dataloader
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
dev_loader = DataLoader(dev_data, batch_size = BATCH_SIZE, shuffle = True)

# Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create Model
class MultiModal(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ResNet50 architecture
        resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

        convolution_layers = nn.Sequential(
            nn.Conv2d(3072, 2048, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(2048, 1024, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
        )

        # Freeze parameters
        for param in resnet50.parameters():
            param.requires_grad = False

        model_name = 'jiasenlu/vl-bert-base'
        self.resnet50 = AutoModel.from_pretrained(model_name)
        self.convolution_layers = convolution_layers

        # BERT
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        dense_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
        )
        self.dense_layers = dense_layers

        # Late Fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(VISUAL_DIMENSION, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, texts):
        # Extract visual features from images
        visual_features = self.convolution_layers(self.resnet50(images))
        visual_features = visual_features.view(visual_features.size(0), -1)
        
        # Extract textual features from texts
        input_ids = (tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')['input_ids']).to(device)
        textual_features = self.dense_layers(self.text_model(input_ids)[0][:, 0, :])
        
        # Concatenate visual and textual features
        fused_features = torch.cat((visual_features, textual_features), dim=1)
        
        # Fuse the multimodal features
        output = self.fusion_fc(fused_features)
        
        return output

# # Initialize the model    
# model = MultiModal()
# model.to(device)

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # Adaptive learning rate left

# # Load model from a previously saved checkpoint
# if os.path.exists(CHECKPOINT):
#     checkpoint = torch.load(CHECKPOINT, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     EPOCHS = EPOCHS - checkpoint['epoch']
#     train_loss = checkpoint['train_loss']
#     train_acc = checkpoint['train_acc']
#     dev_loss = checkpoint['dev_loss']
#     dev_acc = checkpoint['dev_acc']

# for epoch in range(EPOCHS):
#     try:
#         model.train()

#         for images, texts, labels in tqdm(train_loader):
#             images = images.to(device)
#             labels = labels.to(device)
#             labels = torch.reshape(labels, (-1, 1))
#             labels = labels.to(dtype = torch.float32)

#             optimizer.zero_grad()
#             outputs = model(images, texts)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item() * images.size(0)
#             train_acc += torch.sum(torch.max(outputs, dim = 1)[1] == labels)
        
#         train_loss = train_loss / len(train_data)
#         train_acc = train_acc / len(train_data)
#         print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}")
#         model.eval()
#         for images, texts, labels in tqdm(dev_loader):
#             images = images.to(device)
#             labels = labels.to(device)
#             labels = torch.reshape(labels, (-1, 1))
#             labels = labels.to(dtype = torch.float32)
            
#             outputs = model(images, texts)
#             loss = criterion(outputs, labels)
#             dev_los = loss.item() * images.size(0)
#             dev_acc += torch.sum(torch.max(outputs, dim = 1)[1] == labels)

#         dev_loss = dev_loss / len(dev_data)
#         dev_acc = dev_acc / len(dev_data)
#         print(f"Epoch {epoch+1}/{EPOCHS}: Dev Loss = {dev_loss:.4f}, Dev Accuracy = {dev_acc:.4f}")

#         if(highest_dev_acc < dev_acc):
#             highest_dev_acc = dev_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss': train_loss,
#                 'train_acc': train_acc,
#                 'dev_loss': dev_loss,
#                 'dev_acc': dev_acc,
#             }, CHECKPOINT)
#     except Exception as e:
#         print(e)
#         if(highest_dev_acc < dev_acc):
#             highest_dev_acc = dev_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss': train_loss,
#                 'train_acc': train_acc,
#                 'dev_loss': dev_loss,
#                 'dev_acc': dev_acc,
#             }, CHECKPOINT)
#         break