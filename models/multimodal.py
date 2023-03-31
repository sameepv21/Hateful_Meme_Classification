import pandas as pd
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


BATCH_SIZE = 128
EPOCHS = 10
ROOT_PATH = '../data/facebook'
IMAGE_SIZE = 224*224
NUM_CLASSES = 2

train_df = pd.read_json("../data/facebook/train.json")
test_df = pd.read_json("../data/facebook/test.json")
dev_df = pd.read_json("../data/facebook/dev.json")

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

class MultiModal(nn.Module):
    def __init__(self):
        super().__init__()