import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from transformers import BertTokenizer, VisualBertModel, logging
from PIL import Image
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

train_df = pd.read_json("/kaggle/input/facebook-hmcwa/facebook/train.json")
dev_df = pd.read_json("/kaggle/input/facebook-hmcwa/facebook/dev.json")
train_df.head()

# Some global variables
BATCH_SIZE = 128
EPOCHS = 5
ROOT_PATH = '/kaggle/input/facebook-hmcwa/facebook'
IMAGE_SIZE = 224*224
NUM_CLASSES = 2
TEXTUAL_DIMENSION = 512
VISUAL_DIMENSION = 512
CHECKPOINT = '/kaggle/working/model.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else'cpu')

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
    
class Visual_Feature(nn.Module):
    def __init__(self):
        super().__init__()

        # Define resnet50 model
        resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        convolution_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
        )
        
        # Freeze parameters
        for param in resnet50.parameters():
            param.requires_grad = False

        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        self.convolution_layers = convolution_layers

    def get_visual_features(self, images):
        # Extract visual features from resnet50 model
        visual_features = self.convolution_layers(self.resnet50(images))
        visual_features = visual_features.view(visual_features.size(0), -1)

        return visual_features
    
class Textual_Feature(nn.Module):
    def __init__(self):
        super().__init__()

        # Define virtual bert model
        visual_bert = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa')
        dense_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
        )
        
#         # Freeze parameters
#         for param in visual_bert.parameters():
#             param.requires_grad = False

        self.visual_bert = visual_bert
        self.dense_layers = dense_layers

        # Define tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def get_textual_features(self, texts):
        # Define indices and attention mask
        inputs = self.tokenizer.batch_encode_plus(texts, padding = True, return_tensors = 'pt')
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)

        # Extract textual features from virtual bert model
        textual_features = self.visual_bert(input_ids = input_ids, attention_mask = attention_mask, return_dict = False)
        textual_features = textual_features[0][:, 0, :] # Extract the first token of last hidden state
        textual_features = self.dense_layers(textual_features)

        return textual_features
    
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Define fusion layers
        fusion_layers = nn.Sequential(
            nn.Linear((VISUAL_DIMENSION + TEXTUAL_DIMENSION), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.fusion_layers = fusion_layers
    
    def forward(self, images, texts):
        # Initialize text and visual classes
        visual_class = Visual_Feature().to(DEVICE)
        textual_class = Textual_Feature().to(DEVICE)

        # Extract visual and textual features
        visual_features = visual_class.get_visual_features(images)
        textual_features = textual_class.get_textual_features(texts)

        # Concatenate visual and textual features
        features = torch.cat((visual_features, textual_features), dim = 1)

        # Pass through fusion layers
        output = self.fusion_layers(features)

        return output