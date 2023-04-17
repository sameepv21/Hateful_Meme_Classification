import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from multimodal import Fusion
from ocr import OCR
from PIL import Image
import os
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string

# Some global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./model.pt"
ROOT_PATH = "../data/facebook"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TEST_IMAGE = os.path.join(ROOT_PATH, 'test/01284.png')

def load_model():
    # Load the model
    model = Fusion()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    return model

def get_image_and_text(image_path = TEST_IMAGE):
    # Get the text and image
    image = Image.open(image_path)
    image = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Pass through ocr module
    ocr = OCR(image_path)
    text = ocr.detect_text()

    # Preprocess text
    text = text.lower()
    word_tokens = text.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in word_tokens]

    text = " ".join(stripped).strip()

    return image, [text]

def predict(model, image, text):
    # Pass through the model
    output = model(image, text)
    predicted = torch.round(torch.sigmoid(output))

    return predicted

def main(image_path = TEST_IMAGE):
    model = load_model()
    image, text = get_image_and_text(image_path)
    predicted = predict(model, image, text)

    return int(predicted.item())