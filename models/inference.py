import __main__
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytesseract
from PIL import Image
from multimodal import MultiModal

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# Load the model
model = MultiModal()
checkpoint = torch.load('./model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the image transforms
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the OCR function
def ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Define the inference function
def predict(image_path):
    # Extract the text from the image
    text = ocr(image_path)
    
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image_transforms(image).unsqueeze(0).to(device)
    
    # Pass the image through the model
    with torch.no_grad():
        output = model(image)
        output = F.sigmoid(output)
    
    # Return the prediction and the extracted text
    return output.item(), text

print(predict("/home/sameep/Extra_Projects/Hateful_Meme_Classification/data/facebook/test/01284.png"))