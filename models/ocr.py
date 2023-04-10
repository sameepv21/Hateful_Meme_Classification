import os
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()
SERVICE_ACCOUNT_PATH = os.getenv('SERVICE_ACCOUNT_PATH')

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
client = vision.ImageAnnotatorClient(credentials=credentials)

class OCR:
    def __init__(self, image_path):
        self.image_path = image_path

    def detect_text(self):
        with open(self.image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations[0].description
        return texts.replace('\n', ' ')
    
ocr = OCR('../data/facebook/dev/hateful/92058.png')
print(ocr.detect_text())