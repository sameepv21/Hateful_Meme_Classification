from PIL import Image
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

#Define path to image
path_to_image = '/home/sameep/Extra_Projects/Hateful_Meme_Classification/data/facebook/test/01284.png'

img1 = Image.open(path_to_image)
text = pytesseract.image_to_string(img1)

print("Result: ", text)