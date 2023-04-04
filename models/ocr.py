import cv2
import pytesseract

img = cv2.imread('../data/facebook/dev/hateful/01726.png')
img = cv2.bilateralFilter(img, 5, 55,60)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 240, 255, 1) 

custom_config = r'--oem 3 --psm 11'
text = pytesseract.image_to_string(img, config=custom_config)
print(text)

# https://towardsdatascience.com/extract-text-from-memes-with-python-opencv-tesseract-ocr-63c2ccd72b69