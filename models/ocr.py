import cv2
import pytesseract

img = cv2.imread('../data/facebook/dev/hateful/01726.png')
custom_config = r'--psm 12'

print(pytesseract.image_to_string(img, lang = 'eng', config=custom_config))