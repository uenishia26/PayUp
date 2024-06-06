import requests 
import json 
from google.oauth2 import service_account
from google.cloud import vision
import os #Allows interaction with OS 
import io #Input/output operations 
from PIL import Image
import pytesseract
import cv2
import re
import time
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def crop_image(image, text):
    if (image.shape[0] > 1000 and image.shape[1] > 1000):
        xval = []
        yval = []
        for vertex in text.bounding_poly.vertices:  
            xval.append(vertex.x)
            yval.append(vertex.y)
        y = image.shape[1] - max(yval)
        x = min(xval)
        h = image.shape[1] - min(yval)
        w = max(xval)
        image = image[x:w, y:h]
    return image

def preprocess_image(image_path, text):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)

     # Apply thresholding
    image = cv2.threshold(cv2.medianBlur(image, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    image = crop_image(image, text)
    # Resize the image to improve OCR accuracy
    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    return image

def extract_info(imginfo, pattern, label):
    match = re.search(pattern, imginfo)
    if match:
        result = match.group().split(':')[-1].strip()
        print(f"{label}: {result}")
        return result
    else:
        print(f"No {label.lower()} found.")
        return None

# #Function to remove exif
# def removeExif(imgPath):
#     with Image.open(imgPath) as image: 
#         data = list(image.getdata())
#         imageWithoutExif = Image.new(image.mode, image.size)
#         imageWithoutExif.putdata(data)
#         imageWithoutExif.save(imgPath)

# def findMidPoint(vertexLst):
#     sumX = None
#     sumY = None
#     for vertex in vertexLst:
#         sumX = sumX + vertex.x
#         sumY = sumY + vertex.y
#     midX = sumX / vertexLst
#     midY = sumY / vertexLst
#     return (midX, midY)
    

# #Temprorary fix (Rotating Image)
# def rotateImage(imgPath):
#     with Image.open(imgPath) as img: #This will automatically close the img after opening
#         rotated_img = img.rotate(-90, expand=True) #CW .. expand adjusts frame of photo
#         rotated_img.save(imgPath)
#         removeExif(path)


#Creating cloud object to interact with GoogleVisionAPI (Setup for API interaction)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './dataentryautomation-793ea24c158f.json'
client = vision.ImageAnnotatorClient() 

#Editting/Sending img to GoogleVision OCR API
name = 'IMG_0192.jpg'
path = './input/' + name


with io.open(path, 'rb') as image_file: #read in binary mode 
    binaryImg = image_file.read()
clientImage = vision.Image(content=binaryImg) #Creating an image object 
response = client.text_detection(image=clientImage)
texts = response.text_annotations #Returns a strcutured return TextAnnotations object 

try:
    image = preprocess_image(path, texts[0])
    
    start_time = time.time()
    imginfo = pytesseract.image_to_osd(image, config='--psm 0')
    elapsed_time = time.time() - start_time
    print(f"OCR processing time: {elapsed_time:.2f} seconds")
    
    angle = extract_info(imginfo, r'Orientation in degrees: \d+', "Angle in degrees")
    confidence = extract_info(imginfo, r'Orientation confidence: \d+\.\d+', "Confidence in degrees")
    if confidence:
        confidence = float(confidence)
    
    rotated_image = image  # Default to the original image
    
    if angle and confidence:
        angle = int(angle)
        if angle == 90 and confidence > 2.0:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180 and confidence > 2.0:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270 and confidence > 2.0:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            if (confidence < 2.0):
                print("Take a better photo")
            else:
                print("Image is at the correct angle")
    
    # Optionally, save or display the rotated image
    output_path = './output/' + name
    cv2.imwrite(output_path, rotated_image)
    print(f"Rotated image saved at {output_path}")
    
except Exception as e:
    print(f"Take a better photo {e}")


#Creating txt file for parsed OCR Data
f = open('./data.txt', 'a')
for text in texts: 
    f.write(f"Description: {text.description}\n")
    f.write("Vertices:\n")
    for vertex in text.bounding_poly.vertices: 
        f.write(f"({vertex.x}, {vertex.y})\n")
    f.write("\n")