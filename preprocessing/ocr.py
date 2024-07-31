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
import csv

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def crop_image(image, text):
    xval = []
    yval = []
    for vertex in text.bounding_poly.vertices:  
        xval.append(vertex.x)
        yval.append(vertex.y)
    y = min(yval)
    x = min(xval)
    h = image.shape[0] - min(yval)
    w = max(xval)
    image = image[y:h, x:w]
    return image

def ouput_image(image, path):
    # Optionally, save or display the rotated image
    cv2.imwrite(path, image)
    print(f"Rotated image saved at {path}")

def preprocess_image(image_path, text):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    image = crop_image(image, text)

    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[0]

    image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # Apply Gaussian blur
    image = cv2.bilateralFilter(image,9,75,75)

    # Define a kernel for morphological operations
    kernel = np.ones((1, 1), np.uint8)

    # Perform opening (erosion followed by dilation) to remove small dots
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Optionally, perform closing (dilation followed by erosion) to close small holes
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image

def image_orient(path, texts):
    try:
        ogimg = cv2.imread(path)
        ogimg = crop_image(ogimg, texts[0])
        image = preprocess_image(path, texts[0])
        
        start_time = time.time()
        imginfo = pytesseract.image_to_osd(image, config='--psm 0')
        elapsed_time = time.time() - start_time
        print(f"OCR processing time: {elapsed_time:.2f} seconds")
        
        angle = extract_info(imginfo, r'Orientation in degrees: \d+', "Angle in degrees")
        confidence = extract_info(imginfo, r'Orientation confidence: \d+\.\d+', "Confidence in degrees")
        if confidence:
            confidence = float(confidence)
        
        rotated_image = ogimg  # Default to the original image
        
        if angle and confidence:
            angle = int(angle)
            if angle == 90 and confidence > 2.0:
                rotated_image = cv2.rotate(ogimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ouput_image(rotated_image, path)
            elif angle == 180 and confidence > 2.0:
                rotated_image = cv2.rotate(ogimg, cv2.ROTATE_180)
                ouput_image(rotated_image, path)
            elif angle == 270 and confidence > 2.0:
                rotated_image = cv2.rotate(ogimg, cv2.ROTATE_90_CLOCKWISE)
                ouput_image(rotated_image, path)
            else:
                ouput_image(rotated_image, path)
        
    except Exception as e:
        print(f"Take a better photo {e}")
    
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
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'data/APIKeys/apiKey'
client = vision.ImageAnnotatorClient() 
foldername = './testing/'

with open('labeldata.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["word", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "label"]
    writer.writerow(field)

    for name in os.listdir(foldername):
        # Construct full file path
        path = os.path.join(foldername, name)
        # Check if the file is an image by checking its extension
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                # Open the image file
                with Image.open(path) as img:
                    print(f"Opened image: {path}")
                    with io.open(path, 'rb') as image_file: #read in binary mode 
                        binaryImg = image_file.read()
                    clientImage = vision.Image(content=binaryImg) #Creating an image object 
                    response = client.text_detection(image=clientImage)
                    texts = response.text_annotations #Returns a strcutured return TextAnnotations object 
                    img = image_orient(path,texts)
            except Exception as e:
                print(f"Could not open image {path}. Error: {e}")

        with io.open(path, 'rb') as image_file: #read in binary mode 
            binaryImg = image_file.read()
        clientImage = vision.Image(content=binaryImg) #Creating an image object 
        response = client.text_detection(image=clientImage)
        texts = response.text_annotations #Returns a strcutured return TextAnnotations object 

    # #Creating txt file for parsed OCR Data
        ocrdata = {}
        f = open('./data.txt', 'a', encoding="utf-8", errors="replace")
        for text in texts: 
            f.write(f"Description: {text.description}\n")
            f.write("Vertices:\n")
            tmp = []
            csvrows = [text.description]
            for vertex in text.bounding_poly.vertices: 
                f.write(f"({vertex.x}, {vertex.y})\n")
                tmp.append(vertex.x)
                tmp.append(vertex.y)
                csvrows.append(vertex.x/img.shape[1])
                csvrows.append(vertex.y/img.shape[0])
            ocrdata[text.description] = tmp
            writer.writerow(csvrows)
            f.write("\n")

