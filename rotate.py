import pytesseract
import cv2
import re
import time
import numpy as np
#import matplotlib.pyplot as plt

# Path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def crop_image(image_path):
    image = cv2.imread(image_path)
    minval = 999999999
    maxval = 0
    y= image.shape[1] - 2578
    x= image.shape[0] - 4627
    h= image.shape[1] - 1697
    w= image.shape[0] - 1491
    crop_image = image[x:w, y:h]
    cv2.imshow("Cropped", crop_image)
    cv2.waitKey(0)
    return crop_image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Resize the image to improve OCR accuracy
    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    # Apply thresholding
    image = cv2.threshold(cv2.medianBlur(image, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
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

image_path = './output/IMG_0201.jpg'
try:
    image = preprocess_image(image_path)
    
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
    output_path = './output/rotated_image.jpg'
    cv2.imwrite(output_path, rotated_image)
    print(f"Rotated image saved at {output_path}")
    
except Exception as e:
    print("Take a better photo")
