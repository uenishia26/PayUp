import pytesseract
import cv2
import re
import time
import numpy as np
#import matplotlib.pyplot as plt

# Path to the tesseract executable
def crop_image(image):
    xval = [498, 841]
    yval = [136, 2047]
    print(image.shape[0], image.shape[1])
    y = min(yval)
    x = min(xval)
    h = image.shape[0] - min(yval)
    w = max(xval)
    image = image[y:h, x:w]
    
    return image

def ouput_image(image):
    # Optionally, save or display the rotated image
    output_path = './testing/IMG_0193.jpeg'
    cv2.imwrite(output_path, image)
    print(f"Rotated image saved at {output_path}")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    image = crop_image(image)

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

def image_orient(path):
    try:
        image = preprocess_image(path)
        
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
                ouput_image(rotated_image)
            elif angle == 180 and confidence > 2.0:
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
                ouput_image(rotated_image)
            elif angle == 270 and confidence > 2.0:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                ouput_image(rotated_image)
            else:
                ouput_image(rotated_image)
        
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
    
img = image_orient('./Input/IMG_0193.jpeg')
