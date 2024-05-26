import requests 
import json 
from google.oauth2 import service_account
from google.cloud import vision
import os #Allows interaction with OS 
import io #Input/output operations 
from PIL import Image

#Function to remove exif
def removeExif(imgPath):
    image = Image.open(imgPath)
    data = list(image.getdata())
    imageWithoutExif = Image.new(image.mode, image.size)
    imageWithoutExif.putdata(data)
    imageWithoutExif.save(imgPath)

#Temprorary fix (Rotating Image)
def rotateImage(imgPath):
    with Image.open(imgPath) as img: #This will automatically close the img after opening
        rotated_img = img.rotate(90, expand=True) #CW .. expand adjusts frame of photo
        removeExif(path)
        rotated_img.save(imgPath)


#Creating cloud object to interact with GoogleVisionAPI (Setup for API interaction)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/anamuuenishi/Desktop/dataEntryEnv/dataentryautomation-793ea24c158f.json'
client = vision.ImageAnnotatorClient() 

#Editting/Sending img to GoogleVision OCR API
path = '/Users/anamuuenishi/Desktop/dataEntryEnv/Test3.jpg'
rotateImage(path)

with io.open(path, 'rb') as image_file: #read in binary mode 
    binaryImg = image_file.read()
clientImage = vision.Image(content=binaryImg) #Creating an image object 
response = client.text_detection(image=clientImage)
texts = response.text_annotations #Returns a strcutured return TextAnnotations object 



#Creating txt file for parsed OCR Data
f = open('/Users/anamuuenishi/Desktop/dataEntryEnv/data.txt', 'a')
for text in texts: 
    f.write(f"Description: {text.description}\n")
    f.write("Vertices:\n")
    for vertex in text.bounding_poly.vertices: 
        f.write(f"({vertex.x}, {vertex.y})\n")
    f.write("\n")


