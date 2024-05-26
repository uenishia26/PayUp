import requests 
import json 
from google.oauth2 import service_account
from google.cloud import vision
import os #Allows interaction with OS 
import io #Input/output operations 


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './dataentryautomation-793ea24c158f.json'
client = vision.ImageAnnotatorClient() #Creating cloud object to interact with GoogleVisionAPI

path = './output/image_6209779.jpg'
with io.open(path, 'rb') as image_file: #read in binary mode 
    binaryImg = image_file.read()
clientImage = vision.Image(content=binaryImg) #Creating an image object 
response = client.text_detection(image=clientImage)
texts = response.text_annotations #Returns a strcutured return TextAnnotations object 
print(texts[0])
print(texts[1])