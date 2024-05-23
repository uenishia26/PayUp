import requests 
from google.oauth2 import service_account
from google.cloud import vision
import os #Allows interaction with OS 
import io #Input/output operations 


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/anamuuenishi/Desktop/dataEntryEnv/dataentryautomation-793ea24c158f.json'
client = vision.ImageAnnotatorClient() #Creating cloud object to interact with GoogleVisionAPI

path = '/Users/anamuuenishi/Desktop/dataEntryEnv/0001.jpg'
with io.open(path, 'rb') as image_file: #read in binary mode 
    binaryImg = image_file.read()
image = vision.Image(content=binaryImg) #Creating an image object 
response = vision.text_detection()




