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

#Temprorary fix 
#Temprorary fix (Rotating Image)
def rotateImage(imgPath):
    with Image.open(imgPath) as img: #This will automatically close the img after opening
        rotated_img = img.rotate(90, expand=True) #CW .. expand adjusts frame of photo
        removeExif(path)
        rotated_img.save(imgPath)


#Midpoint Function 
def findMidPoint(vertexLst):
    sumX = 0
    sumY = 0
    for vertex in vertexLst:
        sumX = sumX + vertex.x
        sumY = sumY + vertex.y
    midX = sumX / len(vertexLst)
    midY = sumY / len(vertexLst)
    return (midX, midY)

def getOCRdata(imgPath):
    with io.open(imgPath, 'rb') as image_file: #read in binary mode 
        binaryImg = image_file.read()

    clientImage = vision.Image(content=binaryImg) #Creating an image object 
    response = client.text_detection(image=clientImage)
    texts = response.text_annotations #Returns a strcutured return TextAnnotations object 

    #Creating txt file for parsed OCR Data
    f = open('/Users/anamuuenishi/Desktop/dataEntryEnv/data.txt', 'a')
    ocrData = []
    for text in texts: 
        vertices = text.bounding_poly.vertices
        ocrData.append = [{'item':text.description, 
                           'x1': vertices[0].x, 'y1': vertices[0].y,
                           'x2': vertices[1].x, 'y2':vertices[1].y,
                           'x3':vertices[2].x, 'y3':vertices[2].y, 
                           'x4':vertices[3].x, 'y4':vertices[3].y}]
    return ocrData 

def normalizeData(ocrData, width, height):
    for eachData in ocrData: 
        eachData['x1'] /= width
        eachData['x2'] /= width 
        eachData['x3'] /= width 
        eachData['x4'] /= width
        eachData['y1'] /= height
        eachData['y2'] /= height
        eachData['y3'] /= height
        eachData['y4'] /= height

    return ocrData





#Creating cloud object to interact with GoogleVisionAPI (Setup for API interaction)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/anamuuenishi/Desktop/dataEntryEnv/dataentryautomation-793ea24c158f.json'
client = vision.ImageAnnotatorClient() #Creating cloud object to interact with GoogleVisionAPI



#Editting/Sending img to GoogleVision OCR API
path = '/Users/anamuuenishi/Desktop/dataEntryEnv/MLModelTrainImages/Test1.jpg'
ocrData = getOCRdata(path)





'''
#Finding Midpoints 
text_midpoints = {}
for text in texts: 
    key = text.description
    val = findMidPoint(text.bounding_poly.vertices)
    text_midpoints[key] = val

print(text_midpoints)
'''