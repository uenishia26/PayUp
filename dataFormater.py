import pandas as pd
import os
import json 

def jsonCoordinates(row):  #Convert each row of region_shape_attribute to a JSON string Python (Dict)
    if isinstance(row["region_shape_attributes"],str): #If jsonString return Dict else return Dict
        return json.loads(row["region_shape_attributes"])
    return row["region_shape_attributes"] 
    
def jsonLabel(row):
    if isinstance(row["region_attributes"], str):
        return json.loads(row["region_attributes"])
    return row["region_attributes"]

"""def setDimension(): #Custom Coordinates (X1,Y1) width/height from top left origin (0,0)
    width = 2312
    height = 5551"""

def normalizeCoordinates(x1,y1,x2,y2,x3,y3,x4,y4, width, height):
    for index in range(len(x1)):
        x1[index] = x1[index]/width
        x2[index] = x2[index]/width
        x3[index] = x3[index]/width 
        x4[index] = x4[index]/width
        y1[index] = y1[index]/height
        y2[index] = y2[index]/height
        y3[index] = y3[index]/height
        y4[index] = y4[index]/height 

df = pd.read_csv("./data1.csv")
df["region_shape_attributes"] = df.apply(jsonCoordinates, axis=1) #Applies function for every row (axis=1 is row)
df["region_attributes"] = df.apply(jsonLabel, axis=1)

width = 2312 #Width/Height of receipt 
height = 5551


#All necessary lst for dp columns 
x1List = []
y1List = []
x2List = []
y2List = []
x3List = []
y3List = []
x4List = []
y4List = []
labelTyp = []
fileName = df["filename"].tolist() #Convert Series to lst

#Iterate over coordinate column 
for row in df["region_shape_attributes"]:
    xList = row["all_points_x"]
    yList = row["all_points_y"]
    
    x1List.append(xList[0])#Appending All Normalized Coordinates
    y1List.append(yList[0])
    x2List.append(xList[1])
    y2List.append(yList[1])
    x3List.append(xList[2])
    y3List.append(yList[2])
    x4List.append(xList[3])
    y4List.append(yList[3])

#Iterate over label column 
for row in df["region_attributes"]:
    labelTyp.append(row['label'])

#Setting the width and height 
widthLst = [width] * len(df)
heightLst = [height] * len(df)

#Normalize all coordinates by dividing by img width & height

normalizeCoordinates(x1List,y1List, 
                    x2List, y2List, 
                    x3List, y3List, 
                    x4List, y4List, widthLst[0], heightLst[0])

#Creating new CSV file w/h only important data 
csvDict = {"filename":fileName, 
           "label": labelTyp, 
           "width": widthLst,
           "height": heightLst,
           "x1":x1List, "y1":y1List, "x2":x2List, "y2":y2List, "x3":x3List, "y3":y3List, "x4":x4List, "y4":y4List}
newDF = pd.DataFrame(csvDict)
print(newDF)

