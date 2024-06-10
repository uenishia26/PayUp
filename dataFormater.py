import pandas as pd
import os
import json 

# ---- *** Start of all utility functions for the dataFormater program *** ----
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
    for index in range(0,len(x1)):
        x1[index] = x1[index]/width
        x2[index] = x2[index]/width
        x3[index] = x3[index]/width 
        x4[index] = x4[index]/width
        y1[index] = y1[index]/height
        y2[index] = y2[index]/height
        y3[index] = y3[index]/height
        y4[index] = y4[index]/height 

def resetCoord():
    return [],[],[],[],[],[],[],[]

def appendToCSV(newdf, currentcsv):
    global firstTime 
    if firstTime: 
        newdf.to_csv(currentcsv, mode='a', index=False) #to_csv handles opening/closing of file 
        firstTime = False
    else: 
        newdf.to_csv(currentcsv, mode='a', header=False, index=False)

def deleteCSV():
    if os.path.exists("formatedData.csv"):
        os.remove("formatedData.csv")

#Adding class lst & fileName lst 
def finalizecsv(filenamelst, finalcsvname, labeltyp):
    df = pd.read_csv(finalcsvname)
    df.insert(0, 'filename', filenamelst)
    df['label'] = labelTyp 
    return df  
    


# ---- *** End of all utility functions for the dataFormater program *** ----

#---- *** Start of utility varaible names ---- ***
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
fileindex = 0
filenamelst = []
currentfilename = ""
firstTime = True #Prevents adding column name every append to existing dataFrame
finalcsvname = "formatedData.csv"

df = pd.read_csv("./data1.csv")  #*** Change file path *** 
df["region_shape_attributes"] = df.apply(jsonCoordinates, axis=1) #Applies function for every row 
df["region_attributes"] = df.apply(jsonLabel, axis=1)
deleteCSV()

#Getting labelType 
for row in df["region_attributes"]: 
    labelTyp.append(row["label"])

#filename/index setup 
filenamelst = df["filename"].tolist()
currentfilename = filenamelst[0]

for row in df["region_shape_attributes"]:
    """
        - Normalize the coordinates
        - Change the filename to current file name 
        - Reset x1..y4 lst 
    """
    if currentfilename != filenamelst[fileindex]: 
        currentfilename = filenamelst[fileindex] #Changing fileindex

        widthLst = [x1List[0]] * len(x1List) #Need to store width / height info before normalizing coordinates 
        heightLst = [y1List[0]] * len(y1List)
        normalizeCoordinates(x1List,y1List, x2List, y2List, x3List, y3List, x4List, y4List,
                            x1List[0], y1List[0])  
        csvDict = { #Creating Dictionary 
        "width": widthLst,
        "height": heightLst,
        "x1":x1List, "y1":y1List, "x2":x2List, "y2":y2List, "x3":x3List, "y3":y3List, "x4":x4List, "y4":y4List}

        newDF = pd.DataFrame(csvDict)
        appendToCSV(newDF,finalcsvname)

        x1List, y1List, x2List, y2List, x3List, y3List, x4List, y4List = resetCoord() #Resetting all coordinates

    #Appending ALL x,y coordinates for a specifc row 
    xList = row["all_points_x"]
    yList = row["all_points_y"]
    
    x1List.append(xList[0])
    y1List.append(yList[0])
    x2List.append(xList[1])
    y2List.append(yList[1])
    x3List.append(xList[2])
    y3List.append(yList[2])
    x4List.append(xList[3])
    y4List.append(yList[3])

    #Increasing fileIndex everyLoop 
    fileindex = fileindex + 1

    #Checking if we reached end of dataFrame (Appending the last set of Data)
    if fileindex == len(df): 
        widthLst = [x1List[0]] * len(x1List)
        heightLst = [y1List[0]] * len(y1List)
        normalizeCoordinates(x1List,y1List, x2List, y2List, x3List, y3List, x4List, y4List,
                            x1List[0], y1List[0])  
        csvDict = {
        "width": widthLst,
        "height": heightLst,
        "x1":x1List, "y1":y1List, "x2":x2List, "y2":y2List, "x3":x3List, "y3":y3List, "x4":x4List, "y4":y4List}
        newDF = pd.DataFrame(csvDict)
        appendToCSV(newDF,finalcsvname)
        x1List, y1List, x2List, y2List, x3List, y3List, x4List, y4List = resetCoord()


#Adding the name / label column (Finalizing the DataFrame)
finalDF = finalizecsv(filenamelst, finalcsvname, labelTyp)
finalDF = finalDF.drop(finalDF[(finalDF.label == 'dimension')].index)
finalDF = finalDF.drop(['filename', 'width', 'height'], axis=1)
finalDF.to_csv(finalcsvname)

