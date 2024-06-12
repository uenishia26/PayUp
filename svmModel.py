import pandas as pd 
import numpy as np
from sklearn.svm import SVC #Suppoert vector Machine 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold






#Receiving / seprating Data 
df = pd.read_csv("formatedData.csv")
X = df.iloc[:,0:8]
y = df.iloc[:, 8]

#Creating the SVM classifier 
classifier = SVC(kernel='rbf', random_state=42, decision_function_shape='ovo')

#skf 
#nsplits=5, dataEntry=327 (Approx: 80% train, 20% test split)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
classireport = []
for train_index, test_index in skf.split(X,y): 
    print(f"training index: {train_index}")
    print(f"test index: {test_index}\n")
    Xtrain = X.iloc[train_index, :]
    Xtest = X.iloc[test_index, :]
    ytrain = y[train_index]
    ytest = y[test_index]
    classifier.fit(Xtrain, ytrain)
    ypred = classifier.predict(Xtest)
    print(classification_report(ytest, ypred))
    
#Regular testTrainSplit / testSize 20%
"""
xtrain, xtest, ytrain, ytest = train_test_split(X,y, random_state=0, test_size=0.2, stratify=y)
classifier = SVC(kernel='rbf', random_state=1, decision_function_shape='ovo')
classifier.fit(xtrain, ytrain)
ypredict = classifier.predict(xtest)
"""


"""
#[Item, itemprice, name, total]
print(classifier.classes_)
dfunction = classifier.decision_function(xtest)
print(dfunction)

print(ytest) #Predicts 52 as Item 
"""



"""
cm = confusion_matrix(ytest, ypred)
ConfusionMatrixDisplay(cm, display_labels=classifier.classes_).plot() #.classes_ shows all unique classes after fitting
plt.show() """


