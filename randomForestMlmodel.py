import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree #Plot tree sklearn module allows us to plot trees
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("formatedData.csv")

X,y = df.iloc[:,:8], df.iloc[:,-1]

xtrain, xtest, ytrain, ytest = train_test_split(X,y, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
#clf = BalancedRandomForestClassifier(random_state=42,sampling_strategy='all', replacement=True, bootstrap=False)
skf = StratifiedKFold(shuffle=True, random_state=42, n_splits=5)

#Creating lst for both the column and all the unique labels 
columnsLst = list(X.columns)
labelLst = list(y.unique())

"""
clf.fit(xtrain, ytrain)
ypredict = clf.predict(xtest)
print(classification_report(ytest, ypredict))

"""
f1Macroavg = 0
for train_index, test_index in skf.split(X,y): 
    xtrain = X.iloc[train_index,:8]
    xtest = X.iloc[test_index, :8]
    ytrain = y.iloc[train_index]
    ytest = y.iloc[test_index]

    clf.fit(xtrain, ytrain)
    ypredict = clf.predict(xtest)
    f1Macroavg += f1_score(y_true=ytest, y_pred=ypredict, average='macro')
    #confusionMatrix = confusion_matrix(ytest, ypredict)
    #ConfusionMatrixDisplay(confusionMatrix, display_labels=clf.classes_).plot()
    #plt.show()
print(f1Macroavg/5) #0.93268 vs 







"""
parameterDict = {'n_estimators':[10, 50, 100, 250, 500, 1000],
                 'max_depth':[5,7,9,11,13,15],
                'bootstrap':[True, False],
                'min_samples_split': [2,3,4,5,6],
             }

             
"""
"""
skf = StratifiedKFold(shuffle=True, n_splits=5,random_state=42)
gsc = GridSearchCV(clf, param_grid=parameterDict, scoring='f1_macro', verbose=2, cv = skf)
gsc.fit(X,y)


{'bootstrap': False, 'max_depth': 9, 'min_samples_split': 2, 'n_estimators': 10}
print(gsc.best_params_)






singularTree = clf.estimators_[0]
plt.figure(figsize=(20,8))
plot_tree(singularTree, feature_names=X.columns, class_names=y.unique())
plt.show()
"""