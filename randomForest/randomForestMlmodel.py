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
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Integer, Categorical

df = pd.read_csv("formatedData.csv")

X,y = df.iloc[:,:8], df.iloc[:,-1]

xtrain, xtest, ytrain, ytest = train_test_split(X,y, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)


#Creating lst for both the column and all the unique labels 
columnsLst = list(X.columns)




""" Stratified K Fold testing 
f1Macroavg = 0
for train_index, test_index in skf.split(X,y): 
    xtrain = X.iloc[train_index,:8]
    xtest = X.iloc[test_index, :8]
    ytrain = y.iloc[train_index]
    ytest = y.iloc[test_index]
    clf.fit(xtrain, ytrain)
    confusionMatrix = confusion_matrix(ytest, ypredict)
    ConfusionMatrixDisplay(confusionMatrix, display_labels=clf.classes_).plot()
    plt.show()
"""


#(1) Grid Search CV
clf = RandomForestClassifier(class_weight= 'balanced', max_depth= 6, min_samples_split= 3,random_state=42, bootstrap=True, max_features='sqrt',n_estimators=250)
# GridSearch (HyperParameter tuning)
parameterDict = {
                    'max_depth':[5,6,7,8,9],
                    'min_samples_split': [2,3,4,5,6],
                    'class_weight':[None,'balanced','balanced_subsample']
                }     
#skf = StratifiedKFold(shuffle=True, n_splits=5,random_state=42)
#gsc = GridSearchCV(clf, param_grid=parameterDict, scoring='f1_macro', verbose=2, cv = skf)
clf.fit(xtrain,ytrain)
ypredict = clf.predict(xtest)
print(classification_report(ytest, ypredict))

"""
GridSearchCV
{{'class_weight': 'balanced', 'max_depth': 5, 'min_samples_split': 5, random_state=42, bootstrap=True, verbose=2, max_features='sqrt',n_estimators=250}}"""
"""
#randomSearch 
parameterDict = {
    'n_estimators': (5,250),
    'max_depth': (2,10),
    'min_samples_split': (2,20),
    'min_samples_leaf':(1,20),
    'max_features':['sqrt','log2'], 
    'bootstrap':[True, False], 
    'class_weight': ['None', 'balanced', 'balanced_subsample']
}
randomizedTune = RandomizedSearchCV(estimator=clf, scoring='f1_macro', cv=5, random_state=42, param_distributions=parameterDict,n_iter=10)
randomizedTune.fit(X)
"""

"""
#Bayses Optimization
parameterDict = {
    'n_estimators': Integer(5,250),
    'max_depth': Integer(2,10),
    'min_samples_split': Integer(2,20),
    'min_samples_leaf':Integer(1,20),
    'max_features':['sqrt','log2', None], 
    'bootstrap':[True, False],
    'criterion':['gini', 'entropy'],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'min_weight_fraction_leaf': Real(0.0,0.5)
}
baysOptimizer = BayesSearchCV(estimator=clf, scoring='f1_macro', cv=5, random_state=42, search_spaces=parameterDict, n_iter=300, verbose=1) #clf defined line 23
baysOptimizer.fit(xtrain, ytrain)
print(baysOptimizer.best_params_)
print(baysOptimizer.best_score_)
"""



"""
clf = RandomForestClassifier(bootstrap=True, class_weight= 'balanced', criterion= 'gini', max_depth= 7, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 6, min_weight_fraction_leaf= 0.0, n_estimators= 250)
clf.fit(xtrain,ytrain)
ypredict = clf.predict(xtest)
print(classification_report(ytest, ypredict))


"""






"""
#Tree printing for visualization
clf = RandomForestClassifier(random_state=42)
#clf = BalancedRandomForestClassifier(random_state=42,sampling_strategy='all', replacement=True, bootstrap=False)
clf.fit(xtrain, ytrain)
print(xtrain.info())
singularTree = clf.estimators_[98]

plt.figure(figsize=(20,8))
plot_tree(singularTree, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
#"""
