import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier

df = pd.read_csv("./formatedData.csv") 

x = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y, shuffle=True)

# GaussianNB Model
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gnb = GaussianNB()
gs_NB = GridSearchCV(estimator=gnb,  param_grid=params_NB, scoring='recall_macro')

gs_NB.fit(X_train, y_train)
y_pred = gs_NB.predict(X_test)
print("Hyperparameters:",gs_NB.best_params_)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=gs_NB.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gs_NB.classes_)
disp.plot()
plt.show()

# MultinomialNB Model with onevone

clf = OneVsOneClassifier(MultinomialNB(alpha=0))

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()