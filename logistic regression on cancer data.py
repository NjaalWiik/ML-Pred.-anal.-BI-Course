#import necessary modules
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#can also import from cross_validation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
import numpy as np
#specify dataset
cancer=load_breast_cancer()

#Choose and import the model family from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#split training and test data. 
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

#train the model using the default regularization parameter C=1. Try different parameters
#logreg = LogisticRegression(C=1.0).fit(X_train, y_train)

# logistic regression
logreg = LogisticRegression()

# Grid search cross validation
grid={"C":np.logspace(-4,4,30), "penalty":["l1","l2"]}# l1 lasso l2 ridge
#logreg=LogisticRegression(c=117.21022975334793, penalty="l1")
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x_train,y_train)
#logreg.fit(x_train,y_train)
#what are the best parameters?
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

# logistic regression
logreg2=LogisticRegression(C=logreg_cv.best_params_["C"], penalty=logreg_cv.best_params_["penalty"]])
logreg2.fit(x_train,y_train)

logreg2.score(x_train, y_train)
accuracy_score(y_test, y_model)
confusion_matrix(y_test, y_model)

#Thanks to Muller and Guido, 2017
#Note susceptibility to random_state; switch from 42 to 2017 and re-execute
