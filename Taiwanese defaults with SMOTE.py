#Import some libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import preprocessing

#Change directory
import os
os.chdir('C:\Python36')

#Get Taiwanese defaults data
df = pd.read_excel('/Users/wiik/OneDrive/Njål J. Wiik/Utdanning/Handelshøyskolen BI/MSc in Business Analysis/2. Spring 2019/Predictive Analytics and Machine Learning/Notebooks/Lecture 8/default of credit card clients cleaned.xls')
X = df.copy(deep=True)

#isolate data from target
y = X.pop('default_next_month')

#split training and test data. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=2017, test_size = .25)

#choose model, applying pre-pruning 
clf = LogisticRegression(C=1.0).fit(X_train, y_train) 
#clf = KNeighborsClassifier(n_neighbors=5)
#clf=DecisionTreeClassifier(max_depth=6,random_state=0)

#fit model
clf.fit(X_train, y_train)

#used trained/fitted model to predict
y_model = clf.predict(X_test)

#assess results in several different ways 
#First, how accurate were we?
accuracy_score(y_test, y_model)
#but look closer
confusion_matrix(y_test, y_model)

#what's happening with logreg?
df["default_next_month"].mean() #unbalanced class problem

#Apply SMOTE to generate synthetic observations
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train) #Simulates the minority class

#use the same model as before, but re-fit using the upsampled data
clf2 = clf
clf2.fit(X_resampled, y_resampled)
y_model2 = clf2.predict(X_test)

confusion_matrix(y_test, y_model2)
accuracy_score(y_test, y_model2)

#Data from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
#Name: I-Cheng Yeh 
#email addresses: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw 
#institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan. 
#other contact information: 886-2-26215656 ext. 3181 

