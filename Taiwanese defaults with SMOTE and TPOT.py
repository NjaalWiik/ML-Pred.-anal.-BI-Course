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
from tpot import TPOTClassifier


#Change directory
import os
os.chdir('C:\Python36')

#Get Taiwanese defaults data
df = pd.read_excel('default of credit card clients cleaned.xls')
X = df.copy(deep=True)

#isolate data from target
y = X.pop('default_next_month')

#split training and test data. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=2017, test_size = .25)

#Apply SMOTE to generate synthetic observations
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)

#choose model, applying pre-pruning 
#logclf = LogisticRegression(C=1.0).fit(X_train, y_train) 
#clf = KNeighborsClassifier(n_neighbors=5)
#clf=DecisionTreeClassifier(max_depth=6,random_state=0)
clf = TPOTClassifier(generations=5, population_size=50, verbosity=2)

#fit model
clf.fit(X_resampled, y_resampled)
#logclf.fit(X_resampled, y_resampled)

#used trained/fitted model to predict
y_model = clf.predict(X_test)
#log_model = logclf.predict(X_test)

confusion_matrix(y_test, y_model)
#confusion_matrix(y_test, log_model)

#Data from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
#Name: I-Cheng Yeh 
#email addresses: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw 
#institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan. 
#other contact information: 886-2-26215656 ext. 3181 

clf.fitted_pipeline_