#import necessary modules
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#can also import from cross_validation
#from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_breast_cancer

#Choose and import the model family from scikit-learn
from sklearn.neighbors import KNeighborsClassifier

#specify dataset
cancer=load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df.head()
df['benign'] = cancer.target
df.head()

#isolate data from target
y = df.pop('benign')
X = df.copy(deep=True)

#split training and test data. 
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

#choose model
clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(X_train, y_train)
y_model = clf.predict(X_test)
clf.score(X_train, y_train)
accuracy_score(y_test, y_model)
confusion_matrix(y_test, y_model)

#Thanks to Muller and Guido, 2017
#Note susceptibility to random_state; switch from 42 to 2017 and re-execute

###############
#Generate predictions, then append to df, then write to Excel
results = X_test.copy(deep=True)
y_pred=clf.predict(X_test)
y_pred_prob= clf.predict_proba(X_test)
results['actual_benign'] = y_test
results['pred_benign'] = y_pred
results['pred_prob_benign'] = y_pred_prob[:,1]
results.to_excel(r'cancer_model.xls', header=True, index=True)

