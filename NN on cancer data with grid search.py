#import necessary modules
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#can also import from cross_validation
#from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
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

#Run grid search to find parameters
param_grid = [{'n_neighbors': list(range(1,25))}] # params to try in the grid search
clf = KNeighborsClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=20, verbose=1, return_train_score = True)
grid_search.fit(X_train, y_train)
#what are the best parameters?
print(grid_search.best_params_)
#how should we expect this to do based on the validation scores?
print('''best score = {:.2f}'''.format(grid_search.best_score_))

#predict for the holdout, compute accuracy, and view the confusion matrix
y_model = grid_search.predict(X_test)
accuracy_score(y_test, y_model)
confusion_matrix(y_test, y_model)

#Thanks to Muller and Guido, 2017 
#and https://medium.com/@kathrynklarich/exploring-and-evaluating-ml-algorithms-with-the-wisconsin-breast-cancer-dataset-506194ed5a6a
