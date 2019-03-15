#import necessary modules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
#from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV

#specify dataset
cancer=load_breast_cancer()

#split training and test data. 
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=2017)



#choose model, applying pre-pruning (max_depth = 2)
#
tree=DecisionTreeClassifier(max_depth=5,random_state=0)


#Run grid search to find parameters
param_grid = [{'max_depth': list(range(1,10))}] # params to try in the grid search
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=20, verbose=1, return_train_score = True)
grid_search.fit(X_train, y_train)
#what are the best parameters?
print(grid_search.best_params_)
#how should we expect this to do based on the validation scores?
print('''best score = {:.2f}'''.format(grid_search.best_score_))


accuracy_score(y_test, y_model)
confusion_matrix(y_test, y_model)
#fit model
tree.fit(X_train, y_train)

#used trained/fitted model to predict
y_model = tree.predict(X_test)

#plot tree: http://webgraphviz.com/
export_graphviz(tree, out_file="tree.gv", class_names=["malignant", "benign"],feature_names=cancer.feature_names, impurity=False, filled=True)

#assess results in several different ways 
accuracy_score(y_test, y_model)
confusion_matrix(y_test, y_model)
confusion_matrix(y_test, y_model)