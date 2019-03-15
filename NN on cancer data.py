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

#create some lists to store results
training_accuracy = []
test_accuracy = []

#try neighbors 1-10
neighbors_settings = range(1, 25)

#split training and test data. 
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=2017)

#run NN model across the specified range
for num_neighbors in neighbors_settings:
	#specify/fit a model
	clf = KNeighborsClassifier(n_neighbors=num_neighbors)
	clf.fit(X_train, y_train)

	#store accuracy
	training_accuracy.append(clf.score(X_train, y_train))

	#record accuracy on holdout data
	test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

#choose "best" model from plot, and generation confusion matrix for that
#choose model, applying pre-pruning (max_depth = 4).
#We're committing a major sin here. What is it?
clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(X_train, y_train)
y_model = clf.predict(X_test)
clf.score(X_train, y_train)
accuracy_score(y_test, y_model)
confusion_matrix(y_test, y_model)

###############
#Generate predictions, then append to df, then write to Excel
results = X_test.copy(deep=True)
y_pred=clf.predict(X_test)
y_pred_prob= clf.predict_proba(X_test)
results['actual_benign'] = y_test
results['pred_benign'] = y_pred
results['pred_prob_benign'] = y_pred_prob[:,1]
results.to_excel(r'cancer_model.xls', header=True, index=True)