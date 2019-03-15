from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer

#specify dataset
cancer=load_breast_cancer()

#train/test split
#X_train, X_test, y_train, y_test = train_test_split(cancer.data.astype(np.float64),
#    cancer.target.astype(np.float64), train_size=0.75, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state = 12)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs = -1)
#n_jobs indicates the number of threads; -1 indicates "use all available"
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_cancer_pipeline.py')

#check output
accuracy_score(y_test, tpot.predict(X_test))
confusion_matrix(y_test, tpot.predict(X_test))

print(tpot)