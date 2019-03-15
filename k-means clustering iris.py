#First, let's visualize the raw iris data 
#   from:https://seaborn.pydata.org/generated/seaborn.pairplot.html
#   Is there a chance for clustering? If so, how many?
import seaborn as sns; sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)

#now, let's visualize knowing species -- obviously 3 groups once we know labels
g = sns.pairplot(iris, hue="species")

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

#Now, let's suppose we were totally ignorant, and do some k-means clustering

#importing the Iris dataset with pandas
iris = datasets.load_iris()
X = iris.data

#Finding the optimum number of clusters for k-means classification using the "elbow method" and weighted cluster sum of squares
#   Adopted from: https://www.kaggle.com/tonzowonzo/simple-k-means-clustering-on-the-iris-dataset
wcss = [] # Within cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

#Let's cluster
#   adopted from: https://towardsdatascience.com/clustering-based-unsupervised-learning-8d705298ae51

#KMeans based on number of clusters from elbow plot
km = KMeans(n_clusters=3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
km.fit(X)
km.predict(X)
labels = km.labels_

#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)

