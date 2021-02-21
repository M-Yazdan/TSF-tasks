#!/usr/bin/env python
# coding: utf-8

# <h1>The Sparks Foundation<h1>
# <h2>Task#2: Prediction using Unsupervised ML<h2>
# <h4>From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.<h4>
# <h3>Author: Muhammad Yazdan<h3>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Importing Dataset

from sklearn import datasets
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data)
iris_data.columns = iris.feature_names
iris_data['Type']=iris.target
iris_data


# In[6]:


iris_X = iris_data.iloc[:, [0, 1, 2,3]].values


# In[7]:


iris_Y = iris_data['Type']
iris_Y = np.array(iris_Y)


# In[8]:


plt.scatter(iris_X[iris_Y == 0, 0], iris_X[iris_Y == 0, 1], s = 80, c = 'orange', label = 'Iris-setosa')
plt.scatter(iris_X[iris_Y == 1, 0], iris_X[iris_Y == 1, 1], s = 80, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(iris_X[iris_Y == 2, 0], iris_X[iris_Y == 2, 2], s = 80, c = 'green', label = 'Iris-virginica')
plt.title('IRIS dataset Clustering')
plt.legend()


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(iris_X)
    wcss.append(kmeans.inertia_)


# In[11]:


# Plot the Elbow method
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[14]:


# Modelling K-Means

cluster_Kmeans = KMeans(n_clusters=3)
model_kmeans = cluster_Kmeans.fit(iris_X)
pred_kmeans = model_kmeans.labels_
pred_kmeans


# In[15]:


# Plotting the Kmeans Clustering

plt.scatter(iris_X[pred_kmeans == 0, 0], iris_X[pred_kmeans == 0, 1], s = 80, c = 'orange', label = 'Iris-setosa')
plt.scatter(iris_X[pred_kmeans == 1, 0], iris_X[pred_kmeans == 1, 1], s = 80, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(iris_X[pred_kmeans == 2, 0], iris_X[pred_kmeans == 2, 2], s = 80, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
#plt.scatter(cluster_Kmeans.cluster_centers_[:, 0], cluster_Kmeans.cluster_centers_[:,1], s = 80, c = 'red', label = 'Centroids')
plt.title('Kmeans Clustering for IRIS dataset')
plt.legend()


# <h3>Thank you<h3>

# In[ ]:




