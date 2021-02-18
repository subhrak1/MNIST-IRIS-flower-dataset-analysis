#!/usr/bin/env python
# coding: utf-8

# In[114]:


from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import display, HTML

from operator import itemgetter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

InteractiveShell.ast_node_interactivity = "all"
display(HTML("<style>.container { width:100% !important; }</style>"))

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid", font_scale=1.5)

plt.rcParams["figure.figsize"] = (14, 14)


# In[115]:


# First Data set: IRIS DATASET

columns = ["sepal length", "sepal width", "petal length", "petal width", "label"]
iris = pd.read_csv("iris.csv", names=columns)


# In[116]:


# Representing all 4 features of a dataset in a box plot column
sns.set_style("whitegrid")
sns.boxplot(data = iris);


# In[134]:


# 1.replacing all labels with integers
# 2.dropping last column
# 3.getting the label values

mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
df = iris.replace({"label": mapping}) 

X = df.iloc[:, :-1].to_numpy() 
y = df["label"].to_numpy() 

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)


# In[135]:


#KNN classifier algorithm
#Using both euclidean distance and minkowski distanace

#REFERENCE: 
# https://www.geeksforgeeks.org/minkowski-distance-python/
# professor's slides
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

def euclidean(r1, r2):
    dist = 0
    for (a, b) in zip(r1, r2):
        dist = dist + (a - b) ** 2
    return np.sqrt(dist)

def minkowski(r1, r2, p):
    def pRoot(val, r):
        root = 1 / float(r)
        return val ** root
    return (pRoot(sum(pow(abs(a - b), p) for (a, b) in zip(r1, r2)), p))

def get_neighbors(train_data, label_data, test_row, n_neighbors, metric, p=None):
    distance = []
    for (train_row, label_row) in zip(train_data, label_data):
        if metric == "euclidean":
            dist = euclidean(train_row, test_row)
        elif metric == "minkowski":
            dist = minkowski(train_row, test_row, p)
        distance.append((train_row, label_row, dist))
    
    distance.sort(key=itemgetter(2))
    neighbors = [np.append(distance[i][0], distance[i][1]) 
                 for i in range(n_neighbors)]
    return neighbors

def pred_classification(n):
    labels = [row[-1] for row in n]
    pred = max(set(labels), key=labels.count)
    return pred

def accuracy(actual, pred):
    c = 0
    for (a, b) in zip(actual, pred):
        if a == b:
            c = c + 1
    return c / float(len(actual))

def k_nearest_neighbors(train_data, label_data, test_data, n_neighbors, metric, p=None):
    prediction = []
    for test_row in test_data:
        neighbors = get_neighbors(train_data, label_data, test_row, n_neighbors, metric, p)
        pred = pred_classification(neighbors)
        prediction.append(pred)
    return np.array(prediction).astype(int)


# In[136]:


# decision boundaries
# training and test set

Xtrain = Xtrain[:, :2] 
Xtest = Xtest[:, :2]

y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1

h = 0.1
x2, y2 = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

k_VALUE = [1, 2, 4, 6, 10, 15]
mesh = np.c_[x2.ravel(), y2.ravel()]


# In[142]:


# PLOT USING EUCLIDEAN METRIC 

figure, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10, 15))
for x, k in zip(axs.flatten(), k_VALUE):    
    z = k_nearest_neighbors(Xtrain, ytrain, mesh, n_neighbors=k, metric="euclidean")
    z = z.reshape(x2.shape)
    x.contourf(x2, y2, z, alpha=0.5)
    x.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain, s=20, edgecolor= 'k')
    x.set(title="k=%d" % k, 
           xlim=(x2.min(), x2.max()), ylim=(y2.min(), y2.max()))
    
figure.tight_layout()
plt.setp(axs[-1, :], xlabel="sepal length")
plt.setp(axs[:, 0], ylabel="sepal width")
plt.show();


# In[121]:


# You can clearly see first hand when you see the red and yellow areas the classification changes drastically as you increase the 'k' value
# In k=15 you can see quite a lot of red dots are in the yellow area which shows misclassification. Which in turn is a wrong prediction
# Furthermore, as you increase the number KNN prediction area smooths out but in lower KNN value there is noise sensitivity. 


# In[143]:


# PLOT USING MINKOWSKI METRIC (p=3)

figure, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10, 15))
for x, k in zip(axs.flatten(), k_VALUE):
    z = k_nearest_neighbors(Xtrain, ytrain, mesh, n_neighbors=k, metric="minkowski", p=3)
    z = z.reshape(x2.shape)
    x.contourf(x2, y2, z, alpha=0.5)
    x.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain, s=40, edgecolor='k')
    x.set(title="k=%d" % k, 
           xlim=(x2.min(), x2.max()), ylim=(y2.min(), y2.max()))
    
figure.tight_layout()
plt.setp(axs[-1, :], xlabel="sepal length")
plt.setp(axs[:, 0], ylabel="sepal width")
plt.show();


# In[123]:


# If you look at the edges of the predicted areas and compare for both euclidean and minkowski you will find a change. Not significant.
# The edges on Minkowski metric will show you a less rigid picture in comparison to Euclidean metric. The example can be seen in k=10
# Since both calculate the distances differently it impacted on the accuracy of the classifier. 


# In[144]:


# ACCURACY TESTING (Euclidean and sklearn)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

val = []
val2 = []
kRange = range(1, 15)
for k in kRange:
    Z = k_nearest_neighbors(Xtrain, ytrain, Xtest, n_neighbors=k, metric="euclidean")
    val.append(accuracy(ytest, Z))
    
    knn = KNeighborsClassifier(n_neighbors=k, p=2)
    knn.fit(Xtrain, ytrain)
    val2.append(knn.score(Xtest, ytest))
    
s = {"Euclidean": val, "sklearn": val2}
pd.DataFrame.from_dict(s).plot(title="Accuracy using Euclidean metric");


# In[145]:


# ACCURACY TESTING (Minkowski and sklearn)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

val = []
val2 = []
kRange = range(1, 15)
for k in kRange:
    Z = k_nearest_neighbors(Xtrain, ytrain, Xtest, n_neighbors=k, metric="minkowski", p=3)
    val.append(accuracy(ytest, Z))
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain, ytrain)
    val2.append(knn.score(Xtest, ytest))
    
s = {"Minkowski": val, "sklearn": val2}
pd.DataFrame.from_dict(s).plot(title="Accuracy using Minkowski metric (p=3)");


# In[126]:


# Second Data set: MINST DATASET

mnist_train = pd.read_csv("mnist_train.csv", header=None)
mnist_train_label = mnist_train.iloc[:, 0]
mnist_test = pd.read_csv("mnist_test.csv", header=None)
mnist_test_label = mnist_test.iloc[:, 0]


# In[127]:


# using KNN from sklearn since our own implemented KNN takes too much time to process

samples = [500, 1000, 2500, 5000, 10000, 30000, 60000]

Xtest = mnist_test.sample(1000, random_state=42)
ytest = mnist_test_label.sample(1000, random_state=42)

val = []
for n in samples:
    Xtrain = mnist_train.sample(n, random_state=42)
    ytrain = mnist_train_label.sample(n, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(Xtrain, ytrain)
    val.append(knn.score(Xtest, ytest))
    
print(val);


# In[128]:


# plot the accuracy score
x = sns.barplot(x= samples, y=val, color="r")
x.set(xlabel="samples", ylabel="accuracy", title="uniform weighted voting", ylim=(0.8, 1))
plt.plot();


# In[129]:


# When you look at the plot the error slowly starts decreasing as the test samples start to increase
# The least error difference comes at 60000

n = 60000
Xtrain = mnist_train.sample(n, random_state=42)
ytrain = mnist_train_label.sample(n, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(Xtrain, ytrain)
yPred = knn.predict(Xtest)

print(accuracy(ytest, yPred))
print(classification_report(ytest, yPred))
cm = confusion_matrix(ytest, yPred)
print(cm)


# In[130]:


# Heatmap for confusion matrix
plt.figure(figsize=(10,15))
sns.heatmap(cm, annot=True, cmap="Greens");


# In[131]:


# KNN with distance-weighted voting

samples = [500, 1000, 2500, 5000, 10000, 30000, 60000]
samples = [n // 2 for n in samples]

Xtest = mnist_test.sample(1000, random_state=42)
ytest = mnist_test_label.sample(1000, random_state=42)

val2 = []
for n in samples:
    Xtrain = mnist_train.sample(n, random_state=42)
    ytrain = mnist_train_label.sample(n, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=2, weights="distance")
    knn.fit(Xtrain, ytrain)
    val2.append(knn.score(Xtest, ytest))
    
print(val2);


# In[132]:


# plot the accuracy score again
x = sns.barplot(x=samples, y=val2, color="b")
x.set(xlabel="samples", ylabel="accuracy", title="distance weighted voting", ylim=(0.8, 1))
plt.plot();


# In[133]:


# taking both 5000 and 30000 sample as comparison of accuracy

# 5000
# distance weighted has 93.9%
# uniform weighted has 91.6%

# 30000
# distance weighted has 96.4%
# uniform weighted has 95.8%

# Therefore in both cases distance weighted voting has more accuracy than uniform weighted voting. 

