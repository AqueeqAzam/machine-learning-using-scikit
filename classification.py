
 Classification is a very common problems in the real world. For example,
 we want to classify some products into good and bad quality, emails into good or junk,
 books into interesting or boring, and so on. As discussed before, two key factors make
 a problem into a classification problem, (1) the problem has correct answer (labels),
  and (2) the output we want is categorical data, such as Yes or No, or different categories
  Types of classification
  linear model -> logistic regression, support vector machine
  Non-linear models-> k-nearest, kerne--l, Naive bayes, Decision tree, Random forest


# Datasets -> A data set is a collection of numbers or values that relate to a particular subject.
# The number of fish eaten by each dolphin at an aquarium is a data set.
# label -> In Machine Learning feature means property of your training data.
# feature -> The output you get from your model after training it is called a label.
# f(feature) -> label, f(input) -> output, f(data) -> target
# predict gender as Male or Female, f(age, height, body_type) = f(22, 5.3, soft) -> femail

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

# import iris datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1, random_state=1)

# Gaussian naive bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)`
gnb_pred = gnb.predict(x_test)
# print('Accuracy of naive bayes is', accuracy_score(y_test, gnb_pred))

# Support vector machine
svm_clf = svm.SVC(kernel ='linear')

# train the model
svm_clf.fit(x_train, y_train)

# make prediction
svm_clf_pred = svm_clf.predict(x_test)

# predict the accuracy
# print('Accuracy of support vector machine:', accuracy_score(y_test, svm_clf_pred))

# Decision tree classifier
dt = DecisionTreeClassifier(random_state=0)

# train the model
dt.fit(x_train, y_train)

# make prediction

dt_pred = dt.predict(x_test)

# print the accuracy
print('Accuracy of decision tree:', accuracy_score(y_test, dt_pred))


# SVM using breast cancer
from sklearn import datasets
cancer_data = datasets.load_breast_cancer()
# print(cancer_data)
# print(cancer_data.data.shape)
# print(cancer_data.target)

# Spillating data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4, random_state=100)

# Generating the model
from sklearn import svm

# create a classifier
cls = svm.SVC(kernel = 'linear')

# train the model
cls.fit(x_train, y_train)
pred = cls.predict(x_test)

# Evaluate the model
from sklearn import metrics

# calculate accuracy
print('Accuracy:', metrics.accuracy_score(y_test, y_pred=pred))
print('Precision:', metrics.precision_score(y_test, y_pred=pred))
print('Recall:', metrics.recall_score(y_test, y_pred=pred))
print(metrics.classification_report(y_test, y_pred= pred))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
print(iris.target_names)
print(iris.feature_names)
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_pred = gnb.predict(x_test)
print("Accuracy of naive bayes is:", accuracy_score(y_test, gnb_pred))

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(x_train, y_train)
svm_clf_pred = svm_clf.predict(x_test)
print('Accuracy of support vector machine:', accuracy_score(y_test, svm_clf_pred))

dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)
print('Accuracy of decision tree clasifier:', accuracy_score(y_test, dt_pred))

clf = RandomForestClassifier(n_estimators= 100)
clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)
print('Accuracy socre of Random forest classifier:', accuracy_score(y_test, clf_pred))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
print('Accuracy score of KNN:', accuracy_score(y_test, knn_pred))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


# load data
wine = datasets.load_wine()

# Exploring data
# print(wine.values())
print(wine.feature_names)
print(wine.target_names)

# print the wine data (top 5 records)
# print(wine.data[0:5])

# check the records of target sets
print(wine.target)

# print data(feature)shape
print(wine.data.shape)

# print target(label)shape
print(wine.target.shape)

# import train_test_split function
from sklearn.model_selection import train_test_split

# split datasets into training set and test set
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# generating model k = 5
from sklearn.neighbors import KNeighborsClassifier

# create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# train the model using the training sets
knn.fit(x_train, y_train)

# predict the response for test datasets
y_pred = knn.predict(x_test)

# model evaluate for k = 5
from sklearn import metrics

# model acuracy, how ofter is the classifier correct
print('Accuracy of KNN:', accuracy_score(y_test, y_pred))
# print('Precision:', metrics.precision_score(y_test, y_pred))
# print('Recall:', metrics.recall_score(y_test, y_pred))
# print(metrics.classification_report(y_test, y_pred))




