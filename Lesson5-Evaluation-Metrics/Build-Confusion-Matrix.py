# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic-data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
features = X
labels = y
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.25, random_state=0)

clf1 = DecisionTreeClassifier()
clf1.fit(features_train, labels_train)
dt_confusion_matrix = confusion_matrix(labels_test, clf1.predict(features_test))
print "Confusion matrix for this Decision Tree:\n", dt_confusion_matrix
# clf1.fit(X,y)
# print "Confusion matrix for this Decision Tree:\n",confusion_matrix(y,clf1.predict(X))

clf2 = GaussianNB()
clf2.fit(features_train, labels_train)
g_confusion_matrix = confusion_matrix(labels_test, clf2.predict(features_test))
print "GaussianNB confusion matrix:\n", g_confusion_matrix
# clf2.fit(X,y)
# print "GaussianNB confusion matrix:\n",confusion_matrix(y,clf2.predict(X))

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": dt_confusion_matrix,
 "Decision Tree": g_confusion_matrix
}