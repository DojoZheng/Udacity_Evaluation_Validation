# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic-data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
features = X
labels = y
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=0)

clf1 = DecisionTreeClassifier()
clf1.fit(features_train, labels_train)
dt_f1 = f1_score(labels_test, clf1.predict(features_test))
print "Decision Tree F1 score: {:.2f}".format(dt_f1)
# clf1.fit(X, y)
# print "Decision Tree F1 score: {:.2f}".format(f1_score(y, clf1.predict(X)))

clf2 = GaussianNB()
clf2.fit(features_train, labels_train)
nb_f1 = f1_score(labels_test, clf2.predict(features_test))
print "GaussianNB F1 score: {:.2f}".format(nb_f1)
# clf2.fit(X, y)
# print "GaussianNB F1 score: {:.2f}".format(f1_score(y, clf2.predict(X)))

F1_scores = {
 "Naive Bayes": nb_f1,
 "Decision Tree": dt_f1
}